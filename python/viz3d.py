import os
import re
import sys
import cv2
import tqdm
import glob
import torch
import bisect
import argparse
import numpy as np
import keyboard as kbd
import torchvision as tv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_agg import FigureCanvasAgg
from multiprocessing import Pool
from mpl_toolkits import mplot3d
from src.agents import get_config, all_refs, all_agents
from src.agents.pytorch.rl.base import PTNetwork
from src.utils.rand import RandomAgent
from src.utils.logger import Logger
from src.utils.config import Config
from src.utils.misc import make_video, resize
np.set_printoptions(precision=3, sign=" ")
	
class InputController(RandomAgent):
	def __init__(self, state_size, action_size, config=None, **kwargs):
		self.state_size = state_size
		self.action_size = action_size

	def get_action(self, state, eps=0.0, sample=False):
		shape = state.shape[:-len(self.state_size)]
		action = np.zeros([*shape, *self.action_size])
		try:
			if kbd.is_pressed("left"):
				action[...,0] += 1
			if kbd.is_pressed("right"):
				action[...,0] -= 1
			if kbd.is_pressed(kbd.KEY_UP):
				action[...,1] += 1
			if kbd.is_pressed(kbd.KEY_DOWN):
				action[...,1] -= 1
		except Exception as e:
			print(e)
		return action

class ImageDataset(torch.utils.data.Dataset): 
	def __init__(self, config, data_dir, buffer_size=1000000, train=True): 
		self._files = sorted([os.path.join(data_dir, f) for f in glob.glob(f"{data_dir}/**/*.npz", recursive=True)])
		self._files = train_test_split(self._files, train_size=0.8, shuffle=True, random_state=0)[1-int(train)]
		self._cum_size = None
		self._buffer = None
		self._buffer_fnames = None
		self._buffer_index = 0
		self._buffer_size = buffer_size
		self.load_next_buffer()
		self.input_size, self.output_size = map(lambda x: x.shape, self[0])

	def load_next_buffer(self):
		self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
		self._buffer_index += len(self._buffer_fnames)
		self._buffer_index = self._buffer_index % len(self._files)
		self._buffer = []
		self._cum_size = [0]
		for f in self._buffer_fnames:
			with np.load(f) as data:
				self._buffer += [{k: np.copy(v) for k, v in data.items()}]
				self._cum_size += [self._cum_size[-1] + self._data_per_sequence(data)]

	def __len__(self):
		return self._cum_size[-1]

	def __getitem__(self, i):
		file_index = bisect.bisect(self._cum_size, i) - 1
		seq_index = i - self._cum_size[file_index]
		data = self._buffer[file_index]
		return self._get_data(data, seq_index)

	def _get_data(self, data, seq_index):
		image = data["image"]
		output = data["output"].ravel()
		return image, output

	def _data_per_sequence(self, data):
		return 1

class ImageModel(PTNetwork):
	def __init__(self, input_size, output_size, config, load="", gpu=True, name="viz3d"):
		super().__init__(config, gpu, name)
		self.conv1 = torch.nn.Conv2d(input_size[-1], 4, kernel_size=3, stride=2, padding=1)
		self.conv2 = torch.nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
		self.conv3 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
		self.conv4 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
		self.flatten = torch.nn.Flatten()
		self.linear1 = torch.nn.Linear(self.get_conv_output(input_size), 1000)
		self.linear2 = torch.nn.Linear(1000, output_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.9, patience=50)
		self.to(self.device)
		if load: self.load_model(load)

	def forward(self, state):
		out_dims = state.size()[:-3]
		state = state.reshape(-1, *state.size()[-3:])
		state = state.permute(0,3,1,2)
		state = self.conv1(state).tanh()
		state = self.conv2(state).relu() 
		state = self.conv3(state).tanh() 
		state = self.conv4(state).relu() 
		state = self.flatten(state)
		state = self.linear1(state).relu()
		state = self.linear2(state)
		state = state.view(*out_dims, -1)
		return state

	def get_conv_output(self, input_size):
		inputs = torch.randn(1, input_size[-1], *input_size[:-1])
		output = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
		return np.prod(output.size())

	def get_loss(self, inputs, outputs):
		inputs, outputs = [x.to(self.device) for x in [inputs, outputs]]
		preds = self.forward(inputs)
		loss = (preds - outputs).pow(2).sum(-1).mean()
		return loss

	def optimize(self, inputs, outputs):
		loss = self.get_loss(inputs, outputs)
		self.step(self.optimizer, loss)
		return loss.item()

	def schedule(self, test_loss):
		self.scheduler.step(test_loss)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, net_path = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.state_dict(), filepath)
		return net_path
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath):
			try:
				self.load_state_dict(torch.load(filepath, map_location=self.device))
				print(f"Loaded IMAGE model at {filepath}")
			except Exception as e:
				print(f"Error loading IMAGE model at {filepath}")
		return self

class ImageTrainer():
	def __init__(self, make_env, model_cls, config):
		self.data_dir = config.data_dir
		self.dataset_train = ImageDataset(config, self.data_dir, train=True)
		self.dataset_test = ImageDataset(config, self.data_dir, train=False)
		self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.nworkers)
		self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.nworkers)
		self.model = model_cls(self.dataset_train.input_size, self.dataset_train.output_size, config, gpu=True)
		self.logger = Logger(make_env(), self.model, config)
		self.config = config

	def start(self):
		ep_train_losses = []
		ep_test_losses = []
		for ep in range(self.config.epochs):
			train_loss = self.train_loop(ep)
			test_loss = self.test_loop(ep)
			ep_train_losses.append(train_loss)
			ep_test_losses.append(test_loss)
			self.model.schedule(test_loss)
			self.model.save_model(self.config.env_name)
			if ep_test_losses[-1] <= np.min(ep_test_losses): self.model.save_model(self.config.env_name, "best")
			string = f"Ep: {ep:7d}, Reward: {ep_test_losses[-1]:9.3f} [{ep_train_losses[-1]:8.3f}], Avg: {np.mean(ep_test_losses, axis=0):9.3f} ({1.0:.3f})"
			self.logger.log(string, self.model.get_stats())

	def train_loop(self, ep, update=1):
		batch_losses = []
		self.model.train()
		with tqdm.tqdm(total=len(self.dataset_train)) as pbar:
			pbar.set_description_str(f"Train Ep: {ep}, ")
			for i,data in enumerate(self.train_loader):
				images, outputs = map(lambda x: x.float(), data)
				loss = self.model.optimize(images, outputs)
				if i%update == 0:
					pbar.set_postfix_str(f"Loss: {loss:.4f}")
					pbar.update(images.shape[0])
				batch_losses.append(loss)
		return np.mean(batch_losses)

	def test_loop(self, ep):
		batch_losses = []
		self.model.eval()
		with torch.no_grad():
			for data in self.test_loader:
				images, outputs = map(lambda x: x.float(), data)
				loss = self.model.get_loss(images, outputs)
				batch_losses.append(loss.item())
		return np.mean(batch_losses)

def train3d(data_dir, epochs=50, batch_size=32):
	env_name = "CarRacing-curve-v1"
	make_env, agent_cls, config = get_config(env_name, "sac", "pt")
	config.update(batch_size=batch_size, nworkers=0, epochs=epochs, data_dir=data_dir)
	trainer = ImageTrainer(make_env, ImageModel, config)
	trainer.start()

def normalize(arr):
	norm = np.linalg.norm(arr)
	return arr / norm if norm>0 else arr

def homogeneous_divide(point):
	divided = point[...,:2]/point[...,2:3]
	return divided.astype(np.int32)

def viz3d(data_dir, sample=False, save_video=False, noise=0.0):
	os.makedirs(data_dir, exist_ok=True)
	if not save_video: plt.ion()
	renders = []
	env_name = "CarRacing-curve-v1"
	load_name = "CarRacing-curve-v1"
	make_env, agent_cls, config = get_config(env_name, "sac", "pt")
	env = make_env()
	random = RandomAgent(env.observation_space.shape, env.action_space.shape, config)
	agent = agent_cls(env.observation_space.shape, env.action_space.shape, config)
	agent.network.load_model(env_name)
	if hasattr(agent, "network"): agent.network.load_model(load_name)

	track = env.track
	state = env.reset()
	scale = 0.02
	z = 15.0
	f = 1.0
	I = np.eye(3)
	O = np.zeros([3,1])
	o = np.zeros(3)
	i = np.ones(1)
	camera_width = 4.0
	camera_height = 3.0
	nW = 320
	nH = 240
	npoint = 50
	input_size = (nH,nW,3)
	output_size = (npoint * 6,)
	image = np.zeros(input_size)
	k = (nW-1) / camera_width
	l = (nH-1) / camera_height
	cx = (nW-1) / 2
	cy = (nH-1) / 2
	alpha = f * k
	beta = f * l
	K = np.array([[alpha,0,cx],[0,beta,cy],[0,0,1]])
	Kinv = np.linalg.inv(K)
	axfig = plt.figure()
	axfig.tight_layout()
	ax = plt.axes(projection='3d')
	fig = plt.figure()
	train = sample
	show_pred = False
	use_noise = True
	use_tilt = True
	image_model = ImageModel(input_size, output_size, config, load=load_name)
	Rz = lambda z: np.array([[np.cos(z), -np.sin(z), 0],[np.sin(z), np.cos(z), 0],[0,0,1]])
	Ry = lambda x: np.array([[np.cos(x), 0, np.sin(x)],[0,1,0],[-np.sin(x), 0, np.cos(x)]])
	Rx = lambda x: np.array([[1,0,0],[0, np.cos(x), -np.sin(x)],[0, np.sin(x), np.cos(x)]])
	S = lambda x,y,z: np.array([[x,0,0],[0,y,0],[0,0,z]])
	up = np.array([0,0,1])
	tilt = 0
	for time in range(480):
		tilt = np.clip(tilt+0.4*(np.random.rand()-0.5), -1, 1)
		x = env.dynamics.state.X#+125*(np.sin(time*scale)+1)
		y = env.dynamics.state.Y#+125*np.sin(time*scale)
		theta_z = env.dynamics.state.ψ#- time*scale
		theta_x = np.pi/4 + 0.2*tilt * use_tilt
		look_dir = normalize(Rz(theta_z) @ Ry(theta_x) @ np.array([1,0,0]))
		lat = normalize(np.cross(look_dir, up))
		lon = -normalize(np.cross(look_dir, lat))
		fpoint = np.array([x,y,z])
		t = -fpoint[:,None]
		r = np.stack([lat,lon,look_dir])
		T = np.block([[I, t],[o, i]])
		R = np.block([[r, O],[o, i]])
		RT = np.concatenate([r,r@t], -1)
		M = K @ RT
		rc = S(1,1,-1) @ Rx(-np.pi/2+theta_x) 

		pixels = np.concatenate(np.meshgrid(range(nW), range(nH,0,-1), [1], indexing="xy"), -1)
		rays = np.squeeze(r.T @ Kinv @ pixels[...,None], -1)
		t_int = -fpoint[2]/rays[...,2]
		z_int = fpoint[None,None] + t_int[...,None] * rays
		points = z_int[...,:2]
		index, min_dist = track.get_nearest(points)
		image[t_int <= 0] = np.array([135,206,235])/256
		image[np.logical_and(t_int>0, min_dist>=15)] = np.array([90,160,90])/256
		image[np.logical_and(t_int>0, min_dist<15)] = np.array([100,100,100])/256

		index, _ = track.get_nearest([x,y])
		boundaries = track.boundaries[index:index+npoint]
		npoints = boundaries.shape[0]
		sample = sample and npoint == npoints
		min_bounds = np.array([0,0])
		max_bounds = np.array([nW,nH])
		left_lane = np.concatenate([boundaries[:,0], np.zeros([npoints,1]), np.ones([npoints,1])], -1)
		right_lane = np.concatenate([boundaries[:,1], np.zeros([npoints,1]), np.ones([npoints,1])], -1)
		lpoints = RT @ left_lane.T
		rpoints = RT @ right_lane.T
		lpixels = homogeneous_divide((K @ lpoints).T)
		rpixels = homogeneous_divide((K @ rpoints).T)
		lmask = np.logical_and(np.logical_and(lpixels[:,0]>=0, lpixels[:,1]>=0), np.logical_and(lpixels[:,0]<nW, lpixels[:,1]<nH))
		rmask = np.logical_and(np.logical_and(rpixels[:,0]>=0, rpixels[:,1]>=0), np.logical_and(rpixels[:,0]<nW, rpixels[:,1]<nH))
		lworldpoints = (rc@lpoints).T
		rworldpoints = (rc@rpoints).T
		lworldpoints[np.logical_not(lmask)] = 0
		rworldpoints[np.logical_not(rmask)] = 0
		lanes_output = np.concatenate([lworldpoints, rworldpoints])
		if sample:
			number = len(os.listdir(data_dir))
			np.savez(data_dir + f"sample_{number}.npz", image=image, output=lanes_output)
		image[nH-1-lpixels[lmask,1], lpixels[lmask,0]] = np.array([0,0,0])/256
		image[nH-1-rpixels[rmask,1], rpixels[rmask,0]] = np.array([0,0,0])/256
		if show_pred and not sample:
			image_input = image_model.to_tensor(image)
			lanes_pred = image_model(image_input)
			lanes_out_world = lanes_pred.detach().cpu().numpy().reshape(2*npoint, -1)
			lanes_out = (rc.T@lanes_out_world.T).T
			lpreds, rpreds = np.split(lanes_out, 2, 0)
			lpreds, rpreds = map(lambda x: x.T, [lpreds, rpreds])
			lpixels = homogeneous_divide((K @ lpreds).T)
			rpixels = homogeneous_divide((K @ rpreds).T)
			lmask = np.logical_and(np.logical_and(lpixels[:,0]>=0, lpixels[:,1]>=0), np.logical_and(lpixels[:,0]<nW, lpixels[:,1]<nH))
			rmask = np.logical_and(np.logical_and(rpixels[:,0]>=0, rpixels[:,1]>=0), np.logical_and(rpixels[:,0]<nW, rpixels[:,1]<nH))
			image[nH-1-lpixels[lmask,1], lpixels[lmask,0]] = np.array([255,0,0])/256
			image[nH-1-rpixels[rmask,1], rpixels[rmask,0]] = np.array([255,0,0])/256
		
		ax.cla()
		# camera_mid = fpoint - f*look_dir
		# cam_bl = camera_mid - lat*camera_width/2 - lon*camera_height/2
		# cam_tl = camera_mid - lat*camera_width/2 + lon*camera_height/2
		# cam_tr = camera_mid + lat*camera_width/2 + lon*camera_height/2
		# cam_br = camera_mid + lat*camera_width/2 - lon*camera_height/2
		# point_bl = cam_bl + (fpoint - cam_bl)*5
		# point_tl = cam_tl + (fpoint - cam_tl)*5
		# point_tr = cam_tr + (fpoint - cam_tr)*5
		# point_br = cam_br + (fpoint - cam_br)*5
		# for ray,cam in zip([point_bl, point_tl, point_tr, point_br],[cam_bl, cam_tl, cam_tr, cam_br]):
		# 	line = np.stack([cam, ray],-1)
		# 	ax.plot(line[0], line[1], line[2])
		# corners = np.stack([cam_bl, cam_tl, cam_tr, cam_br, cam_bl],-1)
		# ax.plot(corners[0], corners[1], corners[2])
		# ax.scatter(*fpoint[:,None], s=2)
		# ax.scatter(*camera_mid[:,None], s=2)
		# ax.scatter(track.X, track.Y, track.X*0, s=10)
		# ax.scatter(track.boundaries[:,0,0], track.boundaries[:,0,1], track.boundaries[:,0,1]*0, s=1)
		# ax.scatter(track.boundaries[:,1,0], track.boundaries[:,1,1], track.boundaries[:,0,1]*0, s=1)
		# ax.scatter(left_lane[:,0], left_lane[:,1], left_lane[:,0]*0, s=10, c="black")
		# ax.scatter(right_lane[:,0], right_lane[:,1], right_lane[:,0]*0, s=10, c="black")
		# ax.scatter([x], [y], [0], s=2, c="black")
		# ax.set_xlim(x-20, x+20)
		# ax.set_ylim(y-20, y+20)
		# ax.set_zlim(0, 40)

		ax.scatter(lanes_output[:,0], lanes_output[:,1], lanes_output[:,2], s=10, c="black")
		if show_pred and not sample: ax.scatter(lanes_out_world[:,0], lanes_out_world[:,1], lanes_out_world[:,2], s=10, c="red")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_zlabel("z")
		ax.set_xlim(-50, 50)
		ax.set_ylim(-50, 50)
		ax.set_zlim(0, 100)
		if save_video:
			cvs = FigureCanvasAgg(axfig)
			cvs.draw()
			w, h = map(int, axfig.get_size_inches()*axfig.get_dpi())
			graph = np.copy(np.frombuffer(cvs.tostring_rgb(), dtype="uint8").reshape(h,w,3))
			graph = cv2.resize(graph, tuple(map(int, (image.shape[1],image.shape[0]))), interpolation=cv2.INTER_AREA)
			render = np.concatenate([(image*255).astype(np.uint8), graph],1)
			renders.append(render)
			plt.close()
		else:
			plt.gcf()
			plt.clf()
			plt.imshow(image)
			plt.draw()
			plt.pause(0.0000001)
		# action = np.clip(agent.get_action(state, eps=0.0) + noise * random.get_action(state), -1, 1)
		action = agent.get_action(state, eps=0.0) if np.random.rand() > noise else random.get_action(state)
		state, reward, done, _ = env.step(action)
		print(f"T: {time}, reward: {reward}")
		# env.render()
	if len(renders): make_video(renders, f"logging/videos/pt/viz3d/viz3d_noise{noise:4.2f}{'_tilt' if use_tilt else ''}{'_sample' if train else ''}.mp4", fps=24)

if __name__ == "__main__":
	data_dir = f"{os.path.dirname(os.path.abspath(__file__))}/../data/viz3d/"
	os.makedirs(data_dir, exist_ok=True)
	train = False
	viz3d(data_dir, sample=train, save_video=False, noise=0.0)
	if train: 
		viz3d(data_dir, sample=train, save_video=train, noise=0.05)
		viz3d(data_dir, sample=train, save_video=train, noise=0.1)
		viz3d(data_dir, sample=train, save_video=train, noise=0.15)
		viz3d(data_dir, sample=train, save_video=train, noise=0.2)
		train3d(data_dir)