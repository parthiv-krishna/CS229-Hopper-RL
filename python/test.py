import os
import argparse
import numpy as np
try: import keyboard as kbd
except: print("No module keyboard")
from src.agents import get_config, all_refs, all_agents
from src.agents.wrappers import ParallelAgent, RefAgent
from src.envs.wrappers import EnsembleEnv, EnvManager, EnvWorker
from src.utils.data import get_data_dir, save_rollouts
from src.utils.rand import RandomAgent
from src.utils.logger import Logger
from src.utils.config import Config
from src.utils.misc import rollout
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

def test_rl(args, nsteps, log=False, save_video=False):
	iters = ["", 1,2,3,4,5] 
	if not log: iters = [np.random.choice(iters)]
	for num in iters:
		make_env, agent_cls, config = get_config(f"CarRacing-curve{num}-v1", args.agent_name, "pt")
		config.update(**args.__dict__)
		envs = EnsembleEnv(make_env, 0)
		checkpoint = "CarRacing-curve-v1"
		agent = (RefAgent if config.ref else ParallelAgent)(envs.state_size, envs.action_size, agent_cls, config, gpu=True).load_model(checkpoint)
		logger = Logger(envs, agent, config, log_type="runs") if log else None
		state = envs.reset(sample_track=bool(num))
		total_reward = None
		done = False
		for step in range(0,nsteps):
			env_action, action, state = agent.get_env_action(envs.env, state, 0.0)
			state, reward, done, info = envs.step(env_action)
			spec = envs.env.dynamics.observation_spec(state[0])
			total_reward = reward[0] if total_reward is None else total_reward + reward[0]
			log_string = f"Step: {step:8d}, Reward: {f'{reward[0]:5.3f}'.rjust(8,' ')}, Action: {np.array2string(env_action[0], separator=',')}, Done: {done[0]}"
			logger.log(log_string, info[0]) if logger else print(log_string)
			envs.render(mode="video" if save_video else "human")
			if done: break
		print(f"Reward: {total_reward}")
		envs.close(path=logger.log_path.replace("runs","videos").replace(".txt",".mp4") if logger else None)

def test_input(args, nsteps, log=False, save_video=False, new_track=None):
	make_env, agent_cls, config = get_config("CarRacing-sebring-v1", "rand", "pt")
	envs = EnsembleEnv(make_env, 0)
	agent = InputController(envs.state_size, envs.action_size, config, gpu=True)
	state = envs.reset()
	total_reward = None
	done = False
	for step in range(0,nsteps):
		env_action, action = agent.get_env_action(envs.env, state, 0.0)
		state, reward, done, info = envs.step(env_action)
		spec = envs.env.dynamics.observation_spec(state[0])
		total_reward = reward if total_reward is None else total_reward + reward
		log_string = f"Step: {step:8d}, Reward: {reward[0]:5.3f}, Action: {np.array2string(env_action[0], separator=',')}, Done: {done[0]}, {spec.print()}"
		if done: break#envs.reset()
		print(log_string)
		envs.render()
	print(f"Reward: {total_reward}")
	envs.close()

def parse_args():
	parser = argparse.ArgumentParser(description="MPC Tester")
	parser.add_argument("--nsteps", type=int, default=6800, help="Number of steps to train the agent")
	parser.add_argument("--save_run", action="store_true", help="Whether to log each time step's state in a run txt file")
	parser.add_argument("--save_video", action="store_true", help="Whether to save the simulation run as video instead of rendering")
	parser.add_argument("--input", action="store_true", help="Whether to use keyboard as input")
	parser.add_argument("--sample", action="store_true", help="Whether to save the sampled mppi trajectories")
	parser.add_argument("--agent_name", type=str, default=None, choices=all_agents, help="Which agent network to use")
	parser.add_argument("--ref", type=str, default=None, choices=all_refs, help="Which reference processing network to use")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	function = test_input if args.input else test_rl if args.agent_name is not None else test_mppi
	function(args, args.nsteps, args.save_run, args.save_video)