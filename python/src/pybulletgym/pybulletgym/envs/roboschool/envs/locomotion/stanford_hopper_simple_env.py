from pybulletgym.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from pybulletgym.envs.roboschool.robots.locomotors.stanford_hopper_simple import StanfordHopperSimple


class StanfordHopperSimpleEnv(WalkerBaseBulletEnv):
    def __init__(self):
        self.robot = StanfordHopperSimple()
        WalkerBaseBulletEnv.__init__(self, self.robot)
