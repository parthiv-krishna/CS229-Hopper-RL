from pybulletgym.envs.roboschool.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.roboschool.robots.robot_bases import URDFBasedRobot


class StanfordHopperSimple(URDFBasedRobot, WalkerBase):
    foot_list = ['wheel_left','wheel_right']

    def __init__(self):
        WalkerBase.__init__(self, power=2.5)
        print("################# initializing robot")
        URDFBasedRobot.__init__(self, "stanford_hopper_simple.urdf", "stanford_hopper_simple", action_dim=6, obs_dim=8, self_collision=False)


    def alive_bonus(self, z, pitch):
        return 0