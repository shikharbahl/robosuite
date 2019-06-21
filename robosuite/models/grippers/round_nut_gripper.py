"""
Gripper without fingers to wipe a surface
"""
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.grippers.gripper import Gripper


class RoundNutGripper(Gripper):
    def __init__(self):
        super().__init__(xml_path_completion('grippers/round_nut_gripper.xml'))

    def format_action(self, action):
        return action
 #       return np.ones(4) * action

    @property
    def init_qpos(self):
        return []

    @property
    def joints(self):
        return []

    @property
    def dof(self):
        return 0

    def contact_geoms(self):
        return []
