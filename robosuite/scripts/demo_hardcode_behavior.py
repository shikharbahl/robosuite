"""
Test script for trying to hard code a few behaviors.
"""

import argparse
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper

from robosuite.environments.controller import *

import abc # for abstract base class definitions
import six # preserve metaclass compatibility between python 2 and 3

@six.add_metaclass(abc.ABCMeta)
class SuboptimalExpert:
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action(self, obs):
        """
        Takes a single observation and returns
        a single action.
        """
        raise NotImplementedError

class DirectReachAndGrab(SuboptimalExpert):
    """
    A suboptimal expert that maintains a target end effector position 
    while moving directly towards a target object. Once it is sufficiently
    close, the expert attempts to grab the target object. 
    """
    def __init__(self, target_eef_rotation):
        """
        Args:
            target_eef_rotation (np.array): a 3 x 3 rotation matrix that corresponds
                to the target end effector rotation
        """
        self.target_eef_rotation = target_eef_rotation
        self.has_grasped = False
        self.target_close_enough = 0.005

    def get_orn_error(self, current_eef_rotation):
        """
        Get the current orientation error with respect to the target.

        Args:
            current_eef_rotation (np.array): a 3 x 3 rotation matrix that corresponds
                to the current end effector rotation
        """
        ori_error_mat = np.dot(current_eef_rotation, self.target_eef_rotation.T)
        ori_error_mat_44 = np.eye(4)
        ori_error_mat_44[0:3, 0:3] = ori_error_mat
        angle, direction, _ = T.mat2angle_axis_point(ori_error_mat_44)
        return -angle * direction # compute "orientation error"

    def get_action(self, obs):
        delta_x = obs['Can0_pos'] - obs["eef_pos"]
        distance_to_target = np.linalg.norm(delta_x)
        max_coordinate_deviation = max(np.abs(delta_x))
        if max_coordinate_deviation < 0.05:  
            # apply action as actual distance to target
            delta_x /= 0.05  
        else:
            # compute a direction heading towards the can, but normalize so that
            # the largest component of the vector is 1 or -1 
            # (the largest allowed delta action, which is a distance of 0.05)
            delta_x /= max_coordinate_deviation

        # compute orientation error
        orientation_error = self.get_orn_error(T.quat2mat(obs["eef_quat"]))

        # attempt a grasp if close enough. only one attempt is allowed.
        gripper_action = -1.
        if self.has_grasped or distance_to_target < self.target_close_enough:
            gripper_action = 0.
            delta_x = np.zeros(3)
            self.has_grasped = True

        action = np.concatenate([delta_x, orientation_error, [gripper_action]])
        return action

class ClawMachineReachAndGrab(DirectReachAndGrab):
    """
    A suboptimal expert that maintains a target end effector position 
    while moving towards a target object like a claw machine. It first 
    moves translationally into place before moving vertically towards 
    the target. Once it is sufficiently close, the expert attempts 
    to grab the target object. 
    """
    def __init__(self, target_eef_rotation):
        """
        Args:
            target_eef_rotation (np.array): a 3 x 3 rotation matrix that corresponds
                to the target end effector rotation
        """
        super().__init__(target_eef_rotation)
        self.target_close_enough = 0.008
        self.lateral_close_enough = 0.005

    def get_action(self, obs):
        delta_x = obs['Can0_pos'] - obs["eef_pos"]
        distance_to_target = np.linalg.norm(delta_x)
        max_lateral_coordinate_deviation = max(np.abs(delta_x[:2]))
        if max_lateral_coordinate_deviation < self.lateral_close_enough:
            # move vertically
            delta_x[:2] = 0.
            z_coordinate_deviation = np.abs(delta_x[2])
            if z_coordinate_deviation < 0.05:
                # apply action as actual vertical distance to target
                delta_x[2] /= 0.05 
            else:
                delta_x[2] /= z_coordinate_deviation
        else:
            # move laterally
            delta_x[2] = 0.
            if max_lateral_coordinate_deviation < 0.05:  
                # apply action as actual lateral distance to target
                delta_x[:2] /= 0.05 
            else:
                # compute a direction heading towards the can, but normalize so that
                # the largest component of the vector is 1 or -1 
                # (the largest allowed delta action, which is a distance of 0.05)
                delta_x[:2] /= max_lateral_coordinate_deviation

        # compute orientation error
        orientation_error = self.get_orn_error(T.quat2mat(obs["eef_quat"]))

        # attempt a grasp if close enough. only one attempt is allowed.
        gripper_action = -1.
        if self.has_grasped or distance_to_target < self.target_close_enough:
            gripper_action = 0.
            delta_x = np.zeros(3)
            self.has_grasped = True

        action = np.concatenate([delta_x, orientation_error, [gripper_action]])
        return action

class RaiseDropAndRelease(DirectReachAndGrab):
    """
    A suboptimal expert that maintains a target end effector position 
    while trying to reach a target for dropping an object. Assumes that
    an object is already held in the hand, and that it should be 
    dropped at the target position. The expert makes no assumption about
    how to drop the object - it will just open the gripper after getting
    sufficiently close to the target position.

    If the expert is far (laterally) from the target, then it moves the
    end effector towards the target laterally but also vertically to reach a
    target vertical level. This level will be higher than the target z-position. 

    If the expert is close (laterally) to the target, it moves toward the target
    both laterally and vertically (it descends).
    """
    def __init__(self, target_position, target_eef_rotation, target_z_level):
        """
        Args:
            target_position (np.array): a 3-dim array that corresponds to
                to the position to drop the object
            target_eef_rotation (np.array): a 3 x 3 rotation matrix that corresponds
                to the target end effector rotation
            target_z_level (float): the target z-location for the raising part 
                of the expert
        """
        self.target_position = target_position
        self.target_eef_rotation = target_eef_rotation
        self.target_z_level = target_z_level
        self.has_released = False
        self.z_close_enough = 0.1
        self.lateral_close_enough = 0.005

    def get_action(self, obs):
        delta_x = self.target_position - obs["eef_pos"]
        distance_to_target = np.linalg.norm(delta_x)
        max_lateral_coordinate_deviation = max(np.abs(delta_x[:2]))

        # compute z-action
        if max_lateral_coordinate_deviation > self.lateral_close_enough:
            # reset z-target to target z-level, since we're far from the target pos
            delta_x[2] = self.target_z_level - obs["eef_pos"][2]
        z_coordinate_deviation = np.abs(delta_x[2])
        if z_coordinate_deviation < 0.05:
            # apply action as actual vertical distance to target
            delta_x[2] /= 0.05 
        else:
            # apply maximum action in z-direction
            delta_x[2] /= z_coordinate_deviation 

        # compute lateral action
        if max_lateral_coordinate_deviation < 0.05:  
            # apply action as actual lateral distance to target
            delta_x[:2] /= 0.05 
        else:
            # compute a direction heading towards the can, but normalize so that
            # the largest component of the vector is 1 or -1 
            # (the largest allowed delta action, which is a distance of 0.05)
            delta_x[:2] /= max_lateral_coordinate_deviation

        # compute orientation error
        orientation_error = self.get_orn_error(T.quat2mat(obs["eef_quat"]))

        # attempt to drop the object if close enough. only one attempt is allowed.
        gripper_action = 0.
        if self.has_released or \
            (max_lateral_coordinate_deviation < self.lateral_close_enough and \
                np.abs(self.target_position[2] - obs["eef_pos"][2]) < self.z_close_enough):
            gripper_action = -1.
            delta_x = np.zeros(3)
            self.has_released = True

        action = np.concatenate([delta_x, orientation_error, [gripper_action]])
        return action

if __name__ == "__main__":
    env = robosuite.make("SawyerPickPlaceCan", 
        has_renderer=True, 
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=100, #20,
        gripper_visualization=True,
        use_impedance=False,
        reward_shaping=True,
        # controller=ControllerType.POS, # ADDED
        controller=ControllerType.POS_ORI, # ADDED
    )

    ### TODO: support for different kinds of experts easily (manhattan distance, direct distance) and evaluate success ###
    ### TODO: does action range apply correctly to delta orientation? ###

    # target_bin_placements
    # [[0.5025 0.2575 0.8   ]
    #  [0.6975 0.2575 0.8   ]
    #  [0.5025 0.5025 0.8   ]
    #  [0.6975 0.5025 0.8   ]]

    # trying OSC
    while True:
        obs = env.reset()
        env.viewer.set_camera(0)
        target_eef_rotation =T.quat2mat(obs["eef_quat"])
        # expert = DirectReachAndGrab(target_eef_rotation=target_eef_rotation)
        expert = ClawMachineReachAndGrab(
            target_eef_rotation=target_eef_rotation,
        )
        expert2 = RaiseDropAndRelease(
            target_position=np.array([0.6975, 0.5025, 0.8]),
            target_eef_rotation=target_eef_rotation,
            target_z_level=1.0,
        )

        for _ in range(500):
            action = expert.get_action(obs)
            obs, _, _, _ = env.step(action)
            env.render()

        for _ in range(1000):
            action = expert2.get_action(obs)
            obs, _, _, _ = env.step(action)
            env.render()


