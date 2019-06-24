#!/usr/bin/env python3

import os
import time
import numpy as np

from robosuite import *
from robosuite.wrappers import GymWrapper
from robosuite.models import *
from robosuite.wrappers import DataCollectionWrapper
from robosuite.models.tasks import *
from robosuite.scripts.custom_parser import custom_arg_parser, load_defaults, serialize_args
from robosuite.environments.controller import *
    
def configureController(args):
    # note everything is in world frame! 
    controller_args = {}
    controller_args['control_freq'] = args.control_freq
    controller_args['control_range'] = args.control_range

    # add impedance-specific parameters
    if args.controller not in [ControllerType.JOINT_TORQUE, ControllerType.JOINT_VEL]:
        controller_args['impedance_flag'] = args.use_impedance
        controller_args['use_delta_impedance'] = args.use_delta_impedance
        controller_args['kp_max'] = args.kp_max
        controller_args['kp_max_abs_delta'] = args.kp_max_abs_delta
        controller_args['kp_min'] = args.kp_min
        controller_args['damping_max'] = args.damping_max
        controller_args['damping_max_abs_delta'] = args.damping_max_abs_delta
        controller_args['damping_min'] = args.damping_min
        controller_args['initial_damping'] = args.initial_damping

    if args.controller == ControllerType.POS:
        controller_args['control_range_pos'] = args.control_range_pos
        controller_args['initial_impedance_pos'] = args.initial_impedance_pos
        controller_args['initial_impedance_ori'] = args.initial_impedance_ori
        #controller_args['position_limits'] = args.position_limits
        return PositionController(**controller_args)

    if args.controller == ControllerType.POS_ORI:
        controller_args['control_range_pos'] = args.control_range_pos
        controller_args['control_range_ori'] = args.control_range_ori
        controller_args['initial_impedance_pos'] = args.initial_impedance_pos
        controller_args['initial_impedance_ori'] = args.initial_impedance_ori
        # TODO are these ever non-zero?
        #controller_args['position_limits'] = args.position_limits
        #controller_args['orientation_limits'] = args.orientation_limits
        return PositionOrientationController(**controller_args)

    if args.controller == ControllerType.JOINT_IMP:
        return JointImpedanceController(**controller_args)

    if args.controller == ControllerType.JOINT_TORQUE:
        controller_args['inertia_decoupling'] = args.inertia_decoupling
        return JointTorqueController(**controller_args)

    if args.controller == ControllerType.JOINT_VEL:
        controller_args['kv'] = args.kv
        return JointVelocityController(**controller_args)

def createEnvironments(args):
    # ensure perfectly deterministic playback
    if args.replay and not args.stochastic_replay:
        args.randomize_initialization = False
        args.random_point_order = False
        args.point_randomization = 0
        args.random = False

    # TODO - how are these used?
    if args.random:
        if args.task == "Door":
            initializer = True # what does this mean for the door task?
        else: 
            initializer = UniformRandomSampler(x_range=[-0.3,0.3], 
                                           y_range=[-0.3,0.3],
                                           ensure_object_boundary_in_range=False,
                                           z_rotation=False)
    

    else:
        initializer = DeterministicPositionSampler(x_pos=0, 
                                       y_pos=0,
                                       ensure_object_boundary_in_range=False,
                                       z_rotation=False)

    controller = configureController(args)
        
    # minimum common set of args; see custom_parser.py or robot_arm.py for descriptions of arguments
    args_env={
        "controller":controller,
        "use_camera_obs":args.use_camera_obs,
        "use_object_obs":args.use_object_obs,
        "only_cartesian_obs":args.only_cartesian_obs,
        "reward_scale": args.reward_scale,
        "placement_initializer":initializer,
        "randomize_initialization": args.randomize_initialization,
        "has_renderer":args.visualize,
        "has_offscreen_renderer":args.use_camera_obs,
        "control_freq":args.control_freq,
        "horizon":args.horizon,
        "ignore_done":False,
        "initial_policy": args.initial_policy,
        "data_logging": args.data_logging,
        "logging_filename": args.logging_filename,
        "camera_name": args.camera_name,
        "camera_height": args.camera_res,
        "camera_width": args.camera_res,
        "real_robot": args.real_robot,
    }

    if args.task == "Door":
        args_env["gripper_type"] = 'PandaGripper'
        args_env["touch_threshold"] = args.touch_threshold
        args_env["change_door_friction"] = args.change_door_friction
        args_env["gripper_on_handle"] = args.gripper_on_handle
        args_env["energy_penalty"] = args.energy_penalty
        args_env["ee_accel_penalty"] = args.ee_accel_penalty
        args_env["action_delta_penalty"] = args.action_delta_penalty
        args_env["pressure_threshold_max"] = args.pressure_threshold_max
        args_env["excess_force_penalty_mul"] = args.excess_force_penalty_mul
        args_env["use_door_state"] = args.use_door_state 
        args_env["door_friction_min"] = args.door_friction_min 
        args_env["door_friction_max"] = args.door_friction_max 
        args_env["door_damping_min"] = args.door_damping_min  
        args_env["door_damping_max"] = args.door_damping_max 
        args_env["handle_reward"] = args.handle_reward
        args_env["replay"] = args.replay

    elif args.task == "Door2DoF":
        args_env["gripper_type"] = 'PandaGripper'
        args_env["touch_threshold"] = args.touch_threshold
        args_env["change_door_friction"] = args.change_door_friction
        args_env["gripper_on_handle"] = args.gripper_on_handle
        args_env["energy_penalty"] = args.energy_penalty
        args_env["ee_accel_penalty"] = args.ee_accel_penalty
        args_env["action_delta_penalty"] = args.action_delta_penalty
        args_env["pressure_threshold_max"] = args.pressure_threshold_max
        args_env["excess_force_penalty_mul"] = args.excess_force_penalty_mul       
        
    elif args.task == "WipePegs": 
        args_env["gripper_type"] = 'WipingGripper'
        args_env["table_full_size"] = (args.table_size, args.table_size, args.table_height)
        args_env["num_wiping_obj"] = 1
        args_env["arm_collision_penalty"] = args.arm_collision_penalty
        args_env["wipe_contact_reward"] = args.wipe_contact_reward
        args_env["unit_wiped_reward"] = args.unit_wiped_reward

    elif args.task == "WipeTactile":
        args_env["gripper_type"] = 'WipingGripper'
        args_env["table_full_size"] = (args.table_size, args.table_size, args.table_height)
        args_env["num_squares"] = (args.n_units_x,args.n_units_y)
        args_env["touch_threshold"] = args.touch_threshold
        args_env["arm_collision_penalty"] = args.arm_collision_penalty
        args_env["wipe_contact_reward"] = args.wipe_contact_reward
        args_env["unit_wiped_reward"] = args.unit_wiped_reward
        
    elif args.task == "WipeForce":
        args_env["gripper_type"] = 'WipingGripper'
        args_env["table_full_size"] = (args.table_size, args.table_size, args.table_height)
        args_env["num_squares"] = (args.n_units_x,args.n_units_y)
        args_env["touch_threshold"] = args.touch_threshold
        args_env["shear_threshold"] = args.shear_threshold
        args_env["arm_collision_penalty"] = args.arm_collision_penalty
        args_env["wipe_contact_reward"] = args.wipe_contact_reward
        args_env["unit_wiped_reward"] = args.unit_wiped_reward
        args_env["ee_accel_penalty"] = args.ee_accel_penalty
        args_env["with_pos_limits"] = args.with_pos_limits
        args_env["prob_sensor"] = args.prob_sensor
        args_env["table_rot_x"] = args.table_rot_x
        args_env["table_rot_y"] = args.table_rot_y
        args_env["pressure_threshold_max"] = args.pressure_threshold_max
        args_env["table_height_std"] = args.table_height_std
        args_env["excess_force_penalty_mul"] = args.excess_force_penalty_mul       
        args_env["draw_line"] = args.draw_line
        args_env["num_sensors"] = args.num_sensors
        args_env["table_friction"] = (args.table_friction, 0.005, 0.0001)
        args_env["table_friction_std"] = args.table_friction_std
        args_env["distance_multiplier"] = args.distance_multiplier
        args_env["distance_th_multiplier"] = args.distance_th_multiplier
        args_env["line_width"] = args.line_width
        args_env["with_qinits"] = args.with_qinits
        args_env["two_clusters"] = args.two_clusters

    elif args.task == "Wipe3DTactile":
        args_env["gripper_type"] = 'WipingGripper'
        args_env["table_full_size"] = (args.table_size, args.table_size, args.table_height)
        args_env["touch_threshold"] = args.touch_threshold
        args_env["arm_collision_penalty"] = args.arm_collision_penalty
        args_env["wipe_contact_reward"] = args.wipe_contact_reward
        args_env["unit_wiped_reward"] = args.unit_wiped_reward

    elif args.task == "FreeSpaceTraj":
        args_env["gripper_type"] = 'WipingGripper'
        args_env["num_via_points"] = args.num_via_points
        args_env["dist_threshold"] = args.dist_threshold
        args_env["timestep_penalty"] = args.timestep_penalty
        args_env["via_point_reward"] = args.via_point_reward
        args_env["distance_reward_weight"] = args.distance_reward_weight
        args_env["distance_penalty_weight"] = args.distance_penalty_weight
        args_env["use_delta_distance_reward"] = args.use_delta_distance_reward
        args_env["energy_penalty"] = args.energy_penalty
        args_env["ee_accel_penalty"] = args.ee_accel_penalty
        args_env["action_delta_penalty"] = args.action_delta_penalty
        args_env["use_debug_point"] = args.use_debug_point
        args_env["use_debug_cube"] = args.use_debug_cube
        args_env["use_debug_square"] = args.use_debug_square
        args_env["acc_vp_reward_mult"] = args.acc_vp_reward_mult
        args_env["num_already_checked"] = args.num_already_checked
        args_env["end_bonus_multiplier"] = args.end_bonus_multiplier
        args_env["allow_early_end"] = args.allow_early_end
        args_env["random_point_order"] = args.random_point_order
        args_env["point_randomization"] = args.point_randomization

    # elif args.task == "PandaImagesAndProprioWipe":
    #     subproc = DummyVecEnv([lambda: PandaImagesAndProprioWipeEnv(
    #         controller_type = args.controller,
    #         impedance_flag=args.use_impedance,
    #         camera_res= args.camera_res,
    #         use_contact = True)])
    #     subproc.envs[0].horizon = 0

    #     return subproc

    # elif args.task == "RealPandaDoorEnv":
    #     subproc = DummyVecEnv([lambda: RealPandaDoorEnv(
    #         controller_type = args.controller,
    #         impedance_flag=args.use_impedance,
    #         use_contact = True)])
    #     subproc.envs[0].horizon = 0

    #     return subproc

    # else:
    #     logger.error("Wrong task name")
    #     logger.error(args.task)

    observations_keys = ["robot-state"]

    if args.use_object_obs:
        observations_keys += ["object-state"]

    if args.use_camera_obs:
        observations_keys += ["image"]

    if args.use_prev_act_obs:
        observations_keys += ["prev-act"]

    if args.use_contact_obs:
        observations_keys += ["contact-obs"]

    if args.use_delta_impedance:
        observations_keys += ["controller_kp", "controller_damping"]

    return GymWrapper(
        make(args.robot.capitalize()+args.task+"Env", **args_env),
        keys=observations_keys,
        obs_stack_size=args.obs_stack_size
    )

def main(args):
    """
    Train or replay agents with various controllers. 
    See custom_parser.py for a breakdown of arguments with help text.
    """

    env = createEnvironments(args)
    print(env)

    env.reset()
    # env.viewer.set_camera(camera_id=0)

    # do visualization
    for i in range(1000):
        action = np.random.randn(env.dof)
        obs, reward, done, _ = env.step(action)
        env.render()
    
if __name__ == '__main__':
    parser = custom_arg_parser()
    args = parser.parse_args()

    # Note: this will issue errors in the event of an invalid configuration
    load_defaults(args)

    main(args)
