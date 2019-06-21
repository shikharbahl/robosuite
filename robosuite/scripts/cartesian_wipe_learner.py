#!/usr/bin/env python3

from robosuite import *
from robosuite.wrappers import GymWrapper
import numpy as np
import time
from PIL import Image
from IPython import embed
from robosuite.models import *
from robosuite.wrappers import DataCollectionWrapper
from robosuite.models.tasks import *
from gym import wrappers, logger
import time
from baselines.common import tf_util as U
from baselines import bench, logger
from datetime import datetime
import os
from colorama import Fore, Back, Style
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import set_global_seeds
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import json
from calculate_metrics import calculate_metrics
from robosuite.scripts.custom_parser import custom_arg_parser, load_defaults, serialize_args
from robosuite.environments.controller import *

try:
    from real_kuka_gym.envs.robot_transfer_env import RobotTransferEnv
    from real_kuka_gym.envs.panda_images_and_proprio_wipe_env import PandaImagesAndProprioWipeEnv
    from real_kuka_gym.envs.real_panda_door_env import RealPandaDoorEnv

except:
    print("WARNING: real_kuka_gym python code not found!!")

import os
import cv2
import tensorflow as tf
from baselines.common.tf_util import get_session
import csv

try:
    import real_kuka_gym
    import gym
    from real_kuka_gym.envs.robot_transfer_env import RobotTransferEnv
except ImportError:
    logger.error(Fore.BLUE + "WARNING: Couldn't load modules for real robot!")
    logger.error(Fore.BLUE + "Ignore if you are not working on the real robot")
    logger.error(Fore.WHITE + "")


def train_ppo2(env, args, save_model):
    from baselines.common.vec_env.vec_monitor import VecMonitor
    from baselines.ppo2 import ppo2

    env = VecMonitor(env)
    
    def eval_model(savepath):
        args.ncpu = 1
        args.replay = True
        args.stochastic_replay = False
        args.model = savepath
        envs = createEnvironments(args)
        pi = load_policy(envs, args, prefix='eval_policy')
        obs = envs.reset()
        epinfo = None
        while True:
            actions = pi.step(obs)[0]
            obs, rewards, done, infos  = envs.step(actions)
            envs.render2()
            if done:
                epinfo = infos[0]
                break
        envs.close()

        if 'add_vals' in epinfo.keys():
            for add_val in epinfo['add_vals']:
                logger.logkv(add_val+'deterministic', epinfo[add_val])

        logger.info("Now saving deterministic data")

    proprio_dim = 13    # This is cartesian pose (7) and vel (6)
    if not args.only_cartesian_obs:
        proprio_dim += 14 # q and qvel
    if args.use_contact_obs:
        proprio_dim += 1    # Binary contact sensor

    model = ppo2.learn(
        network=args.network, 
        env=env, 
        nsteps=args.nsteps, 
        nminibatches=args.nminibatches,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.value_func_coef,
        lam=args.lam, 
        gamma=float(args.discount_factor), 
        noptepochs=args.n_epochs_per_update, 
        log_interval=1,
        ent_coef=args.entropy_coef,
        lr=float(args.learning_rate),
        cliprange=args.clip_range,
        total_timesteps=args.num_timesteps,
        using_mujocomanip=True,
        save_interval = 1,
        load_path = args.model, 
        num_hidden=args.num_hidden, 
        num_layers=args.num_layers,
        resolution=args.camera_res,
        starting_timestep=args.starting_timestep,
        callback_func=eval_model if args.plot_deterministic_policy else None,
        max_schedule_ent=args.max_schedule_ent,
        proprio_dim=proprio_dim,
        cnn_small= args.cnn_small,
        logstd_anneal_start=args.logstd_anneal_start,
        logstd_anneal_end=args.logstd_anneal_end
    )

    return model

def train(
        env,
        args,
        save_model):

    if args.algorithm == 'ppo2':
        return train_ppo2(env, args, save_model)
    elif args.algorithm == 'td3':
        return train_td3(env, args, save_model)
    else:
        logger.error("Algorithm '{}' not supported!".format(args.algorithm))
    
    
def load_policy(env, args, prefix='learned_policy'):
    from baselines.ppo2.ppo2 import Model
    from baselines.common.policies import build_policy

    # TODO - not sure where 7 and 6 came from
    proprio_dim = 13    # This is cartesian pose (7) and vel (6)
    if not args.only_cartesian_obs:
        proprio_dim += 14 # q and qvel TODO - should this be robot-dependent? not sure if we'll have robots with fewer DoF
    if args.use_contact_obs:
        proprio_dim += 1    # Binary contact sensor
    
    policy_builder = build_policy(env,
                          args.network,
                          stochastic=args.stochastic_replay,
                          num_hidden=args.num_hidden,
                          num_layers=args.num_layers,
                          resolution=args.camera_res,
                          initial_logstd=args.logstd_anneal_start,
                          proprio_dim=proprio_dim,
                          cnn_small=args.cnn_small)
    
    nbatch = env.num_envs * args.nsteps
    nbatch_train = nbatch // args.nminibatches

    # use prefix to ensure no overlap
    with tf.variable_scope(prefix):
        pi = Model(policy=policy_builder,
                   ob_space=env.observation_space,
                   ac_space=env.action_space,
                   nbatch_act=env.num_envs,
                   nbatch_train=nbatch_train,
                   nsteps=args.nsteps,
                   ent_coef=args.entropy_coef,
                   vf_coef=args.value_func_coef,
                   max_grad_norm=args.max_grad_norm,
                   use_entropy_scheduler=(args.max_schedule_ent != 0),
                   training=False)
        pi.load(args.model, prefix=prefix)
    return pi

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
        
    elif args.task == "RealRobotTransfer":
        subproc = DummyVecEnv([lambda: RobotTransferEnv()])
        subproc.envs[0].horizon = 0
        return subproc

    elif args.task == "PandaImagesAndProprioWipe":
        subproc = DummyVecEnv([lambda: PandaImagesAndProprioWipeEnv(
            controller_type = args.controller,
            impedance_flag=args.use_impedance,
            camera_res= args.camera_res,
            use_contact = True)])
        subproc.envs[0].horizon = 0

        return subproc

    elif args.task == "RealPandaDoorEnv":
        subproc = DummyVecEnv([lambda: RealPandaDoorEnv(
            controller_type = args.controller,
            impedance_flag=args.use_impedance,
            use_contact = True)])
        subproc.envs[0].horizon = 0

        return subproc

    else:
        logger.error("Wrong task name")
        logger.error(args.task)

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

    # Makes debugging easier; appears to be necessary for initial policies
    if args.ncpu == 1 and not args.use_camera_obs:
        return DummyVecEnv([lambda: GymWrapper(
            make(args.robot.capitalize()+args.task+"Env", **args_env),
            keys=observations_keys,
            obs_stack_size=args.obs_stack_size
        )])
    
    return SubprocVecEnv([lambda: GymWrapper(
        make(args.robot.capitalize()+args.task+"Env", **args_env),
        keys=observations_keys,
        obs_stack_size=args.obs_stack_size
        ) for i in range(args.ncpu)]) 

def main(args):
    """
    Train or replay agents with various controllers. 
    See custom_parser.py for a breakdown of arguments with help text.
    """
    today = datetime.now()
    if not args.replay or args.force_new_folder:
        dir_name = args.log_dir + today.strftime('%Y-%m-%d--%H-%M-%S')
        dir_name += serialize_args(user_specified) + '_' + args.log_suffix
    else:
        # log to subfolder of original model
        dir_name = os.path.abspath(os.path.dirname(args.model)+"/../../replay_")
        dir_name += os.path.basename(args.model)+'_'
        dir_name += today.strftime('%Y-%m-%d--%H-%M-%S')
    
    args.log_dir = dir_name
    os.mkdir(dir_name)
    os.mkdir(dir_name + '/logging')
    logger.configure(dir_name + '/logging', starting_timestep=args.starting_timestep)
    logger.set_level(logger.DEBUG if not args.quiet else logger.INFO)

    if args.data_logging and args.logging_filename is None:
        args.logging_filename = args.log_dir + "/sim_" + today.strftime('%Y-%m-%d-%H%M%S') + ".h5"
    
    logger.info(Fore.RED + 'Logging saved in ' + dir_name + '/logging')
    logger.info("Current config: ", args)

    # dump config
    with open(os.path.join(dir_name, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # set up TensorFlow session
    config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=args.ncpu,
                        inter_op_parallelism_threads=args.ncpu)
    tf.Session(config=config).__enter__()
    set_global_seeds(args.seed)

        
    # load initial policy (for residual policy learning)
    if args.initial_policy is not None:
        initial_policy_config = os.path.dirname(args.initial_policy)+"/../../config.json"
        if not os.path.isfile(initial_policy_config):
            if args.initial_policy == 'ik_free_space_traj':
                args.initial_policy = None # TODO
            else:
                logger.error("Policy '{}' not found!".format(args.initial_policy))
        else:
            with open(initial_policy_config, 'r') as f:
                from argparse import Namespace
                initial_policy_args = Namespace(**json.load(f))
                initial_policy_args.replay = True
                initial_policy_args.ncpu = 1
                initial_policy_args.model = args.initial_policy
                initial_policy_args.initial_policy = None
            initial_policy_env = createEnvironments(initial_policy_args)
            args.initial_policy = load_policy(initial_policy_env, initial_policy_args, prefix='initial-policy')
            args.initial_policy.env = initial_policy_env.envs[0]
            
    envs = createEnvironments(args)

    # Create a new environment to render images in the real robot experiments
    if args.real_robot:
        # All real robot experiments are on WipeForce
        args.task = "WipeForce"
        rendering_env = createEnvironments(args)
        envs.envs[0]._mujoco_env_render = rendering_env
        q_inits = envs.envs[0].q_inits
        
    # If not visualizing a pretrained model -> Train
    if not args.replay:
        logger.info(Fore.WHITE + 'Dimensions of the action space: {}'.format(envs.action_space))
        logger.info('Dimensions of the observation space: {}'.format(envs.observation_space))
        logger.info('Num timesteps: {}'.format(args.num_timesteps))

        import time
        start = time.time()
        pi = train(envs, args, save_model=True)
        logger.info(Fore.WHITE + 'Training time: ' + str((time.time() - start)/60.0) + ' minutes')

    # If visualizing a pretrained model -> Replay
    else:
        logger.info(Fore.WHITE)
        pi = load_policy(envs, args)
        obs = envs.reset()

        done = False

        # Set initial joint configuration
        if args.with_qinits:
            qinit_now = q_inits[np.random.choice(len(q_inits))]
            envs.set_robot_joint_positions(np.array(qinit_now))

        # Endless loop for replay unless we are data logging
        while True:
            actions = pi.step(obs)[0]            

            obs, _, done, _  = envs.step(actions)

            # If we use a sim robot, render the view
            if not args.real_robot:            
                if args.use_camera_obs :
                    img_ext = envs.render_ext()                
                    img_ext = np.flip(img_ext[0][...,::-1], 0)
                    cv2.imshow('External View',img_ext)
                    cv2.waitKey(10)
                else:
                    envs.render2()

            done = done.any() if isinstance(done, np.ndarray) else done

            if done:
                # if it is done and we are logging data, we stop
                if args.data_logging:
                    break
                # if it is done but not logging data, we reset envs and continue
                else: 
                    envs.reset()
                    if args.with_qinits:                    
                        qinit_now = q_inits[np.random.choice(len(q_inits))]
                        envs.set_robot_joint_positions(np.array(qinit_now))
                envs.reset()
        envs.close()
        if args.data_logging and args.results_aggregation_file is not None:
            calculate_metrics(args.task, args.logging_filename, args.results_aggregation_file)
    
if __name__ == '__main__':
    parser = custom_arg_parser()
    args = parser.parse_args()
    # cache the args that were explicitly set by the user on the command line
    user_specified = {key: value for key, value in vars(args).items() if value is not None}
    # Note: this will issue errors in the event of an invalid configuration
    load_defaults(args)

    main(args)
