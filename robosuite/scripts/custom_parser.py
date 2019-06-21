import argparse
import json
from robosuite.environments.controller import *
import os
from gym import logger
import subprocess
from colorama import Fore, Back, Style

default_config_file = os.path.join(os.getcwd(), os.path.dirname(__file__), 'default.json')

cwd = os.path.dirname(os.path.abspath(__file__))

def add_controller_params(parser):
    """
    Add all controller-related params. Note: must not conflict with other param names.
    """
    # general controller settings
    parser.add_argument("--control_freq", help="Frequency (Hz) of the policy", type=int)
    parser.add_argument("--use_impedance", help="Whether to use impedance control", type=str2bool, const=True, nargs='?')
    parser.add_argument("--use_delta_impedance", help="Whether to use delta impedance as the action", type=str2bool, const=True, nargs='?')
    parser.add_argument("--control_range", help="What range to scale control input from the policy (will be used as +/- max/min respectively", type=str2list)
    
    # position-orientation controller
    parser.add_argument("--control_range_ori", help="What range to scale orientation deltas (will be used as +/- max/min respectively", type=float)
    parser.add_argument("--control_range_pos", help="What range to scale position deltas (will be used as +/- max/min respectively", type=float)
    parser.add_argument("--initial_impedance_pos", help="What impedance to use for position (either constant, or initial)", type=float)
    parser.add_argument("--initial_impedance_ori", help="What impedance to use for orientation (either constant, or initial)", type=float)

    # joint-velocity controller
    parser.add_argument("--kv", help="Kv values for each of the joints.", type=str2list)
    
    parser.add_argument("--damping_max", help="Max damping values (may be per joint or per dimension)", type=str2list)
    parser.add_argument("--damping_min", help="Min damping value (may be per joint or per dimension)", type=str2list)
    parser.add_argument("--kp_max", help="Max kp values (may be per joint or per dimension)", type=str2list)
    parser.add_argument("--kp_min", help="Min kp values (may be per joint or per dimension)", type=str2list)
    
    parser.add_argument("--damping_max_abs_delta", help="Max range of damping delta used as +/- max/min respectively", type=float)
    
    parser.add_argument("--initial_damping", help="What damping to use (either constant, or initial)", type=float)
    
    parser.add_argument("--kp_max_abs_delta", help="Max kp delta value +/- max/min respectively)", type=float)
    
    parser.add_argument("--inertia_decoupling", help="for joint torques, decoupling with inertia matrix", type=str2bool, const=True, nargs='?')

def add_environment_params(parser):
    """
    Add all environment-related params. Note: must not conflict with other param names.
    """
    parser.add_argument("--horizon", help="Time steps before we restart the simulator (property of the environment)", type=int)
    
    # general observation settings
    parser.add_argument("--use_camera_obs", help="Use images as observations for the policy", type=str2bool, const=True, nargs='?')
    parser.add_argument("--use_object_obs", help="Use ground truth object as observations for the policy", type=str2bool, const=True, nargs='?')
    parser.add_argument("--use_prev_act_obs", help="Use previous action as part of the observation for the policy", type=str2bool, const=True, nargs='?')
    parser.add_argument("--use_contact_obs", help="Use the force-torque sensor as contact sensor and take observations from it", type=str2bool, const=True, nargs='?')
    parser.add_argument("--obs_stack_size", help="Size of the observation stack (default 1)", type=int)
    parser.add_argument("--only_cartesian_obs", help="Use only cartesian measurements as observations (e.g. ee pos and vel)", type=str2bool, const=True, nargs='?')
    parser.add_argument("--camera_name", help="Name of the camera to get observations from", type=str)
    parser.add_argument("--camera_res", help="Resolution of the images (we assume square", type=int)

    # task-specific parameters: FreeSpaceTraj
    parser.add_argument("--acc_vp_reward_mult",help="Multiplier for the num of previously crossed points to add to reward at each step", type=float)
    parser.add_argument("--action_delta_penalty",help="How much to weight the mean of the delta in the action when penalizing the robot", type=float)
    parser.add_argument("--dist_threshold", help="Max dist before end effector is considered to be touching something", type=float)
    parser.add_argument("--distance_penalty_weight", help="Weight for how much getting far away from a via point contributes to reward", type=float)
    parser.add_argument("--distance_reward_weight", help="Weight for how much getting close to a via point contributes to reward", type=float)
    parser.add_argument("--ee_accel_penalty", help="How much to weight the acceleration of the end effector when determining reward", type=float)
    parser.add_argument("--end_bonus_multiplier", help="Multiplier for bonus for finishing episode early", type=float)
    parser.add_argument("--energy_penalty", help="How much to penalize the mean energy used by the joints", type=float)
    parser.add_argument("--num_already_checked",help="How many of the via points have already been hit (to simplify the task without changing the observation space.", type=int)
    parser.add_argument("--num_via_points", help="Number of points the robot must go through", type=int)
    parser.add_argument("--point_randomization",help="Absolute value of variation in points of square.", type=float)
    parser.add_argument("--random_point_order", help="Whether or not to switch randomly between going clockwise and going counter-clockwise", type=str2bool, const=True, nargs='?')
    parser.add_argument("--timestep_penalty", help="Amount of reward subtracted at each timestep", type=float)
    parser.add_argument("--use_debug_cube", help="Whether to use a fixed 8 corners of a cube as the via points", type=str2bool, const=True, nargs='?')
    parser.add_argument("--use_debug_point", help="Whether to use a single fixed point", type=str2bool, const=True, nargs='?')
    parser.add_argument("--use_debug_square", help="Whether to use a fixed 4 corners of a square as the via points", type=str2bool, const=True, nargs='?')
    parser.add_argument("--use_delta_distance_reward", help="Whether to only reward agent for getting closer to the point", type=str2bool, const=True, nargs='?')
    parser.add_argument("--via_point_reward", help="Amount of reward added for reaching a via point", type=float)
    
    # task-specific parameters: Wiping
    parser.add_argument("--arm_collision_penalty", help="Penalty in the reward for colliding with the arm", type=float)
    parser.add_argument("--cnn_small",help="When using CNN, if we want it small", type=str2bool, const=True, nargs='?')
    parser.add_argument("--distance_multiplier", help="Multiplier of the dense reward to the mean distance to pegs", type=float)
    parser.add_argument("--distance_th_multiplier", help="Multiplier inside the tanh for the mean distance to pegs", type=float)
    parser.add_argument("--draw_line", help="Limit the desired position of the ee", type=str2bool, const=True, nargs='?')
    parser.add_argument("--excess_force_penalty_mul", help="Multiplier for the excess of force applied to compute the penalty", type=float)
    parser.add_argument("--line_width",help="Width of the painted line.", type=float)
    parser.add_argument("--n_units_x", help="Number of units to divide the table in x", type=int)
    parser.add_argument("--n_units_y", help="Number of units to divide the table in y", type=int)
    parser.add_argument("--num_sensors", help="Probability of place a sensor", type=int)
    parser.add_argument("--pressure_threshold_max", help="Max force the robot can apply on the environment", type=float)
    parser.add_argument("--prob_sensor", help="Probability of place a sensor", type=float)
    parser.add_argument("--shear_threshold", help="Shear force threshold to deactivate a sensor", type=float)
    parser.add_argument("--table_friction", help="Friction of the table", type=float)
    parser.add_argument("--table_friction_std", help="Std for the friction of the table", type=float)
    parser.add_argument("--table_height", help="Height of the table", type=float)
    parser.add_argument("--table_height_std", help="Standard dev of the height of the table", type=float)
    parser.add_argument("--table_rot_x", help="Std dev of the rotation of the table around x", type=float)
    parser.add_argument("--table_rot_y", help="Std dev of the rotation of the table around y", type=float)
    parser.add_argument("--table_size", help="Size of the table (assumed square surface)", type=float)
    parser.add_argument("--touch_threshold", help="Pressure threshold to deactivate a sensor", type=float)
    parser.add_argument("--two_clusters",help="Creates two clusters of units to wipe", type=str2bool, const=True, nargs='?')
    parser.add_argument("--unit_wiped_reward", help="Reward for wiping one unit (sensor or peg)", type=float)
    parser.add_argument("--wipe_contact_reward", help="Reward for maintaining contact", type=float)
    parser.add_argument("--with_pos_limits", help="Limit the desired position of the ee", type=str2bool, const=True, nargs='?')
    parser.add_argument("--with_qinits",help="Picks among a set of initial qs", type=str2bool, const=True, nargs='?')
    
    # task-specific parameters: Door 
    parser.add_argument("--change_door_friction", type=str2bool, const=True, nargs='?')
    parser.add_argument("--door_damping_max", type=float)
    parser.add_argument("--door_damping_min", type=float)
    parser.add_argument("--door_friction_max", type=float)
    parser.add_argument("--door_friction_min", type=float)
    parser.add_argument("--gripper_on_handle", type=str2bool, const=True, nargs='?')
    parser.add_argument("--handle_reward", type=str2bool, const=True, nargs='?')
    parser.add_argument("--use_door_state", help="Use door hinge angle and handle pos as obs", type=str2bool, const=True, nargs='?')

def add_algorithm_params(parser):
    
    # PPO hyperparameters
    parser.add_argument('--num_timesteps', type=float), 
    parser.add_argument("--nsteps", help="number of steps of the vectorized environment per update", type=int)
    parser.add_argument("--nminibatches", help="number of training minibaches per update", type=int)
    parser.add_argument("--entropy_coef", help="policy entropy coefficient in the optimization objective", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--value_func_coef", help="value function loss coefficient in the optimization objective", type=float)
    parser.add_argument("--max_grad_norm", help="gradient norm clipping coefficient", type=float)
    parser.add_argument("--lam", help="advantage estimation discounting factor", type=float)
    parser.add_argument("--n_epochs_per_update", help="number of training epoches per update", type=int)
    parser.add_argument("--discount_factor", help="PPO discount factor", type=float)
    parser.add_argument("--clip_range", help="clipping range for PPO2", type=float)
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)')
    parser.add_argument("--num_layers", help="Number of layers in network (shared)", type=int)
    parser.add_argument("--num_hidden", help="Number of hidden nodes in network per layer (shared)", type=int)
    parser.add_argument("--max_schedule_ent", help="Max entropy reached by the scheduler", type=float)
    parser.add_argument("--logstd_anneal_start", help="Initial value of the log of the standard deviation before annealing", type=float)
    parser.add_argument("--logstd_anneal_end", help="Final value of the log of the standard deviation after annealing", type=float)
    parser.add_argument("--use_logstd_annealing", help="Whether or not to linearly anneal the log std dev of the policy", type=str2bool, const=True, nargs='?')
    
def custom_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # try to load default config file
    parser.add_argument("--config_file", help=".json file to load parameters from")

    # general parameters
    parser.add_argument("--model", help="Directory containing a previously trained model", type=str)
    parser.add_argument("--visualize", help="Visualize the training process", type=str2bool, const=True, nargs='?')
    parser.add_argument("--force_new_folder", help="Run the replay in a new folder", type=str2bool, const=True, nargs='?')
    parser.add_argument("--replay", help="Replay policy of given model, without modifying", type=str2bool, const=True, nargs='?')
    parser.add_argument("--stochastic_replay", help="Whether or not to load a stochastic model", type=str2bool, const=True, nargs='?')
    parser.add_argument("--quiet", help="Whether or not to suppress debug messages", type=str2bool, const=True, nargs='?')
    parser.add_argument("--random", help="Random location of the peg", type=str2bool, const=True, nargs='?')
    parser.add_argument("--robot", help="Which robot to simulate", type=str.lower, choices=['panda', 'sawyer'])
    parser.add_argument("--log_dir", help="Where do you want to save your logs", type=str, default=os.environ.get('ROBOSUITE_LOG_DIR'))
    parser.add_argument("--log_suffix", help="Suffix to append to log names", type=str, default='')
    parser.add_argument("--ncpu", help="how many cpu's", type=int)
    parser.add_argument("--starting_timestep", help="Which timestep to start TensorBoard logging on", type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor.', type=float)
    parser.add_argument('--seed', help='RNG seed', type=int)
    parser.add_argument("--data_logging", type=str2bool, const=True, nargs='?')
    parser.add_argument("--logging_filename", type=str)
    parser.add_argument("--results_aggregation_file", type=str)
    parser.add_argument("--real_robot", help="Using real robot instead of simulated", type=str2bool, const=True, nargs='?')
    parser.add_argument("--additive_action_noise", help="Max amount of noise added to position control of action", type=float)
    parser.add_argument("--plot_deterministic_policy", help="Whether to plot performance of deterministic policy", type=str2bool, const=True, nargs='?')
    parser.add_argument("--allow_early_end", help="Whether the episode ends when the task is complete", type=str2bool, const=True, nargs='?')
    parser.add_argument("--randomize_initialization", help="Whether to slightly perturb the starting position of the robot", type=str2bool, const=True, nargs='?')
    parser.add_argument("--initial_policy", help="Initial policy to use in residual learning. Either a path to a model checkpoint or a string indicating a function in the code base", type=str)
    
    # controller parameters
    parser.add_argument("--controller", help="Name of the controller to use", type=lambda controller: ControllerType[controller.upper()], choices=list(ControllerType))
    controller_group = parser.add_argument_group('controller')
    add_controller_params(controller_group)

    # environment parameters
    parser.add_argument("--task", help="name of the robot task (without robot name)", type=str)
    environment_group = parser.add_argument_group('environment')
    add_environment_params(environment_group)

    # controller parameters
    parser.add_argument("--algorithm", help="Which RL algorithm to use", type=str)
    algorithm_group = parser.add_argument_group('algorithm')
    add_algorithm_params(algorithm_group)
    
    return parser

# Note: if you want to overwrite the checkpoint used in an existing config file, set model to be ''
def load_defaults(args):
    # Note: this happens here so we can tell if the user set a value or the following code did
    set_suggested_values(args)
    
    # If we pass a directory with a pretrained model we load it and train it further
    if args.model is not None and args.model != '':
        model_config_file = os.path.dirname(args.model)+"/../../config.json"
        if os.path.isfile(model_config_file):
            args.config_file = model_config_file
        else:
            exit('No config file found at ' + model_config_file)

    def load_group_defaults(group_name, group_prefix):
        group_config_path = os.path.join(cwd, os.path.join('config','{}_{}_config.json'.format(group_name, group_prefix)))
 
        if os.path.isfile(group_config_path):
            with open(group_config_path) as f:
                default_group_args = json.load(f)
                return default_group_args
        else:
            logger.error("No default arguments found for "+str(group_name))

    updateable_dict = vars(args)
            
    def apply_args(args):
        for key, val in args.items():
            if key in updateable_dict and updateable_dict[key] is None:
                updateable_dict[key] = val

    # load existing config file if present
    if args.config_file is not None:
        with open(args.config_file) as f:
            existing_args = json.load(f)
        apply_args(existing_args)

    # apply overall defaults
    # NOTE: likely won't have any effect if config_file was set, but may provide back-compatibility
    with open(default_config_file) as f:
        default_args = json.load(f)
    apply_args(default_args)
    
    # apply defaults for each of the groups
    for group_name, group_prefix in [(args.algorithm, 'alg'), (args.task, 'task'), (args.controller, 'controller')]:
        group_defaults = load_group_defaults(group_name, group_prefix)
        if group_defaults is not None: apply_args(group_defaults)

    # include info for git repo, if present
    updateable_dict["commit-hash"] = retrieve_hash()

    # special-case for user wanting to override existing model (to null)
    if updateable_dict["model"] == '': updateable_dict["model"] = None

    # special-case for user wanting to override existing scheduling
    if args.use_logstd_annealing == False:
        updateable_dict['logstd_anneal_start'] = None
        updateable_dict['logstd_anneal_end'] = None
        
    # special-case for user wanting to override existing random seed
    if args.seed == -1: args.seed = None
    
    validate_configuration(args)

def set_suggested_values(args):
    updateable_dict = vars(args)
    
    if args.config_file != default_config_file and (args.model is None or args.model != ''):
        print("-"*80)
        print("WARNING: IF A MODEL IS DEFINED IN THIS CONFIG FILE IT WILL BE LOADED.")
        print("-"*80)

    if args.visualize is None and args.replay:
        updateable_dict['visualize'] = True

    if args.ncpu is None and (args.visualize or args.data_logging or args.replay):
        updateable_dict['ncpu'] = 1

    if args.starting_timestep is None:
        if args.model is None or args.model == '':
            updateable_dict['starting_timestep'] = 1
        else:
            last_checkpoint = int(os.path.basename(args.model))
            updateable_dict['starting_timestep'] = last_checkpoint+1

    if args.additive_action_noise is None:
        updateable_dict['additive_action_noise'] = 0

def validate_configuration(args):
    """
    Check the combination of arguments, issuing warnings or exiting the program
    as necessary.
    """
    if args.data_logging and not args.replay:
        print("-"*80)
        print("WARNING: DATA LOGGING IN EFFECT FOR LIVE TRAINING (STOCHASTIC)")
        print("-"*80)

    if (args.logstd_anneal_start is not None or args.logstd_anneal_end is not None) and args.max_schedule_ent is not None and args.max_schedule_ent != 0:
        print("-"*80)
        print("WARNING: ENTROPY SCHEDULING AND ANNEALING ARE BEING USED TOGETHER")
        print("-"*80)

    if not args.log_dir:
        exit("--log-dir not set! Either set ROBOSUITE_LOG_DIR environment variable, or declare on the command line")

    if args.use_camera_obs and "LD_PRELOAD" in os.environ:
        if os.environ["LD_PRELOAD"] != "":
            logger.error(Fore.RED + 'LD_PRELOAD should not be an env variable if you want to use offscreen rendering. ')
            logger.error('Try export LD_PRELOAD=\"\"')
            logger.error(Fore.WHITE + "")
            exit(-1)

    if args.visualize and not args.use_camera_obs and "LD_PRELOAD" not in os.environ:
        logger.error(Fore.RED + 'LD_PRELOAD should be an env variable if you want to use onscreen rendering.')
        logger.error('Try export LD_PRELOAD=\"path_to_your_libGLEW.so\"')
        logger.error(Fore.WHITE + "")
        exit(-1)

    if args.visualize and not args.use_camera_obs and "LD_PRELOAD" in os.environ:
        if os.environ["LD_PRELOAD"] == "":
            logger.error(Fore.RED + 'LD_PRELOAD should be an env variable if you want to use onscreen rendering.')
            logger.error('Try export LD_PRELOAD=\"path_to_your_libGLEW.so\"')
            logger.error(Fore.WHITE + "")
            exit(-1)

    if args.max_schedule_ent != 0 and args.entropy_coef == 0:
        logger.error(Fore.RED + 'You have set a schedule for the entropy but the ent_coef is zero. Change that!')
        logger.error(Fore.WHITE + "")

    if args.use_camera_obs and 'cnn' not in args.network:
        logger.error(Fore.RED + 'You want to use camera observations but do not use a CNN to process it. Change that!')
        logger.error(Fore.WHITE + "")

    if (args.obs_stack_size == 2 and 'double' not in args.network) or ('double' in args.network and args.obs_stack_size == 1):
        logger.error(Fore.RED + 'Your network type and your number of images/observations do not match')
        logger.error(Fore.WHITE + "")

        
def serialize_args(args):
    ret = ""
    for key, value in args.items():
        # skip config file etc. path names
        if type(value) == str:
            if '/' in value: continue
            if 'log_suffix' == key: continue
            if 'log_dir' == key: continue

        # Necessary to deal with lenght
        splits = key.split('_')
        short_key = ""

        for split in splits:
            short_key += split[0]
        if type(value) is str: value = value.replace(' ', '')
        ret += ".{}-{}".format(short_key, value)
    if ret != "" and ret[-1] == '_':
        ret = ret[:-1]
    return ret

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(s):
    """
    Assuming a string of form "[a, b, c]", return a list of floats
    """
    return [float(val) for val in s[1:-1].split(",")]
    
def retrieve_hash():
    """
    Return the hash for the current git repo. 
    Note: this breaks if the code is called from outside its repo.
    """
    hash_values = {}

    try:
        branch_name = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip()
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        hash_values[branch_name.decode("utf-8")] = commit_hash.decode("utf-8")

        submodule_status = subprocess.check_output(['git', 'submodule', 'status']).strip()
        hash_values['submodule_status'] = submodule_status.decode("utf-8")
    except Exception as e:
        print("Error retrieving git repo information: ", e)

    return hash_values
