from collections import OrderedDict, deque
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv

from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import Sawyer

### TODO: which variables are NEEDED for OSC controller and which are just for data logging? ###
### TODO: do infos need to get logged in post_action? ###
### TODO: do we need the "total" variables? ###
### TODO: should I include linear and angular eef velocity in robot state? ###
### TODO: prev-act, contact-obs ###
### TODO: do I need the methods under _check_contact? ###
### TODO: sawyer xml changes... do I need sensors as well? ###
### TODO: do I need set_joint_damping in sawyer_robot.py? ###

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

### defaults ###

# {
#     "acc_vp_reward_mult": 0.0,
#     "action_delta_penalty": 0.0,
#     "additive_action_noise": 0.0,
#     "alg": "ppo2",
#     "allow_early_end": false,
#     "arm_collision_penalty": -20,
#     "camera_name": "birdview",
#     "camera_res": 24,
#     "change_door_friction": false,
#     "clip_range": 0.2,
#     "cnn_small": true,
#     "control_freq": 20,
#     "control_range_ori": 0.2,
#     "control_range_pos": 0.05,
#     "controller": "position",
#     "damping_max": 2,
#     "damping_max_abs_delta": 0.1,
#     "damping_min": 0,
#     "discount_factor": 0.99,
#     "dist_threshold": 0.05,
#     "distance_multiplier": 0.0,
#     "distance_th_multiplier": 5,
#     "distance_penalty_weight": 1,
#     "distance_reward_weight": 30.0,
#     "door_damping_max": 500,
#     "door_damping_min": 500,
#     "door_friction_max": 300,
#     "door_friction_min": 300,
#     "draw_line": false,
#     "ee_accel_penalty": 0,
#     "end_bonus_multiplier": 25,
#     "energy_penalty": 0,
#     "entropy_coef": 0.0,
#     "excess_force_penalty_mul": 0.01,
#     "gripper_on_handle": true,
#     "handle_reward": true,
#     "horizon": 1024,
#     "inertia_decoupling": false,
#     "initial_damping": 1,
#     "initial_impedance_ori": 150,
#     "initial_impedance_pos": 150,
#     "kp_max": 300,
#     "kp_max_abs_delta": 10,
#     "kp_min": 10,
#     "lam": 0.95,
#     "learning_rate": 0.0003,
#     "line_width": 0.02,
#     "max_grad_norm": 0.5,
#     "max_schedule_ent": 0.0,
#     "n_epochs_per_update": 6,
#     "n_units_x": 4,
#     "n_units_y": 4,
#     "ncpu": 4,
#     "network": "mlp",
#     "nminibatches": 16,
#     "nsteps": 1024,
#     "num_already_checked": 0,
#     "num_hidden": 64,
#     "num_layers": 2,
#     "num_sensors": 10,
#     "num_timesteps": 100000000.0,
#     "num_via_points": 3,
#     "obs_stack_size": 1,
#     "only_cartesian_obs": true,
#     "peg_size": false,
#     "plot_deterministic_policy": false,
#     "point_randomization": 0,
#     "pressure_threshold_max": 100,
#     "prob_sensor": 1.0,
#     "quiet": false,
#     "random": false,
#     "random_initialization": true,
#     "random_point_order": false,
#     "real_robot": false,
#     "replay": false,
#     "reward_scale": 1.0,
#     "robot": "sawyer",
#     "seed": 2019,
#     "shear_threshold": 1,
#     "stochastic_replay": false,
#     "table_friction": 0.001,
#     "table_friction_std": 0,
#     "table_height": 1.05,
#     "table_height_std": 0.0,
#     "table_rot_x": 0.0,
#     "table_rot_y": 0.0,
#     "table_size": 0.3,
#     "task": "WipeForce",
#     "timestep_penalty": 0.0,
#     "touch_threshold": 1,
#     "two_clusters": false,
#     "unit_wiped_reward": 20,
#     "use_camera_obs": false,
#     "use_contact_obs": false,
#     "use_debug_cube": false,
#     "use_debug_point": false,
#     "use_debug_square": true,
#     "use_delta_distance_reward": false,
#     "use_delta_impedance": false,
#     "use_door_state": true,
#     "use_impedance": false,
#     "use_object_obs": true,
#     "use_prev_act_obs": false,
#     "value_func_coef": 0.5,
#     "via_point_reward": 100.0,
#     "visualize": false,
#     "wipe_contact_reward": 0.5,
#     "with_pos_limits": false,
#     "with_qinits": false
# }

class SawyerEnv(MujocoEnv):
    """Initializes a Sawyer robot environment."""

    def __init__(
        self,
        gripper_type=None,
        gripper_visualization=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        use_camera_obs=False,
        camera_name="frontview",
        camera_height=256,
        camera_width=256,
        camera_depth=False,
        use_osc_controller=False,
    ):
        """
        Args:
            gripper_type (str): type of gripper, used to instantiate
                gripper models from gripper factory.

            gripper_visualization (bool): True if using gripper visualization.
                Useful for teleoperation.

            use_indicator_object (bool): if True, sets up an indicator object that
                is useful for debugging.

            has_renderer (bool): If true, render the simulation state in
                a viewer instead of headless mode.

            has_offscreen_renderer (bool): True if using off-screen rendering.

            render_collision_mesh (bool): True if rendering collision meshes
                in camera. False otherwise.

            render_visual_mesh (bool): True if rendering visual meshes
                in camera. False otherwise.

            control_freq (float): how many control signals to receive
                in every second. This sets the amount of simulation time
                that passes between every action input.

            horizon (int): Every episode lasts for exactly @horizon timesteps.

            ignore_done (bool): True if never terminating the environment (ignore @horizon).

            use_camera_obs (bool): if True, every observation includes a
                rendered image.

            camera_name (str): name of camera to be rendered. Must be
                set if @use_camera_obs is True.

            camera_height (int): height of camera frame.

            camera_width (int): width of camera frame.

            camera_depth (bool): True if rendering RGB-D, and RGB otherwise.
        """

        self.has_gripper = gripper_type is not None
        self.gripper_type = gripper_type
        self.gripper_visualization = gripper_visualization
        self.use_indicator_object = use_indicator_object

        self.use_osc_controller = use_osc_controller
        if self.use_osc_controller:
            self.goal = np.zeros(3)
            self.goal_orientation = np.zeros(3)
            self.desired_force = np.zeros(3)
            self.desired_torque = np.zeros(3)

            self.ee_force = np.zeros(3)
            self.ee_force_bias = np.zeros(3)
            self.contact_threshold = 1    # Maximum contact variation allowed without contact [N]

            self.ee_torque = np.zeros(3)
            self.ee_torque_bias = np.zeros(3)

            self.controller = controller
            # TODO - check that these are updated properly
            self.total_kp = np.zeros(6)
            self.total_damping = np.zeros(6)

            self.n_avg_ee_acc = 10

            # Current and previous policy step q values, joint torques, ft ee applied and actions
            self.prev_pstep_ft = np.zeros(6)
            self.curr_pstep_ft = np.zeros(6)
            self.prev_pstep_a = np.zeros(self.dof)
            self.curr_pstep_a = np.zeros(self.dof)
            self.prev_pstep_q = np.zeros(len(self._ref_joint_vel_indexes))
            self.curr_pstep_q = np.zeros(len(self._ref_joint_vel_indexes))
            self.prev_pstep_t = np.zeros(len(self._ref_joint_vel_indexes))
            self.curr_pstep_t = np.zeros(len(self._ref_joint_vel_indexes))
            self.prev_pstep_ee_v = np.zeros(6)
            self.curr_pstep_ee_v = np.zeros(6)
            self.buffer_pstep_ee_v = deque(np.zeros(6) for _ in range(self.n_avg_ee_acc))
            self.ee_acc = np.zeros(6)

            self.total_ee_acc = np.zeros(6) # used to compute average
            self.total_js_energy = np.zeros(len(self._ref_joint_vel_indexes))

            self.torque_total = 0
            self.joint_torques = 0

            self.prev_ee_pos = np.zeros(7)       
            self.ee_pos = np.zeros(7)

            ## counting joint limits
            self.joint_limit_count = 0

        super().__init__(
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            use_camera_obs=use_camera_obs,
            camera_name=camera_name,
            camera_height=camera_height,
            camera_width=camera_height,
            camera_depth=camera_depth,
        )

    def _load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        super()._load_model()
        self.mujoco_robot = Sawyer()
        if self.has_gripper:
            self.gripper = gripper_factory(self.gripper_type)
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)

        if self.use_osc_controller:
            self.controller.initial_joint = self.mujoco_robot.init_qpos

    def _reset_internal(self):
        """
        Sets initial pose of arm and grippers.
        """
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos

        if self.has_gripper:
            self.sim.data.qpos[
                self._ref_joint_gripper_actuator_indexes
            ] = self.gripper.init_qpos

        if self.use_osc_controller:
            self.goal = np.zeros(3)
            self.goal_orientation = np.zeros(3)
            self.desired_force = np.zeros(3)
            self.desired_torque = np.zeros(3)
            self.prev_pstep_q = np.array(self.mujoco_robot.init_qpos)
            self.curr_pstep_q = np.array(self.mujoco_robot.init_qpos)
            self.prev_pstep_a = np.zeros(self.dof)
            self.curr_pstep_a = np.zeros(self.dof)
            self.prev_pstep_ee_v = np.zeros(6)
            self.curr_pstep_ee_v = np.zeros(6)
            self.buffer_pstep_ee_v = deque(np.zeros(6) for _ in range(self.n_avg_ee_acc))
            self.ee_acc = np.zeros(6)
            self.total_ee_acc = np.zeros(6) # used to compute average
            self.total_kp = np.zeros(6)
            self.total_damping = np.zeros(6)
            self.total_js_energy = np.zeros(len(self._ref_joint_vel_indexes))
            self.prev_ee_pos = np.zeros(7)       
            self.ee_pos = np.zeros(7)
            self.total_joint_torque = 0
            self.joint_torques = 0

    def _get_reference(self):
        """
        Sets up necessary reference for robots, grippers, and objects.
        """
        super()._get_reference()

        # indices for joints in qpos, qvel
        self.robot_joints = list(self.mujoco_robot.joints)
        self._ref_joint_pos_indexes = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.robot_joints
        ]
        self._ref_joint_vel_indexes = [
            self.sim.model.get_joint_qvel_addr(x) for x in self.robot_joints
        ]

        if self.use_indicator_object:
            ind_qpos = self.sim.model.get_joint_qpos_addr("pos_indicator")
            self._ref_indicator_pos_low, self._ref_indicator_pos_high = ind_qpos

            ind_qvel = self.sim.model.get_joint_qvel_addr("pos_indicator")
            self._ref_indicator_vel_low, self._ref_indicator_vel_high = ind_qvel

            self.indicator_id = self.sim.model.body_name2id("pos_indicator")

        # indices for grippers in qpos, qvel
        if self.has_gripper:
            self.gripper_joints = list(self.gripper.joints)
            self._ref_gripper_joint_pos_indexes = [
                self.sim.model.get_joint_qpos_addr(x) for x in self.gripper_joints
            ]
            self._ref_gripper_joint_vel_indexes = [
                self.sim.model.get_joint_qvel_addr(x) for x in self.gripper_joints
            ]

        # indices for joint pos actuation, joint vel actuation, gripper actuation
        self._ref_joint_pos_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("pos")
        ]

        self._ref_joint_vel_actuator_indexes = [
            self.sim.model.actuator_name2id(actuator)
            for actuator in self.sim.model.actuator_names
            if actuator.startswith("vel")
        ]

        if self.has_gripper:
            self._ref_joint_gripper_actuator_indexes = [
                self.sim.model.actuator_name2id(actuator)
                for actuator in self.sim.model.actuator_names
                if actuator.startswith("gripper")
            ]

        # IDs of sites for gripper visualization
        self.eef_site_id = self.sim.model.site_name2id("grip_site")
        self.eef_cylinder_id = self.sim.model.site_name2id("grip_site_cylinder")

    def move_indicator(self, pos):
        """
        Sets 3d position of indicator object to @pos.
        """
        if self.use_indicator_object:
            index = self._ref_indicator_pos_low
            self.sim.data.qpos[index : index + 3] = pos

    def _pre_action(self, action):
        """
        Overrides the superclass method to actuate the robot with the 
        passed joint velocities and gripper control.

        Args:
            action (numpy array): The control to apply to the robot. The first
                @self.mujoco_robot.dof dimensions should be the desired 
                normalized joint velocities and if the robot has 
                a gripper, the next @self.gripper.dof dimensions should be
                actuation controls for the gripper.
        """

        if self.use_osc_controller:
            action = action.copy()  # ensure that we don't change the action outside of this scope
            self.controller.update_model(self.sim, id_name='right_hand', joint_index=self._ref_joint_pos_indexes)
            torques = self.controller.action_to_torques(action, self.policy_step) # this scales and clips the actions correctly
            self.total_joint_torque += np.sum(abs(torques))
            self.joint_torques = torques

            self.sim.data.ctrl[:] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes] + torques

            if self.policy_step:
                self.prev_pstep_q = np.array(self.curr_pstep_q)
                self.curr_pstep_q = np.array(self.sim.data.qpos[self._ref_joint_vel_indexes])
                self.prev_pstep_a = np.array(self.curr_pstep_a)
                self.curr_pstep_a = np.array(action.copy())
                self.prev_pstep_t = np.array(self.curr_pstep_t)
                self.curr_pstep_t = np.array(self.sim.data.ctrl[:])
                self.prev_pstep_ft = np.array(self.curr_pstep_ft)

                # Assumes a ft sensor on the wrist
                force_sensor_id = self.sim.model.sensor_name2id("force_ee")
                force_ee = self.sensor_data[force_sensor_id*3: force_sensor_id*3+3]
                torque_sensor_id = self.sim.model.sensor_name2id("torque_ee")  
                torque_ee = self.sensor_data[torque_sensor_id*3: torque_sensor_id*3+3] 
                self.curr_pstep_ft = np.concatenate([force_ee, torque_ee])

                self.prev_pstep_ee_v = self.curr_pstep_ee_v
                self.curr_pstep_ee_v = np.concatenate([self.sim.data.body_xvelp[self.sim.model.body_name2id("right_hand")], 
                    self.sim.data.body_xvelr[self.sim.model.body_name2id("right_hand")]])

                self.buffer_pstep_ee_v.popleft()
                self.buffer_pstep_ee_v.append(self.curr_pstep_ee_v)

                #convert to matrix
                buffer_mat = []
                for v in self.buffer_pstep_ee_v:
                    buffer_mat += [v]
                buffer_mat = np.vstack(buffer_mat)

                diffs = np.diff(buffer_mat,axis=0)
                diffs *= self.control_freq
                diffs =  np.vstack([self.ee_acc, diffs])
                diffs.reshape((self.n_avg_ee_acc, 6)) 

                self.ee_acc = np.array([np.convolve(col, np.ones((self.n_avg_ee_acc,))/self.n_avg_ee_acc, mode='valid')[0] for col in diffs.transpose()])
        else:
            # clip actions into valid range
            assert len(action) == self.dof, "environment got invalid action dimension"
            low, high = self.action_spec
            action = np.clip(action, low, high)

            if self.has_gripper:
                arm_action = action[: self.mujoco_robot.dof]
                gripper_action_in = action[
                    self.mujoco_robot.dof : self.mujoco_robot.dof + self.gripper.dof
                ]
                gripper_action_actual = self.gripper.format_action(gripper_action_in)
                action = np.concatenate([arm_action, gripper_action_actual])

            # rescale normalized action to control ranges
            ctrl_range = self.sim.model.actuator_ctrlrange
            bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
            weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
            applied_action = bias + weight * action
            self.sim.data.ctrl[:] = applied_action

            # gravity compensation
            self.sim.data.qfrc_applied[
                self._ref_joint_vel_indexes
            ] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

            if self.use_indicator_object:
                self.sim.data.qfrc_applied[
                    self._ref_indicator_vel_low : self._ref_indicator_vel_high
                ] = self.sim.data.qfrc_bias[
                    self._ref_indicator_vel_low : self._ref_indicator_vel_high
                ]

    def _post_action(self, action):
        """
        (Optional) does gripper visualization after actions.
        """

        self.prev_ee_pos = self.ee_pos
        self.ee_pos = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])

        force_sensor_id = self.sim.model.sensor_name2id("force_ee")
        self.ee_force = np.array(self.sensor_data[force_sensor_id*3: force_sensor_id*3+3])

        if np.linalg.norm(self.ee_force_bias) == 0:
            self.ee_force_bias = self.ee_force

        torque_sensor_id = self.sim.model.sensor_name2id("torque_ee")  
        self.ee_torque = np.array(self.sensor_data[torque_sensor_id*3: torque_sensor_id*3+3])      

        if np.linalg.norm(self.ee_torque_bias) == 0:
            self.ee_torque_bias = self.ee_torque

        ret = super()._post_action(action)

        self.total_kp += self.controller.impedance_kp
        self.total_damping += self.controller.damping
        self.total_js_energy += self._compute_js_energy()
        self.total_ee_acc += abs(self.ee_acc)
        self.torque_total = 0 # reset for next round

        self._gripper_visualization()

        return ret

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
        """

        di = super()._get_observation()
        # proprioceptive features
        di["joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )
        di["joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )

        robot_states = [
            np.sin(di["joint_pos"]),
            np.cos(di["joint_pos"]),
            di["joint_vel"],
        ]

        if self.has_gripper:
            di["gripper_qpos"] = np.array(
                [self.sim.data.qpos[x] for x in self._ref_gripper_joint_pos_indexes]
            )
            di["gripper_qvel"] = np.array(
                [self.sim.data.qvel[x] for x in self._ref_gripper_joint_vel_indexes]
            )

            di["eef_pos"] = np.array(self.sim.data.site_xpos[self.eef_site_id])
            di["eef_quat"] = T.convert_quat(
                self.sim.data.get_body_xquat("right_hand"), to="xyzw"
            )

            di["eef_vlin"] = np.array(self.sim.data.get_body_xvelp('right_hand'))
            di["eef_vang"] = np.array(self.sim.data.get_body_xvelr('right_hand'))

            # add in gripper information
            robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"]])

            # robot_states.extend([di["gripper_qpos"], di["eef_pos"], di["eef_quat"], di["eef_vlin"], di["eef_vang"]])

        di["robot-state"] = np.concatenate(robot_states)

        # di["prev-act"] = self.prev_pstep_a

        # # Adding binary contact observation
        # in_contact = np.linalg.norm(self.ee_force - self.ee_force_bias) > self.contact_threshold
        # di["contact-obs"] = in_contact

        return di

    @property
    def joint_damping(self):
        return self.sim.model.dof_damping

    @property
    def joint_frictionloss(self):
        return self.sim.model.dof_frictionloss

    @property
    def action_spec(self):
        """
        Action lower/upper limits per dimension.
        """
        low = np.ones(self.dof) * -1.
        high = np.ones(self.dof) * 1.
        return low, high

    @property
    def dof(self):
        """
        Returns the DoF of the robot (with grippers).
        """
        if self.use_osc_controller:
            return self.controller.action_dim
        dof = self.mujoco_robot.dof
        if self.has_gripper:
            dof += self.gripper.dof
        return dof

    def pose_in_base_from_name(self, name):
        """
        A helper function that takes in a named data field and returns the pose
        of that object in the base frame.
        """

        pos_in_world = self.sim.data.get_body_xpos(name)
        rot_in_world = self.sim.data.get_body_xmat(name).reshape((3, 3))
        pose_in_world = T.make_pose(pos_in_world, rot_in_world)

        base_pos_in_world = self.sim.data.get_body_xpos("base")
        base_rot_in_world = self.sim.data.get_body_xmat("base").reshape((3, 3))
        base_pose_in_world = T.make_pose(base_pos_in_world, base_rot_in_world)
        world_pose_in_base = T.pose_inv(base_pose_in_world)

        pose_in_base = T.pose_in_A_to_pose_in_B(pose_in_world, world_pose_in_base)
        return pose_in_base

    def set_robot_joint_positions(self, jpos):
        """
        Helper method to force robot joint positions to the passed values.
        """
        self.sim.data.qpos[self._ref_joint_pos_indexes] = jpos
        self.sim.forward()

    @property
    def _right_hand_joint_cartesian_pose(self):
        """
        Returns the cartesian pose of the last robot joint in base frame of robot.
        """
        return self.pose_in_base_from_name("right_l6")

    @property
    def _right_hand_pose(self):
        """
        Returns eef pose in base frame of robot.
        """
        return self.pose_in_base_from_name("right_hand")

    @property
    def _right_hand_quat(self):
        """
        Returns eef quaternion in base frame of robot.
        """
        return T.mat2quat(self._right_hand_orn)

    @property
    def _right_hand_total_velocity(self):
        """
        Returns the total eef velocity (linear + angular) in the base frame
        as a numpy array of shape (6,)
        """

        # Use jacobian to translate joint velocities to end effector velocities.
        Jp = self.sim.data.get_body_jacp("right_hand").reshape((3, -1))
        Jp_joint = Jp[:, self._ref_joint_vel_indexes]

        Jr = self.sim.data.get_body_jacr("right_hand").reshape((3, -1))
        Jr_joint = Jr[:, self._ref_joint_vel_indexes]

        eef_lin_vel = Jp_joint.dot(self._joint_velocities)
        eef_rot_vel = Jr_joint.dot(self._joint_velocities)
        return np.concatenate([eef_lin_vel, eef_rot_vel])

    @property
    def _right_hand_pos(self):
        """
        Returns position of eef in base frame of robot.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, 3]

    @property
    def _right_hand_orn(self):
        """
        Returns orientation of eef in base frame of robot as a rotation matrix.
        """
        eef_pose_in_base = self._right_hand_pose
        return eef_pose_in_base[:3, :3]

    @property
    def _right_hand_vel(self):
        """
        Returns velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[:3]

    @property
    def _right_hand_ang_vel(self):
        """
        Returns angular velocity of eef in base frame of robot.
        """
        return self._right_hand_total_velocity[3:]

    @property
    def _joint_positions(self):
        """
        Returns a numpy array of joint positions.
        Sawyer robots have 7 joints and positions are in rotation angles.
        """
        return self.sim.data.qpos[self._ref_joint_pos_indexes]

    @property
    def _joint_velocities(self):
        """
        Returns a numpy array of joint velocities.
        Sawyer robots have 7 joints and velocities are angular velocities.
        """
        return self.sim.data.qvel[self._ref_joint_vel_indexes]

    def _gripper_visualization(self):
        """
        Do any needed visualization here.
        """

        # By default, don't do any coloring.
        self.sim.model.site_rgba[self.eef_site_id] = [0., 0., 0., 0.]

    def _check_contact(self):
        """
        Returns True if the gripper is in contact with another object.
        """
        return False

    def _check_arm_contact(self):
        """
        Returns True if the arm is in contact with another object.
        """
        collision = False
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in self.mujoco_robot.contact_geoms or \
               self.sim.model.geom_id2name(contact.geom2) in self.mujoco_robot.contact_geoms:
                collision = True
                break
        return collision

    def _check_q_limits(self):
        """
        Returns True if the arm is in joint limits or very close to.
        """
        joint_limits = False
        tolerance = 0.1
        for (idx,(q, q_limits)) in enumerate(zip(self.sim.data.qpos[self._ref_joint_pos_indexes],self.sim.model.jnt_range)) :
            if not ( q > q_limits[0] + tolerance and q < q_limits[1] - tolerance):
                print("Joint limit reached in joint " + str(idx))
                joint_limits = True
                self.joint_limit_count+=1
        return joint_limits

    def _compute_q_delta(self):
        """
        Returns the change in joint space configuration between previous and current steps
        """
        q_delta = self.prev_pstep_q - self.curr_pstep_q

        return q_delta

    def _compute_t_delta(self):
        """
        Returns the change in joint space configuration between previous and current steps
        """
        t_delta = self.prev_pstep_t - self.curr_pstep_t

        return t_delta

    def _compute_a_delta(self):
        """
        Returns the change in policy action between previous and current steps
        """

        a_delta = self.prev_pstep_a - self.curr_pstep_a

        return a_delta

    def _compute_ft_delta(self):
        """
        Returns the change in policy action between previous and current steps
        """

        ft_delta = self.prev_pstep_ft - self.curr_pstep_ft

        return ft_delta

    def _compute_js_energy(self):
        """
        Returns the energy consumed by each joint between previous and current steps
        """
        # Mean torque applied
        mean_t = self.prev_pstep_t - self.curr_pstep_t

        # We assume in the motors torque is proportional to current (and voltage is constant)
        # In that case the amount of power scales proportional to the torque and the energy is the 
        # time integral of that
        js_energy = np.abs((1.0/self.control_freq)*mean_t)

        return js_energy

    def _compute_ee_ft_integral(self):
        """
        Returns the integral over time of the applied ee force-torque
        """

        mean_ft = self.prev_pstep_ft - self.curr_pstep_ft
        integral_ft = np.abs((1.0/self.control_freq)*mean_ft)

        return integral_ft
