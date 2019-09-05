from collections import OrderedDict, deque
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments import MujocoEnv

from robosuite.models.grippers import gripper_factory
from robosuite.models.robots import Sawyer

from robosuite.controllers.osc_controller import PositionOrientationController, PositionController

### Had to change position of eef back to robosuite default, and comment out inertia. ###

### TODO: try only changing actuator type in original sawyer xml and using that for OSC ###

### TODO: with stuff commented out, the gripper is floating in mid-air... ###
### TODO: the orn tilt seems unnatural... ###
### TODO: control freq of controller should match that of env right? ###
### TODO: should I include linear and angular eef velocity in robot state? ###

def make_controller(absolute=False, control_freq=20, position_only=False):
    # return PositionController(
    if position_only:
        return PositionController(
            control_range_pos=0.05, # delta pos action range
            kp_max_abs_delta=10,
            damping_max_abs_delta=0.1,
            use_delta_impedance=False, # only for variable impedance
            initial_impedance_pos=300, #150, # GAINs 
            initial_impedance_ori=300, #150,
            initial_damping=1,
            max_action=1., 
            min_action=-1.,
            impedance_flag=False, # not using variable impedance
            kp_max=300, 
            kp_min=10, 
            damping_max=2, 
            damping_min=0, 
            initial_joint=None,
            control_freq=control_freq,
            position_limits=[[0,0,0],[0,0,0]],
            orientation_limits=[[0,0,0],[0,0,0]],
            absolute=absolute,
        )
    return PositionOrientationController(
        control_range_pos=0.05, # delta pos action range
        control_range_ori=0.2, # delta orn action range (euler angles)
        kp_max_abs_delta=10,
        damping_max_abs_delta=0.1,
        use_delta_impedance=False, # only for variable impedance
        initial_impedance_pos=300, #150, # GAINs 
        initial_impedance_ori=300, #150,
        initial_damping=1,
        max_action=1., 
        min_action=-1.,
        impedance_flag=False, # not using variable impedance
        kp_max=300, 
        kp_min=10, 
        damping_max=2, 
        damping_min=0, 
        initial_joint=None,
        control_freq=control_freq,
        position_limits=[[0,0,0],[0,0,0]],
        orientation_limits=[[0,0,0],[0,0,0]],
        absolute=absolute,
    )

    # # note everything is in world frame! 
    # controller_args = {}
    # controller_args['control_freq'] = args.control_freq
    # controller_args['control_range'] = args.control_range

    # # add impedance-specific parameters
    # if args.controller not in [ControllerType.JOINT_TORQUE, ControllerType.JOINT_VEL]:
    #     controller_args['impedance_flag'] = args.use_impedance
    #     controller_args['use_delta_impedance'] = args.use_delta_impedance
    #     controller_args['kp_max'] = args.kp_max
    #     controller_args['kp_max_abs_delta'] = args.kp_max_abs_delta
    #     controller_args['kp_min'] = args.kp_min
    #     controller_args['damping_max'] = args.damping_max
    #     controller_args['damping_max_abs_delta'] = args.damping_max_abs_delta
    #     controller_args['damping_min'] = args.damping_min
    #     controller_args['initial_damping'] = args.initial_damping

    # if args.controller == ControllerType.POS:
    #     controller_args['control_range_pos'] = args.control_range_pos
    #     controller_args['initial_impedance_pos'] = args.initial_impedance_pos
    #     controller_args['initial_impedance_ori'] = args.initial_impedance_ori
    #     #controller_args['position_limits'] = args.position_limits
    #     return PositionController(**controller_args)

    # if args.controller == ControllerType.POS_ORI:
    #     controller_args['control_range_pos'] = args.control_range_pos
    #     controller_args['control_range_ori'] = args.control_range_ori
    #     controller_args['initial_impedance_pos'] = args.initial_impedance_pos
    #     controller_args['initial_impedance_ori'] = args.initial_impedance_ori
    #     # TODO are these ever non-zero?
    #     #controller_args['position_limits'] = args.position_limits
    #     #controller_args['orientation_limits'] = args.orientation_limits
    #     return PositionOrientationController(**controller_args)

    # if args.controller == ControllerType.JOINT_IMP:
    #     return JointImpedanceController(**controller_args)

    # if args.controller == ControllerType.JOINT_TORQUE:
    #     controller_args['inertia_decoupling'] = args.inertia_decoupling
    #     return JointTorqueController(**controller_args)

    # if args.controller == ControllerType.JOINT_VEL:
    #     controller_args['kv'] = args.kv
    #     return JointVelocityController(**controller_args)


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
        absolute_control=False,
        osc_position_only=False,
        collect_osc_data=False,
        osc_raw_torques=False,
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

        if use_osc_controller:
            self.absolute_control = absolute_control
            self.osc_position_only = osc_position_only
            self.osc_raw_torques = osc_raw_torques
            if not osc_raw_torques:
                self.controller = make_controller(absolute=self.absolute_control, control_freq=control_freq, position_only=osc_position_only)

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
            use_osc_controller=use_osc_controller,
            collect_osc_data=collect_osc_data,
        )

    def _load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        super()._load_model()
        # self.mujoco_robot = Sawyer()
        self.mujoco_robot = Sawyer(use_osc=self.use_osc_controller)
        if self.has_gripper:
            self.gripper = gripper_factory(self.gripper_type)
            if not self.gripper_visualization:
                self.gripper.hide_visualization()
            self.mujoco_robot.add_gripper("right_hand", self.gripper)

        if self.use_osc_controller and not self.osc_raw_torques:
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

    def _pre_action(self, action, policy_step=None):
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

        # clip actions into valid range
        assert len(action) == self.dof, "environment got invalid action dimension"

        if (not self.use_osc_controller) or ((not self.absolute_control) and (not self.osc_raw_torques)):
            low, high = self.action_spec
            action = np.clip(action, low, high)

        if self.use_osc_controller:

            if self.has_gripper:
                # get rescaled gripper action
                arm_action = action[:-self.gripper.dof]
                gripper_action_in = action[-self.gripper.dof:]
                gripper_action_actual = self.gripper.format_action(gripper_action_in)

                # rescale normalized action to control ranges
                ctrl_range = self.sim.model.actuator_ctrlrange[-self.gripper.dof:]
                bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
                weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
                gripper_action = bias + weight * gripper_action_actual
            else:
                arm_action = action
                gripper_action = []

            if self.osc_raw_torques:
                self.torques = np.array(arm_action)
            else:
                # motor torques from controller
                self.controller.update_model(self.sim, id_name='right_hand', joint_index=(self._ref_joint_pos_indexes, self._ref_joint_vel_indexes))
                self.torques = self.controller.action_to_torques(arm_action, policy_step) # this scales and clips the actions correctly
            motor_torque_ctrl = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes] + self.torques

            # apply torques with gripper action
            total_ctrl = np.concatenate([motor_torque_ctrl, gripper_action])
            self.sim.data.ctrl[:] = total_ctrl

        else:

            # only apply control when policy gives new input
            if policy_step:

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
        ret = super()._post_action(action)
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
        if self.use_osc_controller and (not self.osc_raw_torques):
            dof = self.controller.action_dim
        else:
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

