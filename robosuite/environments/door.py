import numpy as np
from collections import OrderedDict
from robosuite.utils import RandomizationError
from robosuite.environments.robot_arm import RobotArmEnv
from robosuite.models import *
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import Task, UniformRandomSampler, HeightTableTask, DoorTask
from robosuite.models.arenas import HeightTableArena, EmptyArena, TableArena
from robosuite.models.objects import CylinderObject, PlateWithHoleObject, BoxObject, DoorObject
import multiprocessing
from robosuite.environments.controller import *

import robosuite.utils.transform_utils as T

class DoorEnv(RobotArmEnv):

    def __init__(
        self,
        action_delta_penalty,
        energy_penalty,
        ee_accel_penalty,
        table_full_size=(0.5, 0.5, 0.8),
        use_object_obs=True,
        placement_initializer=True,
        arm_collision_penalty = -50,
        object_type=None,
        dist_threshold=0.01,
        touch_threshold=0,
        excess_force_penalty_mul=0.01,
        pressure_threshold_max=30,
        use_door_state = False,
        robot_type='Panda',
        change_door_friction = False,
        door_damping_max = 500,
        door_damping_min = 100,
        door_friction_max = 500,
        door_friction_min= 100, 
        gripper_on_handle= True,
        handle_reward = True,
        replay=False,
        **kwargs
        ): 

        # settings for table top
        self.dist_threshold = dist_threshold
        self.timestep = 0
        self.excess_force_penalty_mul = excess_force_penalty_mul
        self.excess_torque_penalty_mul = excess_force_penalty_mul * 10.0
        self.torque_threshold_max = pressure_threshold_max*0.1
        self.pressure_threshold_max = pressure_threshold_max

        # set reward shaping
        self.energy_penalty = energy_penalty
        self.ee_accel_penalty = ee_accel_penalty
        self.action_delta_penalty = action_delta_penalty
        self.handle_reward = handle_reward
        self.arm_collision_penalty = arm_collision_penalty
        self.handle_final_reward = 1
        self.handle_shaped_reward = 0.5
        self.max_hinge_diff = 0.05
        self.max_hinge_vel= 0.1
        self.final_reward = 500
        self.door_shaped_reward = 30
        self.hinge_goal = 1.04
        self.velocity_penalty= 10

        # set what is included in the observation
        self.use_door_state = use_door_state
        self.use_object_obs = True  # ground-truth object states

        # door friction
        self.change_door_friction = change_door_friction 
        self.door_damping_max = door_damping_max 
        self.door_damping_min = door_damping_min 
        self.door_friction_max = door_friction_max 
        self.door_friction_min= door_friction_min

        #self.table_full_size = table_full_size
        self.gripper_on_handle = gripper_on_handle
        self.robot_type = robot_type

        self.collisions = 0
        self.f_excess = 0
        self.t_excess = 0

        self.placement_initializer = placement_initializer
        self.table_origin = [0.50 + table_full_size[0] / 2,0,0]

        super(DoorEnv, self).__init__(robot_type=robot_type, **kwargs)

        if self.data_logging:
            self.file_logging.create_dataset('percent_viapoints_', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('hinge_angle', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('hinge_diff', (self.data_count, 1), maxshape=(None,1))
            self.file_logging.create_dataset('hinge_goal', (self.data_count, 1), maxshape=(None,1))

            self.file_logging.create_dataset('done', (self.data_count, 1), maxshape=(None, 1))

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        self.robot_contact_geoms = self.mujoco_robot.contact_geoms

        self.mujoco_arena = TableArena(table_full_size=(0.8, 0.8, 1.43-0.375))

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin(self.table_origin)

        self.mujoco_objects = OrderedDict() 
        self.door = DoorObject()
        self.mujoco_objects = OrderedDict([("Door", self.door)])

        if self.gripper_on_handle:
            if self.robot_type == 'Sawyer': 
                self.mujoco_robot._init_qpos = np.array([-0.26730423, -1.85458729, 0.63220668, 2.40196438, 0.9033082, -0.80319783, -0.42571791])

            elif self.robot_type == 'Panda':
               self.mujoco_robot._init_qpos = np.array([-0.01068642,-0.05599809,0.22389938,-1.81999415,-1.54907898,2.82220116,2.28768505])

        # task includes arena, robot, and objects of interest
        self.model = DoorTask(self.mujoco_arena, 
                                self.mujoco_robot, 
                                self.mujoco_objects)

        if self.change_door_friction:
            damping = np.random.uniform(high=np.array([self.door_damping_max]), low=np.array([self.door_damping_min]))
            friction = np.random.uniform(high=np.array([self.door_friction_max]), low=np.array([self.door_friction_min]))
            self.model.set_door_damping(damping)
            self.model.set_door_friction(friction)

        self.model.place_objects(randomize=self.placement_initializer)

    def _get_reference(self):
        super()._get_reference()

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()
        # reset joint positions
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(self.mujoco_robot.init_qpos)
        self.timestep = 0
        self.wiped_sensors = []
        self.touched_handle = 0
        self.collisions = 0
        self.f_excess = 0
        self.t_excess = 0

    def reward(self, action):
        reward = 0    
        grip_id = self.sim.model.site_name2id("grip_site")
        eef_position = self.sim.data.site_xpos[grip_id]

        force_sensor_id = self.sim.model.sensor_name2id("force_ee")
        self.force_ee = self.sensor_data[force_sensor_id*3: force_sensor_id*3+3]
        total_force_ee = np.linalg.norm(np.array(self.ee_force))

        torque_sensor_id = self.sim.model.sensor_name2id("torque_ee")  
        self.torque_ee = self.sensor_data[torque_sensor_id*3: torque_sensor_id*3+3] 
        total_torque_ee = np.linalg.norm(np.array(self.torque_ee))

        self.hinge_diff = np.abs(self.hinge_goal - self.hinge_qpos)

        # Neg Reward from collisions of the arm with the table
        if self._check_arm_contact():
            reward = self.arm_collision_penalty
        elif self._check_q_limits():
            reward = self.arm_collision_penalty
        else: 

            # add reward for touching handle or being close to it
            if self.handle_reward:
                dist = np.linalg.norm(eef_position[0:2] - self.handle_position[0:2])

                if dist < self.dist_threshold and abs(eef_position[2]-self.handle_position[2])<0.02:
                    self.touched_handle = 1
                    reward += self.handle_reward
                else:
                    # if robot starts 0.3 away and dist_threshold is 0.05: [0.005, 0.55] without scaling
                    reward += (self.handle_shaped_reward* (1 - np.tanh(3*dist))).squeeze()
                    self.touched_handle = 0

            # penalize excess force
            if total_force_ee > self.pressure_threshold_max:
                reward -= self.excess_force_penalty_mul*total_force_ee
                self.f_excess += 1

            # penalize excess torque
            if total_torque_ee > self.torque_threshold_max:
                reward -= self.excess_torque_penalty_mul * total_torque_ee
                self.t_excess += 1

            # award bonus either for opening door or for making process toward it
            if self.hinge_diff < self.max_hinge_diff and abs(self.hinge_qvel) < self.max_hinge_vel:
                reward += self.final_reward

            else:
                reward += (self.door_shaped_reward*(np.abs(self.hinge_goal) - self.hinge_diff )).squeeze()
                reward -= (self.hinge_qvel*self.velocity_penalty).squeeze()

        # penalize for jerkiness
        reward -= self.energy_penalty * np.sum(np.abs(self.joint_torques))
        reward -= self.ee_accel_penalty * np.mean(abs(self.ee_acc))
        reward -= self.action_delta_penalty * np.mean(abs(self._compute_a_delta()[:3]))

        string_to_print = 'Process {pid}, timestep {ts:>4}: reward: {rw:8.4f} hinge diff: {ha} excess-f: {ef}, excess-t: {et}'.format(
            pid = id(multiprocessing.current_process()) ,
            ts = self.timestep, 
            rw = reward, 
            con = self._check_contact(), 
            ha = self.hinge_diff,
            ef = self.f_excess,
            et = self.t_excess)

        logger.debug(string_to_print)

        return reward

    def _get_observation(self):
        di = super()._get_observation()

        if self.use_camera_obs:
            camera_obs = self.sim.render(camera_name=self.camera_name,
                                     width=self.camera_width,
                                     height=self.camera_height,
                                     depth=self.camera_depth)
            if self.camera_depth:
                di['image'], di['depth'] = camera_obs
            else:
                di['image'] = camera_obs   

        if self.use_object_obs:
            # checking if contact is made, add as state
            contact = self._check_contact()
            di['object-state'] = np.array([[0,1][contact]])

            # door information for rewards
            handle_id = self.sim.model.site_name2id("S_handle")
            self.handle_position = self.sim.data.site_xpos[handle_id]
            handle_orientation_mat = self.sim.data.site_xmat[handle_id].reshape(3,3)
            handle_orientation = T.mat2quat(handle_orientation_mat)
            hinge_id =  self.sim.model.get_joint_qpos_addr("door_hinge")

            self.hinge_qpos = np.array((self.sim.data.qpos[hinge_id])).reshape(-1,)
            self.hinge_qvel = np.array((self.sim.data.qvel[hinge_id])).reshape(-1,)


            if self.use_door_state:
                di['object-state'] = np.concatenate([
                    di['object-state'],
                    self.hinge_qpos,
                    self.hinge_qvel   ])

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in self.gripper.contact_geoms() or \
               self.sim.model.geom_id2name(contact.geom2) in self.gripper.contact_geoms():
                collision = True
                break
        return collision

    def _check_terminated(self):
        """
        Returns True if task is successfully completed
        """
        terminated = False
        if self._check_q_limits() :
            print(40*'-' + " JOINT LIMIT " + 40*'-')
            terminated = True

         # Prematurely terminate if contacting the table with the arm
        if self._check_arm_contact():
            print(40*'-' + " COLLIDED " + 40*'-')
            terminated = True

        return terminated

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        return

    def _post_action(self, action):
        """
        If something to do
        """
        reward, done, info = super()._post_action(action)

        info['add_vals']+= ['hinge_angle', 'hinge_diff', 'percent_viapoints_', 'touched_handle']
        info['hinge_angle'] = self.hinge_qpos
        info['hinge_diff'] = self.hinge_diff
        info['touched_handle'] = self.touched_handle
        info['percent_viapoints_'] = (np.abs(self.hinge_goal)-self.hinge_diff)/(np.abs(self.hinge_goal))

        done = done or self._check_terminated()

        if self.data_logging:

            self.file_logging['percent_viapoints_'][self.counter-1] = (np.abs(self.hinge_goal)-self.hinge_diff)/(np.abs(self.hinge_goal))
            self.file_logging['hinge_angle'][self.counter-1] = self.hinge_qpos
            self.file_logging['hinge_diff'][self.counter-1] = self.hinge_diff
            self.file_logging['hinge_goal'][self.counter-1] = self.hinge_goal

            done = done or self._check_terminated()

            self.file_logging['done'][self.counter-1] = done

        return reward, done, info

class SawyerDoorEnv(DoorEnv):
    def __init__(
        self,
        **kwargs
        ): 

        super().__init__(
            robot_type = 'Sawyer',
            **kwargs
        )

class PandaDoorEnv(DoorEnv):
    def __init__(
        self,
        **kwargs
        ): 

        super().__init__(
            robot_type = 'Panda',
            **kwargs
        )