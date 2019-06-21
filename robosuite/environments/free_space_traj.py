import numpy as np
from robosuite.environments.robot_arm import RobotArmEnv
from robosuite.models import *
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import Task, FreeSpaceTask
from robosuite.models.arenas import EmptyArena
import multiprocessing
from robosuite.environments.controller import *
from robosuite.utils import mjcf_utils
from baselines import logger

class FreeSpaceTrajEnv(RobotArmEnv):

    def __init__(
            self,
            use_object_obs,
            num_via_points,
            dist_threshold,
            timestep_penalty,
            via_point_reward,
            distance_reward_weight,
            distance_penalty_weight,
            use_delta_distance_reward,
            energy_penalty,
            ee_accel_penalty,
            use_debug_cube,
            use_debug_point,
            action_delta_penalty,
            use_debug_square,
            acc_vp_reward_mult,
            num_already_checked,
            end_bonus_multiplier,
            allow_early_end,
            random_point_order,
            point_randomization,
            randomize_initialization,
            placement_initializer=None,
            **kwargs
    ): 

        self.dist_threshold = dist_threshold
        self.timestep_penalty = timestep_penalty
        self.distance_reward_weight = distance_reward_weight
        self.distance_penalty_weight = distance_penalty_weight
        self.use_delta_distance_reward = use_delta_distance_reward
        self.via_point_reward = via_point_reward
        self.energy_penalty = energy_penalty
        self.ee_accel_penalty = ee_accel_penalty
        self.action_delta_penalty = action_delta_penalty
        self.acc_vp_reward_mult = acc_vp_reward_mult
        self.end_bonus_multiplier = end_bonus_multiplier
        self.allow_early_end = allow_early_end
        self.random_point_order = random_point_order
        self.point_randomization = point_randomization
        self.randomize_initialization = randomize_initialization

        # Note: should be mutually exclusive
        self.use_debug_cube = use_debug_cube
        self.use_debug_square = use_debug_square
        self.use_debug_point = use_debug_point

        # create ordered list of random points in 3D space the end-effector must touch
        self.num_already_checked = num_already_checked
        self.num_via_points = num_via_points
        self._place_points()
        self.next_idx = num_already_checked

        self.finished_time = None
        self.use_object_obs = use_object_obs

        super().__init__(**kwargs)

        if self.data_logging:
            # if the data_logging flag is set, the super class already has an h5py object as self.file_logging
            self.file_logging.create_dataset('distance_to_via_point', (self.data_count, 3), maxshape=(None, 3))
            self.file_logging.create_dataset('next_via_point_idx', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('current_via_point', (self.data_count, 3), maxshape=(None, 3))
            self.file_logging.create_dataset('distance_reward', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('acc_vp_reward', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('timestep_penalty', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('energy_penalty', (self.data_count, 1), maxshape=(None, 1))
            self.file_logging.create_dataset('ee_accel_penalty', (self.data_count, 1), maxshape=(None, 1))

    def _place_points(self):
        """
        Randomly generate via points to set self.via_points. Note that each item in self.via_points 
        is 4 elements long: the first element is a 1 if the point has been checked, 0 otherwise.
        The remaining 3 elements are the x, y and z position of the point.
        """
        min_vals = {'x': 0.4, 'y': -0.1, 'z': 1.4}
        max_vals = {'x': 0.6, 'y': 0.1, 'z': 1.6}

        def place_point(min_vals, max_vals):
            pos = []
            for axis in ['x', 'y', 'z']:
                pos.append(np.random.uniform(low=min_vals[axis], high=max_vals[axis]))
            return pos

        if self.use_debug_point:
            self.via_points = [np.array((0.5, 0.15, 1.4))]
            self.num_via_points = 1
        elif self.use_debug_square:
            box_1 = np.array((0.5, -0.15, 1.4))
            box_2 = box_1 + np.array((0.0, 0.3, 0.0))
            box_3 = box_2 + np.array((0.0, 0.0, -0.2))
            box_4 = box_1 + np.array((0.0, 0.0, -0.2))
            if self.random_point_order:
                if np.random.choice([True, False]):
                    # clockwise
                    self.via_points = [box_1, box_2, box_3, box_4]
                else:
                    # counter-clockwise
                    self.via_points = [box_1, box_4, box_3, box_2]
            else:
                # clockwise
                self.via_points = [box_1, box_2, box_3, box_4]
            if self.point_randomization != 0:
                # preserve constant x
                randomized_viapoints = []
                for p in self.via_points:
                    p[1:] += np.random.randn(2)*self.point_randomization
                    randomized_viapoints.append(p)
                self.via_points = randomized_viapoints
            self.num_via_points = len(self.via_points)
        elif self.use_debug_cube:
            self.via_points = [[min_vals['x'], max_vals['y'], max_vals['z']],
                               [min_vals['x'], min_vals['y'], max_vals['z']],
                               [max_vals['x'], min_vals['y'], max_vals['z']],
                               [max_vals['x'], max_vals['y'], max_vals['z']],
                               [min_vals['x'], min_vals['y'], min_vals['z']],
                               [min_vals['x'], max_vals['y'], min_vals['z']],
                               [max_vals['x'], min_vals['y'], min_vals['z']],
                               [max_vals['x'], max_vals['y'], min_vals['z']]]
            self.num_via_points = len(self.via_points)
        else:
            self.via_points = [place_point(min_vals, max_vals) for _ in range(self.num_via_points)]

        final_via_points = []
        for i, point in enumerate(self.via_points):
            final_via_points.append([1 if i < self.num_already_checked else 0, *point])
        self.via_points = np.array(final_via_points)

    def _load_model(self):
        """
        Load the Mujoco model for the robot and its surroundings, inserting the via points
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        self.robot_contact_geoms = self.mujoco_robot.contact_geoms

        # ensure both robots start at similar positions:
        if self.mujoco_robot._model_name == 'sawyer':
            # calculated via IK to match the panda
            self.mujoco_robot._init_qpos = np.array([-0.3069560907729503, -1.1316307809096735, -0.0444000938487099, 2.4458494773056896, 0.47625600105788357, 0.15048738525687322, -0.6397859996642663])
        elif self.mujoco_robot._model_name == 'panda':
            self.mujoco_robot._init_qpos = np.array([0, np.pi/16.0, 0.00, -np.pi/2.0 -np.pi/3.0 , 0.00, np.pi - 0.2, np.pi/4])

        # load model for workspace
        self.mujoco_arena = EmptyArena()
        self.model = FreeSpaceTask(self.mujoco_arena, self.mujoco_robot)

        # add sites for each point
        for i, data in enumerate(self.via_points):
            point = data[1:] # ignore element 0 (just indicates whether point has been pressed)

            # pick a color
            color = None
            if i < self.num_already_checked: color = mjcf_utils.GREEN
            elif i == self.num_already_checked: color = mjcf_utils.RED
            else: color = mjcf_utils.BLUE

            site = mjcf_utils.new_site(name='via_point_%d'%i,
                                       pos=tuple(point),
                                       size=(0.01,),
                                       rgba=color)
            self.model.worldbody.append(site)

    def _get_reference(self):
        super()._get_reference()

    def _reset_internal(self):
        super()._reset_internal()
        self._place_points()        
        # reset joint positions
        if self.randomize_initialization:
            self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(self.mujoco_robot.init_qpos + np.random.randn(7)*0.02)
        else:
            self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(self.mujoco_robot.init_qpos)
        self.next_idx = self.num_already_checked
        self.timestep = 0
        self.finished_time = None

    def reward(self, action):
        """
        Return the reward obtained for a given action. Overall, reward increases as the robot 
        checks via points in order.
        """
        reward = 0
        dist = np.linalg.norm(self.ee_pos[:3] - self.via_points[self.next_idx][1:])

        # check if robot hit the next via point
        if self.finished_time is None and dist < self.dist_threshold:
            self.sim.model.site_rgba[self.next_idx] = mjcf_utils.GREEN
            self.via_points[self.next_idx][0] = 1 # mark as visited
            self.next_idx += 1
            reward += self.via_point_reward

            # if there are still via points to go
            if self.next_idx != self.num_via_points:
                # color next target red
                self.sim.model.site_rgba[self.next_idx] = mjcf_utils.RED

        # reward for remaining distance
        else:
            # if robot starts 0.3 away and dist_threshold is 0.05: [0.005, 0.55] without scaling
            if not self.use_delta_distance_reward:
                reward += self.distance_reward_weight * (1 - np.tanh(5*dist)) # was 10
            else:
                prev_dist = np.linalg.norm(self.prev_ee_pos[:3] - self.via_points[self.next_idx][1:])
                reward += self.distance_reward_weight*(prev_dist - dist)
                reward -= self.distance_penalty_weight*np.tanh(10*dist)

        # What we want is to reach the points fast
        # We add a reward that is proportional to the number of points crossed already
        reward += self.next_idx*self.acc_vp_reward_mult

        # penalize for taking another timestep
        # (e.g. 0.001 per timestep, for a total of 4096 timesteps means a penalty of 40.96)
        reward -= self.timestep_penalty

        # penalize for jerkiness
        reward -= self.energy_penalty * np.sum(np.abs(self.joint_torques))
        reward -= self.ee_accel_penalty * np.mean(abs(self.ee_acc))
        reward -= self.action_delta_penalty * np.mean(abs(self._compute_a_delta()[:3]))

        return reward

    def _get_observation(self):
        di = super()._get_observation()

        if self.use_object_obs:
            # location of each via point, whether or not it's been reached
            di['object-state'] = self.via_points.flatten()

        if getattr(self.controller, 'use_delta_impedance', False):
            di['controller_kp'] = self.controller.impedance_kp
            di['controller_damping'] = self.controller.damping

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

    def _check_success(self):
        """
        Returns True if task is successfully completed
        """
        return self.next_idx == self.num_via_points

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

        # allow episode to finish early
        if self._check_success():
            reward += self.end_bonus_multiplier*(self.horizon - self.timestep)
            self.finished_time = self.timestep
            if self.allow_early_end:
                done = True
            else:
                # reset goal
                self.next_idx -= 1
                self.via_points[self.next_idx][0] = 0 # mark as not visited

        info['add_vals'].extend(['percent_viapoints_', 'finished_time', 'finished'])
        info['percent_viapoints_'] = self.next_idx/self.num_via_points if self.finished_time is None else 1
        info['finished'] = 1 if self.finished_time is not None else 0
        info['finished_time'] = self.finished_time if self.finished_time is not None else self.horizon

        logger.debug('Process {process_id}, timestep {timestep} reward: {reward:.2f}, checked vps: {viapoints}'.format(
            process_id = str(id(multiprocessing.current_process()))[-5:], 
            timestep = self.timestep, 
            reward=reward,
            viapoints=self.next_idx))

        if self.data_logging:
            # NOTE: counter is -1 because it is already incremented in robot_arm.py
            eef_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
            dist = np.linalg.norm(eef_position - self.via_points[self.next_idx][1:]) if not done else 0
            self.file_logging['reward'][self.counter-1] = reward
            self.file_logging['distance_to_via_point'][self.counter-1] = dist
            self.file_logging['next_via_point_idx'][self.counter-1] = self.next_idx if self.finished_time is None else -1
            if not done:
                self.file_logging['current_via_point'][self.counter-1] = self.via_points[self.next_idx][1:]
            self.file_logging['distance_reward'][self.counter-1] = [self.distance_reward_weight * (1 - np.tanh(5*dist)), self.via_point_reward][dist < self.dist_threshold]
            self.file_logging['acc_vp_reward'][self.counter-1] = self.next_idx*self.acc_vp_reward_mult
            self.file_logging['timestep_penalty'][self.counter-1] = -self.timestep_penalty
            self.file_logging['energy_penalty'][self.counter-1] = -self.energy_penalty * self.total_joint_torque
            self.file_logging['ee_accel_penalty'][self.counter-1] = -self.ee_accel_penalty * np.mean(abs(self.ee_acc))

        return reward, done, info

class SawyerFreeSpaceTrajEnv(FreeSpaceTrajEnv):
    def __init__(
        self,
        **kwargs
        ): 

        super().__init__(
            robot_type = 'Sawyer',
            **kwargs
        )

class PandaFreeSpaceTrajEnv(FreeSpaceTrajEnv):
    def __init__(
        self,
        **kwargs
        ): 

        super().__init__(
            robot_type = 'Panda',
            **kwargs
        )