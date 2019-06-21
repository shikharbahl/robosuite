import numpy as np
from collections import OrderedDict
from robosuite.utils import RandomizationError
from robosuite.environments.robot_arm import RobotArmEnv
from robosuite.models import *
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import Task, TableTopTask, UniformRandomSampler, WipingTableTask
from robosuite.models.arenas import TableArena, WipingTableArena
from robosuite.models.objects import CylinderObject, PlateWithHoleObject, BoxObject
import multiprocessing
from robosuite.environments.controller import *

class WipePegsEnv(RobotArmEnv):

    def __init__(
        self,
        table_full_size=(0.5, 0.5, 0.8),
        table_friction=(1, 0.005, 0.0001),
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        single_object_mode=0,
        object_type=None,
        use_indicator_object=False,
        num_wiping_obj=100,
        use_impedance=True,
        arm_collision_penalty = -20,
        wipe_contact_reward= 0.5,
        unit_wiped_reward=20,
        **kwargs
        ): 
        """
            @gripper_type, string that specifies the gripper type
            @use_eef_ctrl, position controller or default joint controllder
            @table_size, full dimension of the table
            @table_friction, friction parameters of the table
            @use_camera_obs, using camera observations
            @use_object_obs, using object physics states
            @camera_name, name of camera to be rendered
            @camera_height, height of camera observation
            @camera_width, width of camera observation
            @camera_depth, rendering depth
            @reward_shaping, using a shaping reward
        """

        # settings for the reward
        self.arm_collision_penalty = arm_collision_penalty
        self.wipe_contact_reward= wipe_contact_reward
        self.unit_wiped_reward = unit_wiped_reward

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        self.num_wiping_obj = num_wiping_obj

        # whether to include and use ground-truth object states
        self.use_object_obs = use_object_obs

        # whether to include and use ground-truth proprioception in the observation
        self.observe_robot_state = True

        # reward configuration
        self.reward_shaping = reward_shaping

        # object placement initializer\
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[0, 0.2], y_range=[0, 0.2],
                ensure_object_boundary_in_range=False,
                z_rotation=True)

        super(WipePegsEnv,self).__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        self.robot_contact_geoms = self.mujoco_robot.contact_geoms

        # load model for table top workspace
        self.mujoco_arena = WipingTableArena(table_full_size=self.table_full_size,
                                       table_friction=self.table_friction
                                       )

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.26 + self.table_full_size[0] / 2,0,0])

        self.mujoco_objects = OrderedDict()

        for i in range(self.num_wiping_obj):
            peg = BoxObject(size=[0.01, 0.01, 0.01],
                               rgba=[0, 1, 0, 1],
                               density=500,
                               friction=0.05)
            # peg = CylinderObject(size=[0.010, 0.005],
            #                     rgba=[0, 1, 0, 1])
            self.mujoco_objects['peg'+str(i)] = peg

        # task includes arena, robot, and objects of interest
        self.model = WipingTableTask(self.mujoco_arena, 
                                self.mujoco_robot, 
                                self.mujoco_objects,
                                initializer=self.placement_initializer)
        self.model.place_objects()

    def _get_reference(self):
        super()._get_reference()

        self.peg_body_ids = []
        for i in range(self.num_wiping_obj):
            self.peg_body_ids += [self.sim.model.body_name2id('peg'+str(i))]
        self.peg_geom_ids = []
        for i in range(self.num_wiping_obj):
            self.peg_geom_ids += [self.sim.model.geom_name2id('peg'+str(i)+"_col")]

    def _reset_internal(self):
        super()._reset_internal()
        # inherited class should reset positions of objects
        self.model.place_objects()
        # reset joint positions
        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(self.mujoco_robot.init_qpos)
        self.timestep = 0
        self.collisions = 0

    def reward(self, action):
        reward = 0

        self.table_id = self.sim.model.geom_name2id('table_visual')
        table_height = self.table_full_size[2]
        table_location = self.mujoco_arena.table_top_abs
        table_x_border_plus = table_location[0] + self.table_full_size[0]*0.5
        table_x_border_minus = table_location[0] - self.table_full_size[0]*0.5
        table_y_border_plus = table_location[1] + self.table_full_size[1]*0.5
        table_y_border_minus = table_location[1] - self.table_full_size[1]*0.5       

        if self._check_arm_contact():
            reward = self.arm_collision_penalty
            self.collisions += 1
        else:
            force_sensor_id = self.sim.model.sensor_name2id("force_ee")
            force_ee = self.sensor_data[force_sensor_id*3: force_sensor_id*3+3]

            #Reward from wiping the table
            peg_heights = []
            for i in range(self.num_wiping_obj):
                if self.sim.data.body_xpos[self.peg_body_ids[i]][2] < (table_height - 0.1) :
                    reward += self.unit_wiped_reward

                # rewards moving the peg towards the border
                else:
                    # Find the closest x border
                    peg_location = self.sim.data.body_xpos[self.peg_body_ids[i]]
                    dist_to_center_x = peg_location[0] - table_location[0]
                    dist_to_center_y = peg_location[1] - table_location[1]
                    dist_to_border_x = [table_x_border_plus,table_x_border_minus][dist_to_center_x < 0] - dist_to_center_x
                    dist_to_border_y = [table_y_border_plus,table_y_border_minus][dist_to_center_y < 0] - dist_to_center_x

                    dist_to_x_border_minus = abs(table_x_border_minus - peg_location[0])

                    # Maximum of 20, minimum of -20, linear with distance to lower x border
                    reward += max(20*(1 - dist_to_x_border_minus/(0.5*self.table_full_size[0])),0)

            #Continuous reward from getting closer to the pegs that are on the table
            gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
            for i in range(self.num_wiping_obj):
                if self.sim.data.body_xpos[self.peg_body_ids[i]][2] > (table_height - 0.1) :
                    peg_pos = np.array(self.sim.data.body_xpos[self.peg_body_ids[i]])
                    peg_rew = min(1e3, 1.0/np.linalg.norm(gripper_site_pos - peg_pos))
                    peg_rew = 1 * (1 - np.tanh(10*np.linalg.norm(gripper_site_pos - peg_pos)))
                    reward += 0.01*peg_rew

            # Reward for keeping contact
            # if self.sim.data.ncon != 0 :
            if np.linalg.norm(np.array(force_ee)) > 1:
                reward += self.wipe_contact_reward

        print('Process %i, reward timestep %i: %5.4f' % (id(multiprocessing.current_process()) ,self.timestep, reward))#'reward ', reward)
        return reward

    def _get_observation(self):
        di = super()._get_observation()

        # object information in the observation
        if self.use_object_obs:
            # position of objects to wipe
            acc = np.array([])
            for i in range(self.num_wiping_obj):
                peg_pos = np.array(self.sim.data.body_xpos[self.peg_body_ids[i]])
                di['peg' + str(i) + '_pos'] = peg_pos
                acc = np.concatenate([acc, di['peg' + str(i) + '_pos'] ])
                # proprioception
                if self.use_proprio_obs:
                    gripper_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
                    di['gripper_to_peg'+str(i)] = gripper_position - peg_pos
                    acc = np.concatenate([acc, di['gripper_to_peg' + str(i)] ])
            di['object-state'] = acc       

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

        # If all the pegs are wiped off
        # cube is higher than the table top above a margin
        terminated = True

        table_height = self.table_size[2]
        for i in range(self.num_wiping_obj):
            if self.sim.data.body_xpos[self.peg_body_ids[i]][2] > (table_height-0.10) :
                terminated = False

        # If any other part of the robot (not the wiping) touches the table
        # if len([c for c in self.find_contacts(['table_collision'],self.robot_contact_geoms)]) > 0:
        #    terminated = True

        if terminated:
            print(60*"*")
            print("TERMINATED")
            print(60*"*")

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
        ret = super()._post_action(action)

        return ret

class SawyerWipePegsEnv(WipePegsEnv):
    def __init__(
        self,
        **kwargs
        ): 

        super().__init__(
            robot_type = 'Sawyer',
            **kwargs
        )

class PandaWipePegsEnv(WipePegsEnv):
    def __init__(
        self,
        **kwargs
        ): 

        super().__init__(
            robot_type = 'Panda',
            **kwargs
        )