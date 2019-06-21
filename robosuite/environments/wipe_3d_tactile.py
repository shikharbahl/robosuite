import numpy as np
from collections import OrderedDict
from robosuite.utils import RandomizationError
from robosuite.environments.robot_arm import RobotArmEnv
from robosuite.models import *
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.tasks import Task, UniformRandomSampler, HeightTableTask
from robosuite.models.arenas import HeightTableArena
from robosuite.models.objects import CylinderObject, PlateWithHoleObject, BoxObject
import multiprocessing
import mujoco_py
import copy
from robosuite.environments.controller import *
import imageio

class Wipe3DTactileEnv(RobotArmEnv):

    def __init__(
        self,
        table_height_full_size=(0.5, 0.5, 0.4, 0.8),
        table_friction=(1, 0.005, 0.0001),
        use_object_obs=True,
        reward_shaping=False,
        placement_initializer=None,
        arm_collision_penalty = -20,
        wipe_contact_reward= 0.5,
        unit_wiped_reward=20,
        touch_threshold= 1,
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
        self.table_full_size = table_height_full_size
        self.table_friction = table_friction

        # whether to include and use ground-truth object states
        self.use_object_obs = use_object_obs

        # whether to include and use ground-truth proprioception in the observation
        self.observe_robot_state = True

        # reward configuration
        self.reward_shaping = reward_shaping

        self.sites_counter = 0
        self.sites_max = 10000

        self.wiped_sensors = []
        self.touch_threshold= touch_threshold

        # object placement initializer\
        if placement_initializer:
            self.placement_initializer = placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                x_range=[0, 0.2], y_range=[0, 0.2],
                ensure_object_boundary_in_range=False,
                z_rotation=True)

        super(Wipe3DTactileEnv,self).__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        self.robot_contact_geoms = self.mujoco_robot.contact_geoms

        self.mujoco_arena = HeightTableArena(table_height_full_size=self.table_full_size,
                                       table_friction=self.table_friction,
                                       )

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.50 + self.table_full_size[0] / 2,0,0])

        self.mujoco_objects = OrderedDict()

        # task includes arena, robot, and objects of interest
        self.model = HeightTableTask(self.mujoco_arena, 
                                self.mujoco_robot, 
                                self.mujoco_objects,
                                initializer=self.placement_initializer)
        self.model.place_objects()

        # import xml.etree.ElementTree as ET

        # print(ET.tostring(self.model.asset.find("./hfield[@name='table_hf']")))
        # print(self.model.asset.find("./hfield[@name='table_hf']").get("file"))
        # exit()

        # print(self.model.get_xml())
        # exit()

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
        self.collisions = 0

    def reward(self, action):
        reward = 0    

        # Neg Reward from collisions of the arm with the table
        if len([c for c in self.find_contacts(['table_hf_geom'],self.robot_contact_geoms)]) > 0:
            reward = -100
        # TODO: Careful! The else here indicates that if the robot is colliding with the table it can wipe anything!
        else:
            #TODO: Use the sensed touch to shape reward
            force_sensor_id = self.sim.model.sensor_name2id("force_ee")
            force_ee = self.sensor_data[force_sensor_id*3: force_sensor_id*3+3]

            # Only do computation if there are active sensors and they weren't active before
            sensors_active_ids = np.argwhere(self.sensor_data > self.touch_threshold).flatten()
            new_sensors_active_ids = sensors_active_ids[np.where( np.isin(sensors_active_ids, self.wiped_sensors, invert=True))]
            if np.any(new_sensors_active_ids):
                ee_pos = self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')]

                # Build a list of contact points
                contact_points = []
                for i in range(self.sim.data.ncon):
                    # Note that the contact array has more than `ncon` entries,
                    # so be careful to only read the valid entries.
                    contact = self.sim.data.contact[i]

                    if self.sim.model.geom_id2name(contact.geom1) in ["table_hf_geom", "wiping_surface"] \
                        and self.sim.model.geom_id2name(contact.geom2) in ["table_hf_geom", "wiping_surface"]:

                        contact_points += [contact.pos]

                for i in new_sensors_active_ids:
                    #HACKY FIX: some sensors far away trigger when they shouldn't. Why?
                    #The fix is to measure distance to the contact points and count contact only if the sensor is close enough to a contact

                    sensor_too_far = True
                    sensor_to_contact_distance_th = 0.05
                    for contact_point in contact_points:
                        if np.linalg.norm(np.array(contact_point)- np.array(self.sim.model.site_pos[i])) < sensor_to_contact_distance_th:
                            sensor_too_far = False
                            break

                    if not sensor_too_far:
                        #print("Contact force in square " + str(i) + " " + str(j) + " " + str(force_in_ij) + " Newton")
                        self.sim.model.site_rgba[i] = [1, 1, 1, 1]
                        self.wiped_sensors += [i]
                        reward += self.unit_wiped_reward

            reward += len(self.wiped_sensors)

            # Reward for keeping contact
            # if self.sim.data.ncon != 0 :
            if np.linalg.norm(np.array(force_ee)) > 1:
                reward += self.wipe_contact_reward

        print('Process %i, timestep %i: reward: %5.4f wiped sensors: %i collisions: %i' % (id(multiprocessing.current_process()) ,self.timestep, reward, len(self.wiped_sensors), self.collisions))#'reward ', reward)
        return reward

    def _get_observation(self):
        di = super()._get_observation()

        # # object information in the observation
        if self.use_object_obs:
            # position of objects to wipe
            acc = np.array([])
            for i in range(len(self.sensor_data)):
                sensor_position = np.array(self.sim.model.site_pos[i])
                di['sensor' + str(i) + '_pos'] = sensor_position
                acc = np.concatenate([acc, di['sensor' + str(i) + '_pos'] ])
                acc = np.concatenate([acc, [[0,1][i in self.wiped_sensors]] ])
                # proprioception
                if self.observe_robot_state:
                    gripper_position = np.array(self.sim.data.body_xpos[self.sim.model.body_name2id('right_hand')])
                    di['gripper_to_sensor'+str(i)] = gripper_position - sensor_position
                    acc = np.concatenate([acc, di['gripper_to_sensor' + str(i)] ])
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

class SawyerWipe3DTactileEnv(Wipe3DTactileEnv):
    def __init__(
        self,
        **kwargs
        ): 

        super().__init__(
            robot_type = 'Sawyer',
            **kwargs
        )

class PandaWipe3DTactileEnv(Wipe3DTactileEnv):
    def __init__(
        self,
        **kwargs
        ): 

        super().__init__(
            robot_type = 'Panda',
            **kwargs
        )