from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.sawyer import SawyerEnv
from robosuite.environments.sawyer_lift import SawyerLift

from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.robots import Sawyer
from robosuite.models.tasks import TableTopTask, UniformRandomSampler

from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.wrappers.ik_wrapper import IKWrapper


class GymSawyerLift(SawyerEnv):
    """
    This class corresponds to the lifting task for the Sawyer robot arm with IK
    """

    def __init__(
                self,
                gripper_type="TwoFingerGripper",
                table_full_size=(1., 1.3, 0.75),
                table_friction=(1., 5e-3, 1e-4),
                use_camera_obs=True,
                use_object_obs=True,
                reward_shaping=False,
                placement_initializer=None,
                gripper_visualization=False,
                use_indicator_object=False,
                has_renderer=False,
                has_offscreen_renderer=True,
                render_collision_mesh=False,
                render_visual_mesh=True,
                control_freq=10,
                horizon=1000,
                ignore_done=False,
                camera_name="frontview",
                camera_height=256,
                camera_width=256,
                camera_depth=False,
                keys=None,
                markov_obs=False,
                finger_obs=False
                ):
        
        #has_renderer=render
        #has_offscreen_renderer=image_state
        #use_camera_obs=image_state
        #reward_shaping=True
        #camera_name='agentview'
        #camera_height=84
        #camera_width=84

        self.env = SawyerLift(
            gripper_type=gripper_type,
            gripper_visualization=gripper_visualization,
            use_indicator_object=use_indicator_object,
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
            camera_width=camera_width,
            camera_depth=camera_depth,
            reward_shaping=reward_shaping,
        )

        self.env = IKWrapper(self.env, markov_obs=markov_obs, finger_obs=finger_obs)
        #self.action_space = self.env.action_space
        #self.observation_space = self.env.observation_space
        #self.keys = self.env.keys

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        return self.env._load_model()

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict: ordered dictionary of observations
        """
        return self.env._flatten_obs(obs_dict, verbose=verbose)
        
    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        return self.env._get_reference()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        return self.env._reset_internal()
        
    def reward(self, action=None):
        """
        Reward function for the task.
        """
        return self.env.reward(action=action)

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].
        """
        return self.env._get_observation()

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        return self.env._check_contact()

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        return self.env._check_success()

    def _gripper_visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """
        return self.env._gripper_visualization()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, *args, **kwargs):
        return self.env.render(**kwargs)

    def observation_spec(self):
        return self.env.observation_spec()

    def action_spec(self):
        return self.env.action_spec()

    def get_observation(self):
        return self.env.get_observation()

    @property
    def dof(self):
        return self.env.dof

    @property
    def unwrapped(self):
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def __getattr__(self, attr):
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.env:
                    return self
                return result

            return hooked
        else:
            return orig_attr
