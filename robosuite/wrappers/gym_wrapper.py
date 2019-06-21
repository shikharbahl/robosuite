"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from robosuite.wrappers import Wrapper
from collections import deque 


class GymWrapper(Wrapper):
    env = None

    def __init__(self, env, keys=None, obs_stack_size =1):
        """
        Initializes the Gym wrapper.

        Args:
            env (MujocoEnv instance): The environment to wrap.
            keys (list of strings): If provided, each observation will
                consist of concatenated keys from the wrapped environment's
                observation dictionary. Defaults to robot-state and object-state.
        """
        self.env = env

        if keys is None:
            assert self.env.use_object_obs, "Object observations need to be enabled."
            keys = ["robot-state", "object-state"]
        self.keys = keys
        self.obs_stack_size = obs_stack_size
        self.obs_stack = deque()

        # set up observation and action spaces
        flat_ob = self._flatten_obs(self.env.reset(), verbose=True)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict: ordered dictionary of observations
        """
        ob_lst = []
        for key in obs_dict:
            if key in self.keys:
                if verbose:
                    print("adding key: {}".format(key))
                if len(np.array(obs_dict[key]).shape) == 1: #vector
                    ob_lst.append(obs_dict[key])
                else:
                    ob_lst.append(np.array(obs_dict[key]).flatten())
        concatenated = np.concatenate(ob_lst)

        # Managing the stack of observations
        if not self.obs_stack:
            for repeated_first_obs in range(self.obs_stack_size):
                self.obs_stack.append(concatenated)
        else:
            self.obs_stack.pop()
            self.obs_stack.appendleft(concatenated)

        # Read current stack of observations
        obs = []
        for ob in self.obs_stack:
            obs.append(ob)
        obs = np.concatenate(obs)

        return obs

    def reset(self):
        self.obs_stack = deque()
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info
