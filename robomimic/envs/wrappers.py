"""
A collection of useful environment wrappers.
"""
from copy import deepcopy
import textwrap
import cv2
import numpy as np
from collections import deque

import robomimic.envs.env_base as EB
from robocasa.models.robots import (
    GROOT_ROBOCASA_ENVS_GR1_ARMS_ONLY,
    GROOT_ROBOCASA_ENVS_GR1_ARMS_AND_WAIST,
    GROOT_ROBOCASA_ENVS_GR1_FIXED_LOWER_BODY,
    gather_robot_observations,
    make_key_converter,
)

FINAL_IMAGE_RESOLUTION = (224, 224)
RESIZED_IMAGE_RESOLUTION = (720, 480)


class EnvWrapper(object):
    """
    Base class for all environment wrappers in robomimic.
    """
    def __init__(self, env):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
        """
        assert isinstance(env, EB.EnvBase) or isinstance(env, EnvWrapper)
        self.env = env

    @classmethod
    def class_name(cls):
        return cls.__name__

    def _warn_double_wrap(self):
        """
        Utility function that checks if we're accidentally trying to double wrap an env
        Raises:
            Exception: [Double wrapping env]
        """
        env = self.env
        while True:
            if isinstance(env, EnvWrapper):
                if env.class_name() == self.class_name():
                    raise Exception(
                        "Attempted to double wrap with Wrapper: {}".format(
                            self.__class__.__name__
                        )
                    )
                env = env.env
            else:
                break

    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (EnvBase instance): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env

    def _to_string(self):
        """
        Subclasses should override this method to print out info about the 
        wrapper (such as arguments passed to it).
        """
        return ''

    def __repr__(self):
        """Pretty print environment."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\nenv={}".format(self.env), indent)
        msg = header + '(' + msg + '\n)'
        return msg

    # this method is a fallback option on any methods the original env might support
    def __getattr__(self, attr):
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.env, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.env):
                    return self
                return result

            return hooked
        else:
            return orig_attr


class FrameStackWrapper(EnvWrapper):
    """
    Wrapper for frame stacking observations during rollouts. The agent
    receives a sequence of past observations instead of a single observation
    when it calls @env.reset, @env.reset_to, or @env.step in the rollout loop.
    """
    def __init__(self, env, num_frames):
        """
        Args:
            env (EnvBase instance): The environment to wrap.
            num_frames (int): number of past observations (including current observation)
                to stack together. Must be greater than 1 (otherwise this wrapper would
                be a no-op).
        """
        assert num_frames > 1, "error: FrameStackWrapper must have num_frames > 1 but got num_frames of {}".format(num_frames)

        super(FrameStackWrapper, self).__init__(env=env)
        self.num_frames = num_frames

        ### TODO: add action padding option + adding action to obs to include action history in obs ###

        # keep track of last @num_frames observations for each obs key
        self.obs_history = None

    def _get_initial_obs_history(self, init_obs):
        """
        Helper method to get observation history from the initial observation, by
        repeating it.

        Returns:
            obs_history (dict): a deque for each observation key, with an extra
                leading dimension of 1 for each key (for easy concatenation later)
        """
        obs_history = {}
        for k in init_obs:
            obs_history[k] = deque(
                [init_obs[k][None] for _ in range(self.num_frames)], 
                maxlen=self.num_frames,
            )
        return obs_history

    def _get_stacked_obs_from_history(self):
        """
        Helper method to convert internal variable @self.obs_history to a 
        stacked observation where each key is a numpy array with leading dimension
        @self.num_frames.
        """
        # concatenate all frames per key so we return a numpy array per key
        return { k : np.concatenate(self.obs_history[k], axis=0) for k in self.obs_history }

    def cache_obs_history(self):
        self.obs_history_cache = deepcopy(self.obs_history)

    def uncache_obs_history(self):
        self.obs_history = self.obs_history_cache
        self.obs_history_cache = None

    def reset(self):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset()
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def reset_to(self, state):
        """
        Modify to return frame stacked observation which is @self.num_frames copies of 
        the initial observation.

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
        """
        obs = self.env.reset_to(state)
        self.timestep = 0  # always zero regardless of timestep type
        self.update_obs(obs, reset=True)
        self.obs_history = self._get_initial_obs_history(init_obs=obs)
        return self._get_stacked_obs_from_history()

    def step(self, action):
        """
        Modify to update the internal frame history and return frame stacked observation,
        which will have leading dimension @self.num_frames for each key.

        Args:
            action (np.array): action to take

        Returns:
            obs_stacked (dict): each observation key in original observation now has
                leading shape @self.num_frames and consists of the previous @self.num_frames
                observations
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """
        obs, r, done, info = self.env.step(action)
        self.update_obs(obs, action=action, reset=False)
        # update frame history
        for k in obs:
            # make sure to have leading dim of 1 for easy concatenation
            self.obs_history[k].append(obs[k][None])
        obs_ret = self._get_stacked_obs_from_history()
        return obs_ret, r, done, info

    def update_obs(self, obs, action=None, reset=False):
        obs["timesteps"] = np.array([self.timestep])
        
        if reset:
            obs["actions"] = np.zeros(self.env.action_dimension)
        else:
            self.timestep += 1
            obs["actions"] = action[: self.env.action_dimension]

    def _to_string(self):
        """Info to pretty print."""
        return "num_frames={}".format(self.num_frames)



class ObservationMapperWrapper(EnvWrapper):
    """
    A wrapper that applies the map_obs_keys function to observations
    returned by the environment's reset and step methods.
    """
    
    def __init__(self, env, image_crop=True, image_crop_size=[310, 770, 110, 1130]):
        """
        Args:
            env (EnvBase): The environment to wrap
        """
        super(ObservationMapperWrapper, self).__init__(env=env)
        self.env = env
        
        # Forward all attributes from wrapped env
        for attr in dir(self.env):
            if not attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, getattr(self.env, attr))

        self.key_converter = make_key_converter(robots_name=self.env.env.robot_names[0])

        self.image_crop = image_crop
        self.image_crop_size = image_crop_size
        self.process_img = (
            self.process_img_w_crop if self.image_crop else self.process_img_no_crop
        )

    def process_img_no_crop(self, img):
        h, w, _ = img.shape
        if h != w:
            dim = max(h, w)
            y_offset = (dim - h) // 2
            x_offset = (dim - w) // 2
            img = np.pad(
                img,
                ((y_offset, y_offset), (x_offset, x_offset), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            h, w = dim, dim
        if (h, w) != FINAL_IMAGE_RESOLUTION:
            img = cv2.resize(img, FINAL_IMAGE_RESOLUTION, cv2.INTER_AREA)
        # Convert from (H, W, C) to (C, H, W)
        img = np.copy(
            (np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0).clip(0.0, 1.0)
        )
        return np.copy(img)

    def process_img_w_crop(self, img):
        h, w, _ = img.shape
        if (h, w) == FINAL_IMAGE_RESOLUTION:
            return img

        crop_size = self.image_crop_size
        # print(f"Cropping image to {crop_size}")
        img = img[crop_size[0] : crop_size[1], crop_size[2] : crop_size[3]]
        img_resized = cv2.resize(img, RESIZED_IMAGE_RESOLUTION, cv2.INTER_AREA)

        h, w = img_resized.shape[:2]
        if h != w:
            dim = max(h, w)
            y_offset = (dim - h) // 2
            x_offset = (dim - w) // 2
            img_resized = np.pad(
                img_resized,
                ((y_offset, y_offset), (x_offset, x_offset), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            h, w = dim, dim
        if (h, w) != FINAL_IMAGE_RESOLUTION:
            img_resized = cv2.resize(
                img_resized, FINAL_IMAGE_RESOLUTION, cv2.INTER_AREA
            )

        # Convert from (H, W, C) to (C, H, W)
        img_resized = np.copy(
            (np.transpose(img_resized, (2, 0, 1)).astype(np.float32) / 255.0).clip(
                0.0, 1.0
            )
        )
        return np.copy(img_resized)

    def get_basic_observation(self, raw_obs):
        raw_obs.update(gather_robot_observations(self.env))
        
        # Image are in (H, W, C), flip it upside down
        def process_img(img):
            print(f"Processing image {img.shape}")
            return np.copy(img[::-1, :, :])

        for obs_name, obs_value in raw_obs.items():
            if obs_name.endswith("_image"):
                # image observations
                raw_obs[obs_name] = process_img(obs_value)
            else:
                # non-image observations
                raw_obs[obs_name] = obs_value.astype(np.float32)

        self.render_cache = raw_obs[self.render_camera + "_image"]
        
        raw_obs["language"] = self.env.get_ep_meta().get("lang", "")

        return raw_obs

    def get_gearbc_observation(self, raw_obs, reward=-1):
        obs = {}
        temp_obs = self.key_converter.map_obs(raw_obs)
        for k, v in temp_obs.items():
            if k.startswith("hand.") or k.startswith("body."):
                obs[k[5:]] = v
            else:
                raise ValueError(f"Unknown key: {k}")
        mapped_names, camera_names, _, _ = self.key_converter.get_camera_config()
        for mapped_name, camera_name in zip(mapped_names, camera_names):
            obs[camera_name + "_image"] = self.process_img(
                raw_obs[camera_name + "_image"]
            )
        self._ep_lang_str = raw_obs["language"]
        return obs

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        raw_obs = self.env.env.reset()  # skip the EnvRobosuite wrapper
        # return obs
        raw_obs = self.get_basic_observation(raw_obs)

        # info = {}
        # info["success"] = False

        obs = self.get_gearbc_observation(raw_obs)
        return obs

    def step(self, action):
        temp_action = action.copy()
        action = {}
        for k, v in temp_action.items():
            assert k.endswith("_action")
            action["action." + k[:-7]] = v
        # for k, v in action.items():
        #     self.verbose and print("<ACTION>", k, v)

        self.success = False
        import ipdb; ipdb.set_trace(context=10)
        # action = self.key_converter.unmap_action(action)
        raw_obs, reward, terminated, truncated, info = self.env.env.step(action)  # skip the EnvRobosuite wrapper
        raw_obs = self.get_basic_observation(raw_obs)
        obs = self.get_gearbc_observation(raw_obs, reward)

        # if `reward > 0:
        #     import os
        #     import random
        #     import string
        #     # save the render cache
        #     random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        #     os.makedirs(f"/mnt/amlfs-01/home/runyud/workspace/outputs/random", exist_ok=True)
        #     cv2.imwrite(f"`/mnt/amlfs-01/home/runyud/workspace/outputs/random/render_{random_str}.png", self.render_cache[..., ::-1])

        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """
        Fallback attribute access to the wrapped environment.
        """
        return getattr(self.env, name) 