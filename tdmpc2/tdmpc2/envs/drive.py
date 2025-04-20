from torchdriveenv.env_utils import load_default_train_data, load_default_validation_data
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np

import torch
from dataclasses import dataclass, field
from enum import Enum
from omegaconf import OmegaConf
from typing import Optional

from torchdriveenv.gym_env import EnvConfig
from torchdriveenv.env_utils import construct_env_config

from gymnasium import spaces # or gym.spaces
from collections import deque


class BaselineAlgorithm(Enum):
    sac = 'sac'
    ppo = 'ppo'
    a2c = 'a2c'
    td3 = 'td3'
    grpo = 'grpo'

@dataclass
class RlCallbackConfig:
    n_steps: int = 1000
    eval_n_episodes: int = 10
    deterministic: bool = True 
    record: bool = True

@dataclass
class WandbCallbackConfig: 
    verbose: bool = True
    gradient_save_freq: int = 100
    model_save_freq: int = 100

@dataclass
class RlTrainingConfig:
    algorithm: BaselineAlgorithm = None
    parallel_env_num: Optional[int] = 2
    project: str = "stable_baselines3" 
    total_timesteps: int = 5e6 
    record_training_examples: bool = True 
    env: EnvConfig = field(default_factory=lambda:EnvConfig())
    eval_train_callback: RlCallbackConfig = field(default_factory=lambda:RlCallbackConfig())
    eval_val_callback: RlCallbackConfig = field(default_factory=lambda:RlCallbackConfig())
    wandb_callback: WandbCallbackConfig = field(default_factory=lambda:WandbCallbackConfig())

def load_rl_training_config(yaml_path):
    config_from_yaml = OmegaConf.to_object(OmegaConf.load(yaml_path))
    rl_training_config = RlTrainingConfig(**config_from_yaml)
    rl_training_config.algorithm = BaselineAlgorithm(rl_training_config.algorithm)
    rl_training_config.env = construct_env_config(rl_training_config.env)
    
    return rl_training_config


class FrameStackingWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque([], maxlen=num_stack)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3*num_stack, 64, 64),
            dtype=env.observation_space.dtype
        )

    def _get_observation(self):
        """Returns the stacked frames as a single numpy array."""
        assert len(self.frames) == self.num_stack, "Not enough frames in deque!"
        # Stack frames along a new dimension (axis=0)
        return torch.from_numpy(np.concatenate(self.frames))

    def reset(self):
        """Resets the environment and initializes the frame buffer."""
        obs = self.env.reset() # Sync seed with parent wrapper if needed

        # Initialize the deque with the first observation
        for _ in range(self.num_stack):
            self.frames.append(obs)

        return self._get_observation()

    def step(self, action):
        """Steps the environment, adds the new frame, and returns the stacked observation."""
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs) # Add the *new* frame to the deque
        return self._get_observation(), reward, done, info


class DriveWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
		
    def reset(self):
        return self.env.reset()[0]

	
    def step(self, action):

        action[1] /= (10/3)

        for _ in range(2):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

        info['success'] = info['is_success']

        if done:
            info['terminated'] = True
        else:
            info['terminated'] = False

        return obs, reward, done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped



def make_env(cfg):

    rl_training_config = load_rl_training_config(cfg.env_config_path)
    env_config = rl_training_config.env

    training_data = load_default_train_data()
    eval_data = load_default_validation_data()

    if x:
        env = gym.make('torchdriveenv-v0', args={'cfg': env_config, 'data': training_data})
    else:
        env = gym.make('torchdriveenv-v0', args={'cfg': env_config, 'data': eval_data})
        
    env = Monitor(env)
    env = DriveWrapper(env)
    env = FrameStackingWrapper(env, cfg.num_frames)

    return env