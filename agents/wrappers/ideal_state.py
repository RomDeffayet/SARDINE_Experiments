"""Fully observable MDP with ideal state."""
from typing import Tuple, Dict
import numpy as np
import gymnasium as gym


class IdealState(gym.Wrapper):
    '''
        Observable variant of SlateSim, i.e., the user state is known.
    '''
    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)

        ### Observation = [cur_user_embedd, norm_recent_topics_hist, norm_bored_timeout]
        self.observation_space = gym.spaces.Box(low = 0, high = 1, shape=(env.num_topics * 3,), dtype=np.float32)

        # Note: when boredom_influence = "item" the user embedding will be static

    def reset(self, seed = None, options = None) -> Tuple[np.ndarray, Dict]:
        _, info = super().reset(seed = seed)
        return info["user_state"], info

    def step(self, slate) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        _, reward, terminated, truncated, info = super().step(slate)
        return info["user_state"], reward, terminated, truncated, info