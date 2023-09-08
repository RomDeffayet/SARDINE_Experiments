"""Generative Modeling of Slates."""
from typing import Tuple
import numpy as np
import torch
import gymnasium as gym

class GeMS(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        path: str,
        device: torch.device,
        decoder: torch.nn.Module,
    ):
        super().__init__(env)

        if decoder is None:
            self.decoder = torch.load(path).to(device)
        else:
            self.decoder = decoder
        self.slate_size = env.slate_size
        self.latent_dim = self.decoder.latent_dim

        min_action, max_action = self._get_action_bounds()
        self.action_space = gym.spaces.Box(low = np.zeros(self.latent_dim, dtype=np.float32) + min_action,
                                        high = np.zeros(self.latent_dim, dtype=np.float32) + max_action,
                                        shape=(self.latent_dim,), dtype=np.float32)

    def _get_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return -2, 2

    def action(self, action: torch.FloatTensor) -> np.ndarray:
        with torch.inference_mode():
            logits = self.decoder(action)
        slate = logits[0].squeeze(dim=0).argmax(dim = -1).cpu().numpy()
        self.latest_slate = slate
        return slate
