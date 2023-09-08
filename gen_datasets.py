from pathlib import Path
import torch

import gymnasium as gym
import numpy as np

import sardine
from sardine.wrappers import IdealState
from sardine.policies import EpsilonGreedyOracle, EpsilonGreedyAntiOracle

seed = 2023
env_id = "SlateRerank-Bored-v0"
data_dir = "/scratch/2/user/rdeffaye/SlateRL/datasets/"
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(data_dir + "embeddings/").mkdir(parents=True, exist_ok=True)
lp = "antioracle"
eps = 0.0

## Let's create the environment of our choice
env = gym.make(env_id)

## If you want to work with Fully observable state, add a wrapper to the environment
env = IdealState(env)

## Generate a dataset of 10 users with 50% random actions and 50% greedy actions
if lp == "oracle":
    logging_policy = EpsilonGreedyOracle(epsilon = eps, env = env, seed = seed)
elif lp == "antioracle":
    logging_policy = EpsilonGreedyAntiOracle(epsilon = eps, env = env, seed = seed)
dataset = env.generate_dataset(n_users = 10000, policy = logging_policy, seed = seed, dataset_type="sb3_replay")
print(dataset.size())
print(dataset.sample(batch_size = 2))

path = env_id + "_" + lp + "_epsilon" + str(eps) + "_seed" + str(seed) + ".pt"
torch.save(dataset, data_dir + path)
torch.save(env.item_embedd, data_dir + "embeddings/" + path)