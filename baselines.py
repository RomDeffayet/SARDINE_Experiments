import gymnasium as gym
import numpy as np
import os
import csv
import sardine
from agents.wrappers import IdealState

list_env_ids = ["SlateTopK-Uncertain-v0"]#, "SlateRerank-Static-v0"]
#list_env_ids = ["SingleItem-Static-v0", "SingleItem-BoredInf-v0", "SingleItem-Uncertain-v0", "SingleItem-PartialObs-v0", "SlateTopK-Bored-v0", "SlateTopK-BoredInf-v0", "SlateTopK-PartialObs-v0"]#, "SlateRerank-Static-v0"]
list_methods = ["greedyoracle", "random"]
list_seeds = [2705, 3751, 4685, 3688, 6383]
n_val_episodes = 25
total_timesteps = 500000
val_interval = 50000

for env_id in list_env_ids:
    # Make the environment
    env = gym.make(env_id)
    env_name = "-".join(env_id.lower().split("-")[:-1])

    ## If you want to work with fully observable state, add a wrapper to the environment
    env = IdealState(env)

    ## Evaluate the baseline on the simulator. Here an example with a random agent.
    for method in list_methods:
        for seed in list_seeds:
            # CSV logger
            csv_filename = "run_" + env_name + "-" + method + "-0-" + str(seed) + ".log"
            csv_path = "logs/" + csv_filename
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            csv_file = open(csv_path, "w+", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["field", "value", "step"])

            # Environment initialization
            observation, info = env.reset(seed = seed)
            env.action_space.seed(seed)

            # Run the agent on the environment
            val_returns, val_boredom = [], []
            for _ in range(n_val_episodes):
                cum_reward, cum_boredom = 0, 0
                terminated, truncated = False, False
                while not (terminated or truncated):
                    #action = env.action_space.sample()
                    if method == "greedyoracle": # Greedy oracle
                        action = - np.ones(env.unwrapped.slate_size, dtype = int)
                    elif method == "random": # Random slate
                        action = -2 * np.ones(env.unwrapped.slate_size, dtype = int)
                    observation, reward, terminated, truncated, info = env.step(action)

                    cum_reward += reward
                    cum_boredom += (1.0 if np.sum(info["bored"] == True) > 0 else 0.0)

                    if terminated or truncated:
                        observation, info = env.reset()

                val_returns.append(cum_reward)
                val_boredom.append(cum_boredom)

            for i in range(total_timesteps // val_interval + 1):
                global_step = i * val_interval # Simulate a global step as in trained agents for plotting purposes
                csv_writer.writerow(["val_charts/episodic_return", np.mean(val_returns), global_step])
                csv_writer.writerow(["val_charts/boredom", np.mean(val_boredom), global_step])
                #print(["val_charts/episodic_return", np.mean(val_returns), global_step])
                #print(["val_charts/boredom", np.mean(val_boredom), global_step])

            print(method, "--", seed, "-- summary: cum_reward =", np.mean(val_returns), "/ cum_boredom =", np.mean(val_boredom))

    env.close()
