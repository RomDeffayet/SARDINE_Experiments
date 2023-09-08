import argparse
import os
import csv
import random
import time
from distutils.util import strtobool
import sardine
import gymnasium as gym
import numpy as np

# import scipy.optimize as sco

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .buffer import RolloutBuffer, DictRolloutBuffer
from .wrappers import IdealState, TopK, GeMS
from .state_encoders import GRUStateEncoder, TransformerStateEncoder

from utils.parser import get_generic_parser
from utils.file import hash_config, args2str

torch.set_float32_matmul_precision('high')

def get_parser(parents = []):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    # Training arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="SlateBorNoInf-ucar-v0",
        help="the id of the environment",
    )
    parser.add_argument(
        "--observable",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, an observation with full state environment will be passed.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=50000,
        help="Number of timesteps between validation episodes.",
    )
    parser.add_argument(
        "--n-val-episodes",
        type=int,
        default=10,
        help="Number of validation episodes.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=int(100),
        help="the replay memory buffer size",
    )
    # REINFORCE arguments
    parser.add_argument(
        "--gamma", type=float, default=0.8, help="the discount factor gamma"
    )
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=3e-4,
        help="the learning rate of the policy network optimizer",
    )
    parser.add_argument(
        "--policy-frequency",
        type=int,
        default=2,
        help="the frequency of training policy (delayed)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Number of neurons in hidden layers of all models.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=None,
        help="State dimension in POMDP settings.",
    )
    parser.add_argument(
        "--sampled-seq-len",
        type=int,
        default=100,
        help="State dimension in POMDP settings.",
    )
    parser.add_argument(
        "--state-encoder",
        type=str,
        default="gru",
        choices=["gru", "transformer"],
        help="Type of state encoder (only for POMDP)",
    )
    parser.add_argument(
        "--ideal-se",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Ideal embeddings used in the state encoder",
    )
    parser.add_argument(
        "--shared-se",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Shared state encoder across all actors and belief networks",
    )
    parser.add_argument(
        "--item-dim-se",
        type=int,
        default=16,
        help="Dimension of item embeddings in the state encoder.",
    )
    parser.add_argument(
        "--click-dim-se",
        type=int,
        default=2,
        help="Dimension of click embeddings in the state encoder.",
    )
    parser.add_argument(
        "--num-layers-se",
        type=int,
        default=2,
        help="Number of layers in the state encoder.",
    )
    parser.add_argument(
        "--num-heads-se",
        type=int,
        default=4,
        help="Number of heads in the state encoder (only for Transformer).",
    )
    parser.add_argument(
        "--dropout-rate-se",
        type=float,
        default=0.1,
        help="Dropout rate in the state encoder (only for Transformer).",
    )
    parser.add_argument(
        "--forward-dim-se",
        type=int,
        default=64,
        help="Feed-forward net dimension in the state encoder (only for Transformer).",
    )
    return parser


def make_env(
    env_id,
    idx,
    observable,
    args,
):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if observable:
            env = IdealState(env)
        return env

    return thunk

class Actor(nn.Module):
    def __init__(self, env, hidden_size, state_dim, num_items, slate_size):
        super().__init__()

        if state_dim is None:
            state_dim = np.array(env.single_observation_space.shape).prod()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_items)

        self.slate_size = slate_size
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, torch.sqrt(torch.tensor(2)))
            m.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = nn.Softmax(dim=-1)(self.fc3(x))
        return action_probs

    def get_action(self, x, return_prob = False):
        probs = self(x)
        action = torch.multinomial(probs, self.slate_size)
        if return_prob:
            return action, probs.gather(-1, action)
        else:
            return action
    
    def get_probs(self, x, actions):
        probs = self(x)
        return probs.gather(-1, actions)

def train(args, decoder = None):
    run_name = f"{args.env_id}__{args.run_name}__{args.seed}__{int(time.time())}"
    if args.track == "wandb":
        import wandb
    elif args.track == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # CSV logger
    csv_filename = str(args.run_name) + "-" + hash_config(args, index=True) + "-" + str(args.seed) + ".log"
    csv_path = "logs/" + csv_filename
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, "w+", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["field", "value", "step"])

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                0,
                args.observable,
                args,
            )
        ]
    )
    val_envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                0,
                args.observable,
                args,
            )
        ]
    )

    slate_size = envs.envs[0].unwrapped.slate_size
    num_items = envs.envs[0].unwrapped.num_items
    actor = Actor(envs, args.hidden_size, args.state_dim, num_items, slate_size).to(args.device)
    actor_params = list(actor.parameters())

    if not args.observable:
        if args.state_encoder == "gru":
            StateEncoder = GRUStateEncoder
        elif args.state_encoder == "transformer":
            StateEncoder = TransformerStateEncoder
        else:
            StateEncoder = None
        actor_state_encoder = StateEncoder(envs, args).to(args.device)
        actor_params += list(actor_state_encoder.parameters())

    actor_optimizer = optim.Adam(actor_params, lr=args.policy_lr)

    envs.single_observation_space.dtype = np.dtype(np.float32)
    envs.single_action_space.dtype = np.dtype(np.float32)
    if args.observable:
        obs_space = gym.spaces.Dict(
            {
                "state": envs.single_observation_space,
                "clicks": gym.spaces.MultiBinary(n = slate_size),
            }
        )
    else:
        obs_space = envs.single_observation_space
    rb = DictRolloutBuffer(
        args.policy_frequency * args.buffer_size,
        obs_space,
        envs.single_action_space,
        args.device,
        gamma = args.gamma,
    )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    if not args.observable:
        actor_state_encoder.reset()
    envs.single_action_space.seed(args.seed)

    start_ep = False
    count_ep = 0
    start_time = time.time()
    for global_step in range(args.total_timesteps + 1):
        with torch.inference_mode():
            if args.observable:
                obs_d = torch.Tensor(obs)
            else:
                obs_d = actor_state_encoder.step(obs)
            actions = actor.get_action(obs_d).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # Run validation episodes
        if global_step % args.val_interval == 0:
            val_start_time = time.time()
            val_obs, _ = val_envs.reset(seed=args.seed + 1)
            if not args.observable:
                actor_state_encoder.reset()
            ep = 0
            cum_boredom = 0
            val_returns, val_lengths, val_boredom = [], [], []
            val_slates, val_user_pref = [[] for _ in range(args.n_val_episodes)], [[] for _ in range(args.n_val_episodes)]
            ep_rewards = []
            while ep < args.n_val_episodes:
                with torch.inference_mode():
                    if args.observable:
                        val_obs = torch.Tensor(val_obs)
                    else:
                        val_obs = actor_state_encoder.step(val_obs)
                    val_action = actor.get_action(val_obs).cpu().numpy()
                (
                    val_next_obs,
                    val_rewards,
                    _,
                    _,
                    val_infos,
                ) = val_envs.step(val_action)
                val_slates[ep].append(val_action)
                val_user_pref[ep].append(val_envs.envs[0].unwrapped.user_embedd)
                ep_rewards.append(val_rewards[0])
                val_obs = val_next_obs
                if "final_info" in val_infos:
                    if not args.observable:
                        actor_state_encoder.reset()
                    for info in val_infos["final_info"]:
                        # Skip the envs that are not done
                        if info is None:
                            continue
                        val_returns.append(info["episode"]["r"])
                        val_lengths.append(info["episode"]["l"])
                        val_boredom.append(cum_boredom)
                        cum_boredom = 0
                        ep += 1
                        ep_rewards = []
                else:
                    cum_boredom += (1.0 if np.sum(val_infos["bored"][0] == True) > 0 else 0.0)

            print(
                f"Step {global_step}: return={np.mean(val_returns):.2f} (+- {np.std(val_returns):.2f}), boredom={np.mean(val_boredom):.2f}"
            )
            if args.track == "wandb":
                val_user_pref = np.array(val_user_pref)
                val_slates = np.array(val_slates)
                val_categories = val_envs.envs[0].unwrapped.item_comp[val_slates]
                average_div = np.mean([[len(np.unique(cat)) for cat in cat_ep] for cat_ep in val_categories])
                user_drift = np.linalg.norm(val_user_pref[:, 0] - val_user_pref[:, -2], axis = -1).mean()
                avg_final_user = val_user_pref[:, -2].mean(axis = 0)
                final_user_dispersion = np.linalg.norm(val_user_pref[:, -2] - avg_final_user, axis = -1).mean()
                slates_table = wandb.Table(columns=[i for i in range(1, len(val_slates[0,0]) + 1) ], data=val_slates[0])
                categories_table = wandb.Table(columns=[i for i in range(1, len(val_categories[0,0]) + 1)], data=val_categories[0])
                wandb.log(
                    {
                        "val_charts/episodic_return": np.mean(val_returns),
                        "val_charts/episodic_length": np.mean(val_lengths),
                        "val_charts/SPS": int(np.sum(val_lengths) / (time.time() - val_start_time)),
                        "val_charts/boredom": np.mean(val_boredom),
                        "misc/diversity": average_div,
                        "misc/user_drift": user_drift,
                        "misc/slates": slates_table,
                        "misc/categories": categories_table,
                        "misc/final_user_dispersion": final_user_dispersion,
                    },
                    global_step,
                )
            elif args.track == "tensorboard":
                writer.add_scalar(
                "val_charts/episodic_return", np.mean(val_returns), global_step
                )
                writer.add_scalar(
                    "val_charts/episodic_length", np.mean(val_lengths), global_step
                )
                writer.add_scalar(
                    "val_charts/SPS", int(np.sum(val_lengths) / (time.time() - val_start_time)), global_step
                )
                writer.add_scalar(
                    "val_charts/boredom", np.mean(val_boredom), global_step
                )
            csv_writer.writerow(["val_charts/episodic_return", np.mean(val_returns), global_step])
            csv_writer.writerow(["val_charts/episodic_length", np.mean(val_lengths), global_step])
            csv_writer.writerow(["val_charts/SPS", int(np.sum(val_lengths) / (time.time() - val_start_time)), global_step])
            csv_writer.writerow(["val_charts/boredom", np.mean(val_boredom), global_step])
            csv_file.flush()

        if "final_info" in infos:
            if not args.observable:
                actor_state_encoder.reset()
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                if args.track == "wandb":
                    wandb.log(
                        {
                            "train_charts/episodic_return": info["episode"]["r"],
                            "train_charts/episodic_length": info["episode"]["l"],
                        },
                        global_step,
                    )
                elif args.track == "tensorboard":
                    writer.add_scalar(
                        "train_charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "train_charts/episodic_length", info["episode"]["l"], global_step
                    )
                csv_writer.writerow(["train_charts/episodic_return", np.mean(info["episode"]["r"]), global_step])
                csv_writer.writerow(["train_charts/episodic_length", np.mean(info["episode"]["l"]), global_step])

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, d in enumerate(infos["_final_observation"]):
                if d:
                    real_next_obs[idx] = infos["final_observation"][idx]

        if args.observable:
            obs = {"state": obs, "clicks": infos["clicks"][0]}
        rb.add(obs, actions, rewards, start_ep, torch.zeros(rewards.shape), torch.zeros(rewards.shape))
        start_ep = np.logical_or(terminated, truncated)
        count_ep += 1

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if start_ep and global_step % args.policy_frequency == 0:
            data_it = rb.get()
            for _ in range(
                args.policy_frequency
            ):  # compensate  delay by doing 'actor_update_interval' instead of 1 
                data = next(data_it)

                # The following assumes at most one click per slate ! If there are more than 1 click per slate, we only consider the first one.
                clicked_slates = torch.any(data.observations["clicks"], dim = 1)
                first_click = torch.argmax(data.observations["clicks"][clicked_slates], dim = 1)
                clicked_items = data.actions[clicked_slates, first_click].long()

                if args.gamma > 0.05:   ## This is nicer but fails with low gammas
                    gamma_pow = torch.tensor(args.gamma).pow(torch.arange(len(data.rewards)))
                    returns_to_go = torch.flip(torch.cumsum(torch.flip(gamma_pow * data.rewards, [0]), dim = 0), [0]) / gamma_pow
                else:
                    returns_to_go = torch.zeros(len(data.rewards))
                    returns_to_go[-1] = data.rewards[-1]
                    for i in range(len(data.rewards)-2, -1, -1):
                        returns_to_go[i] = data.rewards[i] + args.gamma * returns_to_go[i+1]
                returns_to_go = returns_to_go[clicked_slates]

                if args.observable:
                    pi_clicked = actor.get_probs(data.observations["state"], clicked_items.unsqueeze(1))
                else:
                    pi_clicked = actor.get_probs(actor_state_encoder(data.observations), clicked_items.unsqueeze(1))
                log_alphas = torch.log(1 - (1 - pi_clicked).pow(slate_size) + 1e-6).squeeze()
                actor_loss = - torch.sum(returns_to_go * log_alphas)

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

            if args.track == "wandb":
                metric_dict = {
                    "losses/actor_loss": actor_loss.item(),
                    "train_charts/SPS": int(global_step / (time.time() - start_time)),
                }
                wandb.log(metric_dict, global_step)
            elif args.track == "tensorboard":
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("train_charts/SPS",
                    int(global_step / (time.time() - start_time)),
                )
            csv_writer.writerow(["losses/actor_loss", actor_loss.item(), global_step])
            csv_writer.writerow(["train_charts/SPS", int(global_step / (time.time() - start_time)), global_step])
            rb.reset()

    envs.close()
    if args.track == "tensorboard":
        writer.close()
    csv_file.close()


if __name__ == "__main__":
    args = get_parser([get_generic_parser()]).parse_args()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if args.track == "wandb":
        import wandb
        run_name = f"{args.env_id}__{args.run_name}__{args.seed}__{int(time.time())}"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name,
            monitor_gym=False,
            save_code=True,
        )

    train(args)
