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

from .buffer import ReplayBuffer, POMDPDictReplayBuffer
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
        default="SlateTopK-Bored-v0",
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
        default=500000,
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
        default=int(1e6),
        help="the replay memory buffer size",
    )
    parser.add_argument(
        "--sampled-seq-len",
        type=int,
        default=10,
        help="Number of timesteps to be sampled from replay buffer for each trajectory (only for POMDP)",
    )
    parser.add_argument(
        "--learning-starts", type=int, default=1e4, help="timestep to start learning"
    )
    parser.add_argument(
        "--ranker",
        type=str,
        default="topk",
        choices=["topk", "gems"],
        help="Type of ranker for slate generation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="the batch size of sample from the reply memory",
    )
    # SAC arguments
    parser.add_argument(
        "--gamma", type=float, default=0.8, help="the discount factor gamma"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="target smoothing coefficient (default: 0.005)",
    )
    parser.add_argument(
        "--exploration-noise",
        type=float,
        default=0.1,
        help="the scale of exploration noise",
    )
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=3e-4,
        help="the learning rate of the policy network optimizer",
    )
    parser.add_argument(
        "--q-lr",
        type=float,
        default=1e-3,
        help="the learning rate of the Q network network optimizer",
    )
    parser.add_argument(
        "--policy-frequency",
        type=int,
        default=2,
        help="the frequency of training policy (delayed)",
    )
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=1,
        help="the frequency of updates for the target networks",
    )
    parser.add_argument(
        "--noise-clip",
        type=float,
        default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Entropy regularization coefficient."
    )
    parser.add_argument(
        "--autotune",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="automatic tuning of the entropy coefficient",
    )
    parser.add_argument(
        "--n-updates", type=int, default=1, help="Number of Q updates per sample."
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
        "--singleq",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Remove second Q-Network",
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
    ranker,
    args,
    decoder,
):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if ranker == "topk":
            # if args.item_embeddings == "ideal":
            env = TopK(env, "ideal", min_action = 0, max_action = 1)
            #elif args.item_embeddings == "custom":
            # path = args.data_dir + "datasets/mf_embeddings/" + args.env_id + "_epsilon0.5_seed2023.npy"
            # env = TopK(env, path, min_action = 0, max_action = 1)
        elif ranker == "gems":
            env = GeMS(env,
                       path = args.data_dir + "GeMS/decoder/" + args.exp_name + "/" + args.run_name + ".pt",
                       device = args.device,
                       decoder = decoder,
                    )
        if observable:
            env = IdealState(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_size, state_dim):
        super().__init__()

        if state_dim is None:
            state_dim = np.array(env.single_observation_space.shape).prod()

        self.model = nn.Sequential(
            nn.Linear(
                state_dim + np.prod(env.single_action_space.shape),
                hidden_size
            ),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, torch.sqrt(torch.tensor(2)))
            m.bias.data.fill_(0.0)

    def forward(self, x, a):
        x = torch.cat([x, a], dim = -1)
        return self.model(x)

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env, hidden_size, state_dim):
        super().__init__()

        if state_dim is None:
            state_dim = np.array(env.single_observation_space.shape).prod()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(hidden_size, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.FloatTensor((env.action_space.high - env.action_space.low) / 2.0),
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor((env.action_space.high + env.action_space.low) / 2.0),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, torch.sqrt(torch.tensor(2)))
            m.bias.data.fill_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, return_prob = False):
        mean, log_std = self(x)
        std = log_std.exp()
        eps = torch.randn_like(std)
        x_t = mean + eps * std  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        if return_prob:
            log_prob = - log_std - .5 * torch.log(2 * torch.pi * torch.ones_like(log_std)) - .5 * eps.pow(2)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob
        else:
            return action

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
                args.ranker,
                args,
                decoder,
            )
        ]
    )
    val_envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                0,
                args.observable,
                args.ranker,
                args,
                decoder,
            )
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    actor = Actor(envs, args.hidden_size, args.state_dim).to(args.device)
    qf1 = SoftQNetwork(envs, args.hidden_size, args.state_dim).to(args.device)
    qf1_target = SoftQNetwork(envs, args.hidden_size, args.state_dim).to(args.device)
    qf1_target.load_state_dict(qf1.state_dict())
    actor_params = list(actor.parameters())
    critic_params = list(qf1.parameters())
    if not args.observable:
        if args.state_encoder == "gru":
            StateEncoder = GRUStateEncoder
        elif args.state_encoder == "transformer":
            StateEncoder = TransformerStateEncoder
        else:
            StateEncoder = None
        actor_state_encoder = StateEncoder(envs, args).to(args.device)
        qf1_state_encoder = actor_state_encoder if args.shared_se else StateEncoder(envs, args).to(args.device)
        qf1_state_encoder_target = StateEncoder(envs, args).to(args.device)
        qf1_state_encoder_target.load_state_dict(qf1_state_encoder.state_dict())
        actor_params += list(actor_state_encoder.parameters())
        critic_params += list(qf1_state_encoder.parameters())

    if not args.singleq:
        qf2 = SoftQNetwork(envs, args.hidden_size, args.state_dim).to(args.device)
        qf2_target = SoftQNetwork(envs, args.hidden_size, args.state_dim).to(args.device)
        qf2_target.load_state_dict(qf2.state_dict())
        critic_params += list(qf2.parameters())
        if not args.observable:
            qf2_state_encoder = actor_state_encoder if args.shared_se else StateEncoder(envs, args).to(args.device)
            qf2_state_encoder_target = StateEncoder(envs, args).to(args.device)
            qf2_state_encoder_target.load_state_dict(qf2_state_encoder.state_dict())
            critic_params += list(qf2_state_encoder.parameters())

    q_optimizer = optim.Adam(critic_params, lr=args.q_lr)
    actor_optimizer = optim.Adam(actor_params, lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.dtype(np.float32)
    envs.single_action_space.dtype = np.dtype(np.float32)
    if args.observable:
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            args.device,
            handle_timeout_termination=True,
        )
    else:
        rb = POMDPDictReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            args.sampled_seq_len,
            args.device,
            handle_timeout_termination=True,
        )

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    if not args.observable:
        actor_state_encoder.reset()
    envs.single_action_space.seed(args.seed)

    for global_step in range(args.total_timesteps + 1):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
            if args.ranker == "gems":
                actions = torch.tensor(actions)
        else:
            with torch.inference_mode():
                if args.observable:
                    obs_d = torch.Tensor(obs)
                else:
                    obs_d = actor_state_encoder.step(obs)
                actions = actor.get_action(obs_d)
                if args.ranker != "gems":
                    actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        if args.ranker == "gems":
            actions = actions.cpu().numpy()

        # Run validation episodes
        if global_step % args.val_interval == 0:
            val_start_time = time.time()
            val_obs, _ = val_envs.reset(seed=args.seed + 1)
            if not args.observable:
                actor_state_encoder.reset()
            ep = 0
            cum_boredom = 0
            val_returns, val_lengths, val_boredom = [], [], []
            val_errors, val_errors_norm = [], []
            val_slates, val_user_pref = [[] for _ in range(args.n_val_episodes)], [[] for _ in range(args.n_val_episodes)]
            val_protoactions = [[] for _ in range(args.n_val_episodes)]
            ep_rewards, pred_q_values = [], []
            while ep < args.n_val_episodes:
                with torch.inference_mode():
                    if args.observable:
                        val_obs = torch.Tensor(val_obs)
                    else:
                        val_obs = actor_state_encoder.step(val_obs)
                    val_action = actor.get_action(val_obs)
                    pred_q_values.append(qf1(val_obs, val_action).item())
                if args.ranker == "gems":
                    (
                        val_next_obs,
                        val_rewards,
                        _,
                        _,
                        val_infos,
                    ) = val_envs.step(val_action)
                else:
                    (
                        val_next_obs,
                        val_rewards,
                        _,
                        _,
                        val_infos,
                    ) = val_envs.step(val_action.cpu().numpy())
                val_slates[ep].append(val_envs.envs[0].latest_slate)
                val_user_pref[ep].append(val_envs.envs[0].unwrapped.user_embedd)
                val_protoactions[ep].append(val_action.squeeze().cpu().numpy())
                ep_rewards.append(val_rewards[0])
                val_obs = val_next_obs
                if "final_info" in val_infos:
                    if not args.observable:
                        actor_state_encoder.reset()
                    max_t = len(ep_rewards) - 20
                    if max_t > 0:
                        gamma_pow = np.power(args.gamma, np.arange(max_t))
                        discounted_returns = np.array(
                            [
                                (gamma_pow[:-i] * ep_rewards[i:max_t]).sum()
                                if i > 0
                            else (gamma_pow * ep_rewards[:max_t]).sum()
                                        for i in range(len(gamma_pow))
                            ]
                        )
                        val_errors.extend(pred_q_values[:max_t] - discounted_returns)
                        val_errors_norm.extend(np.abs(discounted_returns - pred_q_values[:max_t]) / discounted_returns.clip(min=0.01))
                    for info in val_infos["final_info"]:
                        # Skip the envs that are not done
                        if info is None:
                            continue
                        val_returns.append(info["episode"]["r"])
                        val_lengths.append(info["episode"]["l"])
                        val_boredom.append(cum_boredom)
                        cum_boredom = 0
                        ep += 1
                        ep_rewards, pred_q_values = [], []
                else:
                    cum_boredom += (1.0 if np.sum(val_infos["bored"][0] == True) > 0 else 0.0)

            print(
                f"Step {global_step}: return={np.mean(val_returns):.2f} (+- {np.std(val_returns):.2f}), boredom={np.mean(val_boredom):.2f}"
            )
            if args.track == "wandb":
                val_user_pref = np.array(val_user_pref)
                val_slates = np.array(val_slates)
                val_protoactions = np.array(val_protoactions)
                val_categories = val_envs.envs[0].unwrapped.item_comp[val_slates]
                average_div = np.mean([[len(np.unique(cat)) for cat in cat_ep] for cat_ep in val_categories])
                user_drift = np.linalg.norm(val_user_pref[:, 0] - val_user_pref[:, -2], axis = -1).mean()
                avg_final_user = val_user_pref[:, -2].mean(axis = 0)
                final_user_dispersion = np.linalg.norm(val_user_pref[:, -2] - avg_final_user, axis = -1).mean()
                slates_table = wandb.Table(columns=[i for i in range(1, len(val_slates[0,0]) + 1) ], data=val_slates[0])
                categories_table = wandb.Table(columns=[i for i in range(1, len(val_categories[0,0]) + 1)], data=val_categories[0])
                actions_table = wandb.Table(columns=[i for i in range(0, len(val_protoactions[0,0]))], data=val_protoactions[0])
                wandb.log(
                    {
                        "val_charts/episodic_return": np.mean(val_returns),
                        "val_charts/episodic_length": np.mean(val_lengths),
                        "val_charts/q_error": np.mean(val_errors),
                        "val_charts/q_error_norm": np.mean(val_errors_norm),
                        "val_charts/SPS": int(np.sum(val_lengths) / (time.time() - val_start_time)),
                        "val_charts/boredom": np.mean(val_boredom),
                        "misc/diversity": average_div,
                        "misc/user_drift": user_drift,
                        "misc/slates": slates_table,
                        "misc/categories": categories_table,
                        "misc/final_user_dispersion": final_user_dispersion,
                        "misc/protoactions": actions_table,
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
                    "val_charts/q_error", np.mean(val_errors), global_step
                )
                writer.add_scalar(
                    "val_charts/q_error_norm", np.mean(val_errors_norm), global_step
                )
                writer.add_scalar(
                    "val_charts/SPS", int(np.sum(val_lengths) / (time.time() - val_start_time)), global_step
                )
                writer.add_scalar(
                    "val_charts/boredom", np.mean(val_boredom), global_step
                )
            csv_writer.writerow(["val_charts/episodic_return", np.mean(val_returns), global_step])
            csv_writer.writerow(["val_charts/episodic_length", np.mean(val_lengths), global_step])
            csv_writer.writerow(["val_charts/q_error", np.mean(val_errors), global_step])
            csv_writer.writerow(["val_charts/q_error_norm", np.mean(val_errors_norm), global_step])
            csv_writer.writerow(["val_charts/SPS", int(np.sum(val_lengths) / (time.time() - val_start_time)), global_step])
            csv_writer.writerow(["val_charts/boredom", np.mean(val_boredom), global_step])
            csv_file.flush()

        done = np.logical_or(terminated, truncated)
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
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step == args.learning_starts:
            start_time = time.time()
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            for _ in range(args.n_updates):
                with torch.no_grad():
                    if args.observable:
                        actor_next_observations = qf1_next_observations = data.next_observations
                    else:
                        actor_next_observations = actor_state_encoder(data.next_observations)
                        qf1_next_observations = qf1_state_encoder_target(data.next_observations)
                    next_state_actions, next_state_log_pi = actor.get_action(
                        actor_next_observations, return_prob = True
                    )
                    if args.singleq:
                        qf_next_target = qf1_target(
                            qf1_next_observations,
                            next_state_actions,
                        )
                        next_q_value = data.rewards.flatten() + (
                            1 - data.dones.flatten()
                        ) * args.gamma * qf_next_target.view(-1)
                    else:
                        if args.observable:
                            qf2_next_observations = data.next_observations
                        else:
                            qf2_next_observations = qf2_state_encoder_target(data.next_observations)
                        qf1_next_target = qf1_target(
                            qf1_next_observations,
                            next_state_actions,
                        )
                        qf2_next_target = qf2_target(
                            qf2_next_observations,
                            next_state_actions,
                        )
                        min_qf_next_target = (
                            torch.min(qf1_next_target, qf2_next_target)
                            - alpha * next_state_log_pi
                        )
                        next_q_value = data.rewards.flatten() + (
                            1 - data.dones.flatten()
                        ) * args.gamma * min_qf_next_target.view(-1)
                if args.observable:
                    observations_qf1 = observations_qf2 = data.observations
                else:
                    observations_qf1 = qf1_state_encoder(data.observations)
                qf1_a_values = qf1(observations_qf1, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf_loss = qf1_loss
                if not args.singleq:
                    if not args.observable:
                        observations_qf2 = qf2_state_encoder(data.observations)
                    qf2_a_values = qf2(observations_qf2, data.actions).view(-1)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss += qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate  delay by doing 'actor_update_interval' instead of 1
                    if args.observable:
                        observations_actor = observations_qf1 = observations_qf2 = data.observations
                    else:
                        observations_actor = actor_state_encoder(data.observations)
                        observations_qf1 = qf1_state_encoder(data.observations)
                    pi, log_pi = actor.get_action(observations_actor, return_prob = True)
                    if args.singleq:
                        qf_pi = qf1(observations_qf1, pi)
                        actor_loss = ((alpha * log_pi.clip(max=100)) - qf_pi).mean()
                    else:
                        qf1_pi = qf1(observations_qf1, pi)
                        if not args.observable:
                            observations_qf2 = qf2_state_encoder(data.observations)
                        qf2_pi = qf2(observations_qf2, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                        actor_loss = ((alpha * log_pi.clip(max=100)) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            if args.observable:
                                observations_actor = observations_qf1 = observations_qf2 = data.observations
                            else:
                                observations_actor = actor_state_encoder(data.observations)
                            _, log_pi = actor.get_action(observations_actor, return_prob = True)

                        alpha_loss = (
                            -log_alpha * (log_pi.clip(max=100) + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                if not args.observable:
                    for param, target_param in zip(
                        qf1_state_encoder.parameters(), qf1_state_encoder_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                if not args.singleq:
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    if not args.observable:
                        for param, target_param in zip(
                            qf2_state_encoder.parameters(), qf2_state_encoder_target.parameters()
                        ):
                            target_param.data.copy_(
                                args.tau * param.data + (1 - args.tau) * target_param.data
                            )

            if global_step % 1000 == 0:
                if args.track == "wandb":
                    metric_dict = {
                        "train_charts/qf1_values": qf1_a_values.mean().item(),
                        "losses/qf_loss": qf_loss.item() / 2.0,
                        "losses/actor_loss": actor_loss.item(),
                        "train_charts/alpha": alpha,
                        "train_charts/SPS": int((global_step - args.learning_starts) / (time.time() - start_time)),
                    }
                    if args.autotune:
                        metric_dict["losses/alpha_loss"] = alpha_loss.item()
                    wandb.log(metric_dict, global_step)
                elif args.track == "tensorboard":
                    writer.add_scalar(
                        "train_charts/qf1_values", qf1_a_values.mean().item(), global_step
                    )
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    writer.add_scalar("train_charts/alpha", alpha, global_step)
                    writer.add_scalar("train_charts/SPS",
                        int((global_step - args.learning_starts) / (time.time() - start_time)),
                    )
                    if args.autotune:
                        writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), global_step
                        )
                csv_writer.writerow(["train_charts/qf1_values", qf1_a_values.mean().item(), global_step])
                csv_writer.writerow(["losses/qf_loss", qf_loss.item() / 2.0, global_step])
                csv_writer.writerow(["losses/actor_loss", actor_loss.item(), global_step])
                csv_writer.writerow(["train_charts/alpha", alpha, global_step])
                csv_writer.writerow(["train_charts/SPS", int((global_step - args.learning_starts) / (time.time() - start_time)), global_step])
                if args.autotune:
                    csv_writer.writerow(["losses/alpha_loss", alpha_loss.item(), global_step])

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
