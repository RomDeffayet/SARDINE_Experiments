import argparse
from distutils.util import strtobool
from pathlib import Path
import time
from typing import Dict, List, Union, NamedTuple

import torch
import gymnasium as gym
import numpy as np
import scipy.stats as st
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from .naive import NaiveClickModel
from .pbm import PBM
from utils.parser import get_generic_parser
from utils.file import hash_config
from sardine.wrappers import IdealState

def get_parser(parents = []):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument(
        "--click-model",
        type=str,
        required=True,
        choices=["naive", "pbm"],
        help="Click model to be trained",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=30,
        help="Dimension of state",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=10,
        help="Number of items",
    )
    parser.add_argument(
        "--slate-size",
        type=int,
        default=10,
        help="Slate size",
    )
    parser.add_argument(
        "--item-embedd-size",
        type=int,
        default=32,
        help="Size of item embeddings for relevance computation",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Hidden size of neural networks",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Type of item embeddings",
    )
    parser.add_argument(
        "--cm-batch-size",
        type=int,
        default=128,
        help="Batch size for GeMS pretraining.",
    )
    parser.add_argument(
        "--cm-lr",
        type=int,
        default=3e-4,
        help="Batch size for GeMS pretraining.",
    )
    parser.add_argument(
        "--n-test-episodes",
        type=int,
        default=100,
        help="Number of episodes for online evaluation of click models.",
    )
    return parser

TensorDict = Dict[Union[str, int], torch.Tensor]

class ReplayBatch(NamedTuple):
    observations: Union[TensorDict, torch.FloatTensor]
    actions: torch.Tensor
    next_observations: Union[TensorDict, torch.FloatTensor]
    dones: torch.Tensor
    rewards: torch.Tensor

class ReplayBufferDataSet(torch.utils.data.Dataset):
    """
        Pytorch Dataset from SB3's ReplayBufferSamples. 
    """
    def __init__(self, data):
        self.data = data
        self.dict_obs = isinstance(data.observations, dict)

    def __len__(self):
        return len(self.data.rewards)

    def __getitem__(self, idx):
        if self.dict_obs:
            return ReplayBatch(
                    observations = {key: val[idx] for key, val in self.data.observations.items()}, 
                    actions = self.data.actions[idx],
                    next_observations = {key: val[idx] for key, val in self.data.next_observations.items()}, 
                    dones = self.data.dones[idx],
                    rewards = self.data.rewards[idx],
                )
        else:
            return ReplayBatch(
                    observations = self.data.observations[idx], 
                    actions = self.data.actions[idx],
                    next_observations = self.data.next_observations[idx], 
                    dones = self.data.dones[idx],
                    rewards = self.data.rewards[idx],
                )

class ReplayBufferDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, dataset: str, cm_batch_size: int, device: torch.device, **kwargs):
        super().__init__()

        self.data_dir = data_dir
        self.dataset = dataset

        self.device = device
        self.batch_size = cm_batch_size
        self.train_to_val_ratio = 4

    def setup(self, stage: str):
        dataset = torch.load(self.data_dir + "datasets/" + self.dataset, map_location = self.device)
        dataset_len = dataset.size()
        perm = np.random.permutation(dataset_len)
        train_inds = perm[dataset_len // (self.train_to_val_ratio + 1):]
        val_inds = perm[:dataset_len // (self.train_to_val_ratio + 1)]
        self.train = ReplayBufferDataSet(dataset._get_samples(train_inds))
        self.val = ReplayBufferDataSet(dataset._get_samples(val_inds))
        self.dict_obs = self.train.dict_obs
    
    def collate_fn(self, batch : List[ReplayBatch]) -> ReplayBatch:
        if self.dict_obs:
            observations = {key: torch.stack([b.observations[key] for b in batch]) for key in batch[0].observations.keys()}
            next_observations = {key: torch.stack([b.next_observations[key] for b in batch]) for key in batch[0].next_observations.keys()}
        else:
            observations, next_observations = torch.stack([b.observations for b in batch]), torch.stack([b.next_observations for b in batch])
        actions, rewards = torch.stack([b.actions for b in batch]), torch.stack([b.rewards for b in batch])
        dones = torch.stack([b.dones for b in batch])
        return ReplayBatch(observations, actions, next_observations, dones, rewards)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)

def train(args, config_hash: str):

    # Model
    if args.click_model == "naive":
        cm = NaiveClickModel(**vars(args))
    elif args.click_model == "pbm":
        cm = PBM(**vars(args))
    else:
        raise NotImplementedError
    
    # Loggers
    loggers = []
    if args.track == "wandb":
        wandb_logger = WandbLogger(project=args.wandb_project_name, 
                                    save_dir = args.data_dir,
                                    prefix = "click_models",
                                    entity=args.wandb_entity,
                                    config=vars(args),
                                    name=f"{args.exp_name}_{args.run_name}_seed{args.seed}_{int(time.time())}",)
        loggers.append(wandb_logger)
    csv_logger = CSVLogger(args.data_dir + "click_models/results/", 
                            name=config_hash,
                            flush_logs_every_n_steps=1000,
                        )
    csv_logger.log_hyperparams(vars(args))
    loggers.append(csv_logger)

    # Trainer
    if args.exp_name == 'test':
        trainer = pl.Trainer(logger=loggers, enable_progress_bar = False, devices = 1,
                                accelerator = "gpu" if args.device == "cuda" else "cpu", 
                                max_epochs = args.max_epochs, num_sanity_val_steps=0)
    else :
        ckpt_dir =  args.data_dir + "click_models/checkpoints/" + args.exp_name + "/"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        model_checkpoint = ModelCheckpoint(monitor = 'val/loss', dirpath = ckpt_dir, filename = config_hash)

        trainer = pl.Trainer(logger=loggers, enable_progress_bar = False, devices = 1,
                                accelerator = "gpu" if args.device == "cuda" else "cpu", 
                                callbacks = [model_checkpoint], max_epochs = args.max_epochs)

    ## Load data and intialize data module
    datamod = ReplayBufferDataModule(**vars(args))

    ## Train the model
    trainer.fit(cm, datamod)
    return cm

def evaluate(args, config_hash: str):
    ckpt_dir =  args.data_dir + "click_models/checkpoints/" + args.exp_name + "/"
    if args.click_model == "naive":
        click_model = NaiveClickModel.load_from_checkpoint(ckpt_dir + config_hash + ".ckpt", **vars(args))
    elif args.click_model == "pbm":
        click_model = PBM.load_from_checkpoint(ckpt_dir + config_hash + ".ckpt", **vars(args))
        print(click_model.propensities)
        print(torch.nn.Sigmoid()(click_model.propensities.weight))
    else:
        raise NotImplementedError


    env_id = args.dataset.split('_')[0]
    print(env_id)
    env = IdealState(gym.make(env_id))

    obs, _ = env.reset(seed = args.seed)
    ep, cum_reward = 0, 0
    returns = []
    while ep < args.n_test_episodes:
        state = torch.tensor(obs)
        items = torch.arange(args.num_items)

        with torch.inference_mode():
            click_probs, relevance_probs = click_model(state, items)
        slate = np.flip(np.argsort(relevance_probs.cpu().numpy()))

        obs, reward, terminated, truncated, _ = env.step(slate)
        cum_reward += reward

        #obs = next_obs

        if terminated:
            obs, _ = env.reset()
            returns.append(cum_reward)
            cum_reward = 0
            ep += 1

    return returns

if __name__ == "__main__":
    args = get_parser([get_generic_parser()]).parse_args()

    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    config_hash = hash_config(args)
    train(args, config_hash)
    returns = evaluate(args, config_hash)
    avg_return = np.mean(returns)
    interval = st.t.interval(confidence=0.95, df=len(returns)-1, loc=avg_return, scale=st.sem(returns))
    print(f"Average online return: {avg_return:.2f} (+- {avg_return - interval[0]:.2f})")