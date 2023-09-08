import argparse
from distutils.util import strtobool
from pathlib import Path
import os
import time
from typing import Dict, List, Union, NamedTuple

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from .models import GeMS
from utils.parser import get_generic_parser
from utils.file import hash_config

def get_parser(parents = []):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument(
        "--item-embeddings",
        type=str,
        default="ideal",
        help="Type of item embeddings",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Type of item embeddings",
    )
    parser.add_argument(
        "--gems-batch-size",
        type=int,
        default=128,
        help="Batch size for GeMS pretraining.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="Size of the latent space.",
    )
    parser.add_argument(
        "--gems-lr",
        type=int,
        default=1e-3,
        help="Batch size for GeMS pretraining.",
    )
    parser.add_argument(
        "--lambda-KL",
        type=float,
        default=1.0,
        help="KL loss weight in GeMS.",
    )
    parser.add_argument(
        "--lambda-click",
        type=float,
        default=0.2,
        help="Click loss weight in GeMS.",
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
    def __init__(self, data_dir: str, dataset: str, gems_batch_size: int, device: torch.device, **kwargs):
        super().__init__()

        self.data_dir = data_dir
        self.dataset = dataset

        self.device = device
        self.batch_size = gems_batch_size
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

def train(args, config_hash):
    # Model
    gems = GeMS(**vars(args))
    
    # Loggers
    loggers = []
    if args.track == "wandb":
        wandb_logger = WandbLogger(project=args.wandb_project_name, 
                                    save_dir = args.data_dir,
                                    prefix = "GeMS",
                                    entity=args.wandb_entity,
                                    config=vars(args),
                                    name=f"{args.exp_name}_{args.run_name}_seed{args.seed}_{int(time.time())}",)
        loggers.append(wandb_logger)
    csv_logger = CSVLogger(args.data_dir + "GeMS/results/", 
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
        ckpt_dir =  args.data_dir + "GeMS/checkpoints/" + args.exp_name + "/"
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        model_checkpoint = ModelCheckpoint(monitor = 'val/loss', dirpath = ckpt_dir, filename = config_hash)

        trainer = pl.Trainer(logger=loggers, enable_progress_bar = False, devices = 1,
                                accelerator = "gpu" if args.device == "cuda" else "cpu", 
                                callbacks = [model_checkpoint], max_epochs = args.max_epochs)

    ## Load data and intialize data module
    datamod = ReplayBufferDataModule(**vars(args))

    ## Train the model
    trainer.fit(gems, datamod)

    # Save state_dict of decoder for downstream RL training
    gems = GeMS.load_from_checkpoint(ckpt_dir + config_hash + ".ckpt", **vars(args))
    decoder_dir = args.data_dir + "GeMS/decoder/" + args.exp_name + "/"
    Path(decoder_dir).mkdir(parents=True, exist_ok=True)
    torch.save(gems.decoder, decoder_dir + config_hash +".pt")
    return gems.decoder.to(args.device)

if __name__ == "__main__":
    args = get_parser([get_generic_parser()]).parse_args()

    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    config_hash = hash_config(args)
    train(args, config_hash)