import argparse
import os
from distutils.util import strtobool


def get_generic_parser(parents = []):
    parser = argparse.ArgumentParser(parents = parents)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/2/user/rdeffaye/SlateRL/",
        help="Path to data directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed.",
    )
    parser.add_argument(
        "--info",
        type=str,
        default="sac",
        help="info on the run",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="test-exp",
        help="the name of this experiment",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="test-run",
        help="the name of this run",
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training.",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        choices = [None ,"wandb", "tensorboard"],
        help="Type of experiment tracking",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="SlateRL",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    return parser