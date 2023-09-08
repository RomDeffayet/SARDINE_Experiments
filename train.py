import argparse
from distutils.util import strtobool
from pathlib import Path
import time

import pytorch_lightning as pl
import torch
import random
import numpy as np
from agents import sac, ddpg, hac, reinforce, topk_reinforce
from gems import gems
from utils.parser import get_generic_parser
from utils.file import hash_config

def get_parser(parents = [], args = None):
    parser = argparse.ArgumentParser(parents = parents, add_help = False)
    parser.add_argument(
        "--agent",
        type=str,
        required = True,
        choices=["sac", "ddpg", "hac", "reinforce", "topk_reinforce"],
        help="Type of agent",
    )
    parser.add_argument(
        "--pretrain",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Whether to pretrain GeMS",
    )

    if args is not None:
        args, _ = parser.parse_known_args(args)
    else:
        args, _ = parser.parse_known_args()

    if args.pretrain:
        parser = gems.get_parser(parents = [parser])
    if args.agent == "sac":
        parser = sac.get_parser(parents = [parser])
    elif args.agent == "ddpg":
        parser = ddpg.get_parser(parents = [parser])
    elif args.agent == "hac":
        parser = hac.get_parser(parents = [parser])
    elif args.agent == "reinforce":
        parser = reinforce.get_parser(parents = [parser])
    elif args.agent == "topk_reinforce":
        parser = topk_reinforce.get_parser(parents = [parser])

    return parser

def main(parents = []):
    parser = get_parser(parents = parents)
    args = parser.parse_args()
    decoder = None

    pl.seed_everything(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cpu":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if args.pretrain:
        print("### Pretraining GeMS ###")
        decoder_dir = args.data_dir + "GeMS/decoder/" + args.exp_name + "/"
        config_hash = hash_config(args)
        if Path(decoder_dir, config_hash + '.pt').is_file() and (args.run_name != 'test'):
            # checkpoint already exists.
            print("Skipping GeMS training") # since it has already been done"
            decoder = torch.load(decoder_dir + config_hash + ".pt").to(device)

            if args.track == "wandb":
                import wandb
                run_name = f"{args.exp_name}_{args.run_name}_seed{args.seed}_{int(time.time())}"
                wandb.init(
                    project=args.wandb_project_name,
                    entity=args.wandb_entity,
                    config=vars(args),
                    name=run_name,
                    monitor_gym=False,
                    save_code=True,
                )
        else:
            decoder = gems.train(args, config_hash)
    elif args.track == "wandb":
        import wandb
        run_name = f"{args.exp_name}_{args.run_name}_seed{args.seed}_{int(time.time())}"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    print("### Training agent ###")
    if args.agent == "sac":
        sac.train(args, decoder = decoder)
    elif args.agent == "ddpg":
        ddpg.train(args, decoder = decoder)
    elif args.agent == "hac":
        hac.train(args)
    elif args.agent == "reinforce":
        reinforce.train(args, decoder = decoder)
    elif args.agent == "topk_reinforce":
        topk_reinforce.train(args, decoder = decoder)

if __name__ == "__main__":
    parser = get_generic_parser()
    main([parser])
