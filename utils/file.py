import hashlib
import pandas as pd
import os

# def hash_config(args):
#     hash_sha256 = hashlib.sha256()
#     hash_sha256.update(str(vars(args)).encode("utf-8"))
#     return hash_sha256.hexdigest()

def hash_config(args, index=False):
    hash_sha256 = hashlib.sha256()
    args_str = args2str(args)
    hash_sha256.update(args_str.encode("utf-8"))
    hash_hex = str(hash_sha256.hexdigest())
    if index: # Add the args_str to the index file if it's not there already
        index_path = "logs/index.csv"
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        open(index_path, "a+", newline="").close() # Create file it doesn't exist
        index_df = pd.read_csv(index_path, sep="\t", header=0, names=["hash", "args"])
        if hash_hex not in index_df['hash'].unique():
            index_df.loc[len(index_df)] = {"hash": hash_hex, "args": args_str}
            index_df.to_csv(index_path, sep="\t", index=False)
    return hash_hex

def args2str(args, exclude=["data_dir", "seed", "info", "exp_name", "run_name", "track", "wandb_project_name", "wandb_entity"]):
    args = vars(args).copy()
    for arg in exclude:
        if arg in args.keys():
            args.pop(arg)
    return str(args)
