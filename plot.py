import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import argparse
import yaml
from yaml.loader import FullLoader
from train import get_parser
from utils.parser import get_generic_parser
from utils.file import hash_config

def yml2str(config):
    # Convert the YAML file into a dictionary
    config_file = open("config/" + config + ".yml", "r")
    config_dict = yaml.load(config_file, Loader=FullLoader)
    config_file.close()
    if type(config_dict["env-id"]) == list:
        assert len(config_dict["env-id"]) == 1 # Assuming the list contains a single env
        config_dict["env-id"] = config_dict["env-id"][0]
    if type(config_dict["seed"]) == list:
        config_dict["seed"] = config_dict["seed"][0] # Seed doesn't matter for config hash

    # Convert the YML dictionary into a list of arguments
    arg_list = []
    for (k, v) in config_dict.items():
        arg_list.append("--" + str(k) + "=" + str(v))

    return arg_list

parser = argparse.ArgumentParser(parents = [], add_help = False)
parser.add_argument('--config', nargs='+', help='List of configs to compare', required=True)
parser.add_argument('--indicator', nargs='+', help='List of indicators to report on', default=['val_charts/episodic_return' ,'val_charts/boredom'])
args = parser.parse_args()

fig_path = "figs/"
os.makedirs(os.path.dirname(fig_path), exist_ok=True)

# Fetch configs based on hash
index_path = "logs/index.csv"
index_df = pd.read_csv(index_path, sep="\t", header=0, names=["hash", "args"])
list_configs = args.config
list_hashes = []
list_envs = []
hash2config = {}
for config in list_configs:
    arg_list = yml2str(config)
    config_parser = get_generic_parser()
    config_parser = get_parser(parents=[config_parser], args=arg_list)
    config_args, _ = config_parser.parse_known_args(arg_list)
    hash = hash_config(config_args, index=False)
    config_split = config.split("-")
    list_envs.append("-".join(config_split[0:2]))
    hash2config[hash] = " ".join(config_split[2:])
    list_hashes.append(hash)

if len(list_hashes) == 0:
    print("No recorded config found to match the input config(s) in logs/index.csv")
    exit()

if len(set(list_envs)) != 1:
    print("Error: comparing runs obtained for different environments")
    exit()
else:
    env = str(list_envs[0])

if args.indicator is not None:
    list_indicators = args.indicator

#list_hashes = ["29a0b3f520ac365007da0966eeba26884ed4b430154c2f59b6d5d7aa41796f6b", "6abae326bcd877fdf521747efd720abb9bcd49d95272e9cad186ec2cfcbb0b14"]
#list_indicators = ["val_charts/episodic_return", "val_charts/boredom"]
#colors = sns.color_palette("husl", len(list_hashes))

# Make a data frame from the results of the relevant runs (determined by the specified configs)
pd.options.display.width = 0
log_dir = "logs/"
log_df = pd.DataFrame({'field': [], 'value': [], 'step': [], 'id': [], 'seed': []})
for log_file in os.listdir(log_dir):
    log_file_split = log_file.split(".")
    if log_file_split[-1] != "log":
        continue
    log_file_split = ".".join(log_file_split[:-1]).split("-")
    file_env = "-".join(log_file_split[:2])[4:] # Remove prefix "run_" from the env name
    file_hash = log_file_split[-2]
    file_seed = log_file_split[-1]
    if file_env == env:
        if file_hash in list_hashes:
            df = pd.read_csv(log_dir + "/" + log_file)
            df = df[df["field"].isin(list_indicators)]
            df["id"] = hash2config[file_hash] # Get the config corresponding to the hash
            df["seed"] = file_seed
            log_df = pd.concat([log_df, df])
        elif log_file_split[2] in ["greedyoracle", "random"]:
            df = pd.read_csv(log_dir + "/" + log_file)
            df = df[df["field"].isin(list_indicators)]
            df["id"] = "Random" if log_file_split[2] == "random" else "Greedy Oracle"
            df["seed"] = file_seed
            log_df = pd.concat([log_df, df])

# Define the order among approaches, and move Random and Greedy Oracle first
hue_order = list(log_df["id"].unique())
if "Greedy Oracle" in hue_order:
    hue_order.remove("Greedy Oracle")
    hue_order = ["Greedy Oracle"] + hue_order
if "Random" in hue_order:
    hue_order.remove("Random")
    hue_order = ["Random"] + hue_order

# Make the plot for each field
for field in log_df["field"].unique():
    df = log_df[log_df["field"] == field]
    field = field.replace("charts/", "")
    field = field.replace("losses/", "")
    plt.figure(figsize=(10,8))
    sns.lineplot(
        x="step", y="value", hue="id",
        #color=colors,
        data=df,
        alpha=0.7,
        hue_order=hue_order
    )
    plt.xlabel("number of steps", fontsize=20, labelpad=12)
    plt.ylabel(field.replace("_", " "), fontsize=20, labelpad=12)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right', fontsize=18, framealpha=0.6)
    plt.savefig("figs/" + env + "_" + field + ".pdf")
    #plt.show()
