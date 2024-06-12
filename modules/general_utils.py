import json
import os
from pathlib import Path

import torch

# --- GENERAL UTILS ---


def find_device(training: bool = False):
    """
    Description: Find the device to use for the pipeline. If cuda is available, use it. If not, check if MPS is available and use it. If not, use CPU.

    Input: training (bool) : Whether the pipeline is being used for training or not.

    Returns: device (str) : The device to use for the pipeline.
    """
    print("[INFO] Finding device.")
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        if training == False:
            # loading metadata on mps for inference is quite slow. So disabling for now.
            return "cpu"
        return "mps"
    else:
        return "cpu"


def load_config_and_device(config_file: str, training: bool = False):
    """
    Description: Load the config file and find the device to use for the pipeline.

    Input: config_file (str) : The path to the config file.
    training (bool) : Whether the pipeline is being used for training or not.

    Returns: config (dict) : The config dictionary + device (str) : The device to use for the pipeline.
    """
    # Check if the config file exists and load it
    if not os.path.exists(config_file):
        raise Exception("Config file does not exist.")
    with open(config_file, "r") as f:
        config = json.load(f)

    # Find device and set it in the config between cpu and cuda and mps if available
    config["device"] = find_device(training)
    print(f"[INFO] Device found: {config['device']}")
    return config
