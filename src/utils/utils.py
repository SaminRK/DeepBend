import re
from typing import Dict


def is_float(s: str) -> bool:
    return re.match(r"^-?\d+(?:\.\d+)$", s) is not None


def get_hyperparameters(filename: str) -> Dict:
    params = {}
    with open(filename) as f:
        for line in f:
            (key, val) = line.split()
            params[key] = val

    # change string values to integer values
    for k, v in params.items():
        if v.isdigit():
            params[k] = int(v)
        elif is_float(v):
            params[k] = float(v)

    return params


def get_model_id(model_name: str, hyperparameter_filename: str, train_filename: str, seed: int) -> str:
    hyperparameter_filename_wo_ext = hyperparameter_filename.replace("/", "_").split(".")[0]
    train_filename_wo_ext = train_filename.split("/")[-1].split(".")[0]
    return f"{model_name}_{hyperparameter_filename_wo_ext}_{train_filename_wo_ext}_{seed}"


def get_cross_validation_id(model_name, hyperparameter_filename, data_filename):
    hyperparameter_filename_wo_ext = hyperparameter_filename.replace("/", "_").split(".")[0]
    data_filename_wo_ext = data_filename.split("/")[-1].split(".")[0]
    return f"{model_name}_{hyperparameter_filename_wo_ext}_{data_filename_wo_ext}"
