from __future__ import annotations
from typing import TypedDict


class ModelParams6and30(TypedDict, total=False):
    filters: int
    kernel_size: int
    pool_type: str
    regularizer: str
    activation_type: str
    epochs: int
    batch_size: int
    loss_func: str
    optimizer: str


class ModelParams35(TypedDict, total=False):
    filters_0: int
    filters_1: int
    kernel_size_0: int
    kernel_size_1: int
    regularizer_2: str
    activation_type: str
    epochs: int
    batch_size: int
    loss_func: str
    optimizer: str
    alpha: float


class ParamsReader:
    def __init__(self, model_no: int) -> None:
        self._model_no = model_no

    # TODO: Use .ini for parameters
    def get_parameters(self, file_name: str) -> ModelParams6and30 | ModelParams35:
        if self._model_no == 6 or self._model_no == 30:
            dict = ModelParams6and30()
        elif self._model_no == 35:
            dict = ModelParams35()

        with open(file_name) as f:
            for line in f:
                (key, val) = line.split()
                dict[key] = val

        # change string values to integer values
        if self._model_no == 6 or self._model_no == 30:
            dict["filters"] = int(dict["filters"])
            dict["kernel_size"] = int(dict["kernel_size"])
        elif self._model_no == 35:
            dict["filters_0"] = int(dict["filters_0"])
            dict["filters_1"] = int(dict["filters_1"])
            dict["kernel_size_0"] = int(dict["kernel_size_0"])
            dict["kernel_size_1"] = int(dict["kernel_size_1"])
            dict["alpha"] = float(dict["alpha"])

        dict["epochs"] = int(dict["epochs"])
        dict["batch_size"] = int(dict["batch_size"])

        return dict
