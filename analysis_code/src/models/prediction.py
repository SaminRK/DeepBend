from __future__ import annotations
import inspect
import time
from pathlib import Path
from typing import Union

import tensorflow as tf
import keras
from keras.utils.vis_utils import plot_model
import pandas as pd
import numpy as np
import logomaker as lm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from plotnine import ggplot, aes, xlim, ylim, stat_bin_2d

from .cnnmodel import CNNModel, CNNModel6, CNNModel30, CNNModel35
from .data_preprocess import Preprocess
from .parameters import ParamsReader
from util.custom_types import LIBRARY_NAMES
from util.reader import DNASequenceReader, SEQ_COL, SEQ_NUM_COL, C0_COL
from util.util import FileSave, PathObtain
from util.constants import FigSubDir, GDataSubDir

C0_PREDICT = "c0_predict"


class Prediction:
    # TODO: Use default 30
    def __init__(self, model: Union[int, keras.Model] = 6):
        if isinstance(model, int):
            self._model_no = model
            self._model = self._load_model()
        elif isinstance(model, keras.Model):
            self._model = model

    def __str__(self):
        return str(self._model_no)

    def predict_metrics_lib(self, lib: LIBRARY_NAMES, n: int = None) -> None | Path:
        pred_df = self.predict_lib(lib)
        if n is not None:
            pred_df = pred_df.loc[:n]
        y = pred_df[C0_COL].to_numpy()
        y_pred = self.predict(pred_df)[C0_PREDICT].to_numpy()

        metrics_df = pd.DataFrame(
            {
                "library": [lib],
                "r2_score": [r2_score(y, y_pred)],
                "pearsons_corr": [pearsonr(y, y_pred)[0]],
                "spearmans_corr": [spearmanr(y, y_pred)[0]],
            }
        )
        if n is None:
            return FileSave.append_tsv(
                metrics_df,
                f"{PathObtain.data_dir()}/generated_data/prediction_metrics/pred_m_{self._model_no}.tsv",
            )

    def predict_lib(self, lib: LIBRARY_NAMES, n: int = None) -> pd.DataFrame:
        test_df = DNASequenceReader().get_processed_data()[lib]
        if n is not None:
            test_df = test_df.loc[:n]
        predict_df = self.predict(test_df)
        if n is None:
            FileSave.tsv(
                predict_df,
                f"{PathObtain.data_dir()}/generated_data/predictions/{lib}_pred_m_{self._model_no}.tsv",
            )
        return predict_df

    def predict(
        self,
        df: pd.DataFrame[SEQ_NUM_COL:int, SEQ_COL:str, C0_COL:float],
        print_metrics=False,
        plot_scatter=False,
    ) -> pd.DataFrame[SEQ_NUM_COL:int, SEQ_COL:str, C0_PREDICT:float]:
        prep = Preprocess(df)
        data = prep.one_hot_encode()

        y_pred = self._model.predict(
            {"forward": data["forward"], "reverse": data["reverse"]}, verbose=1
        ).flatten()

        if C0_COL in df.columns:
            y = df[C0_COL].to_numpy()
            if print_metrics:
                print("r2 score:", r2_score(y, y_pred))
                print("Pearson's correlation:", pearsonr(y, y_pred)[0])
                print("Spearman's correlation: ", spearmanr(y, y_pred)[0])

        if plot_scatter:
            self._plot_scatter(df)

        return df.assign(c0_predict=y_pred)

    def check_performance(self, df: pd.DataFrame) -> None:
        """
        Check model performance on a sequence library and return predicted values.
        """
        start_time = time.time()

        prep = Preprocess(df)
        data = prep.one_hot_encode()

        x1 = data["forward"]
        x2 = data["reverse"]
        y = data["target"]

        history2 = self._model.evaluate({"forward": x1, "reverse": x2}, y)

        print("metric values of model.evaluate: " + str(history2))
        print("metrics names are " + str(self._model.metrics_names))

        print(f"Took --- {time.time() - start_time} seconds ---")

    def _load_model(self) -> keras.Model:
        NNModel, parameter_file, weight_file = self._select_model()
        params = ParamsReader(self._model_no).get_parameters(parameter_file)
        dim_num = (-1, 50, 4)

        if self._model_no == 35:
            nn = NNModel(hyperparameters=params)
        elif self._model_no == 6 or self._model_no == 30:
            nn = NNModel(dim_num=dim_num, **params)
        model = nn.create_model()
        model.load_weights(weight_file)
        return model

    def _select_model(self) -> tuple[CNNModel, str, str]:
        parent_dir = PathObtain.parent_dir(inspect.currentframe())

        if self._model_no == 6:
            return (
                CNNModel6,
                f"{parent_dir}/parameter_model6.txt",
                f"{parent_dir}/model_weights/w6.h5_archived",
            )
        elif self._model_no == 30:
            return (
                CNNModel30,
                f"{parent_dir}/parameter_model30.txt",
                f"{parent_dir}/model_weights/w30.h5",
            )

        elif self._model_no == 35:
            return (
                CNNModel35,
                f"{parent_dir}/parameter_model35.txt",
                f"{parent_dir}/model_weights/model35_parameters_parameter_274",
            )

    def plot_model(self):
        f = f"{PathObtain.figure_dir()}/{FigSubDir.MODELS}/cnn_{self._model_no}.png"
        print(f)
        plot_model(
            self._model,
            to_file=f,
            show_shapes=True,
            show_layer_names=True,
        )

    def _plot_scatter(self, df: pd.DataFrame) -> None:
        p = (
            ggplot(data=df, mapping=aes(x="True Value", y="Predicted Value"))
            + stat_bin_2d(bins=150)
            + xlim(-2.75, 2.75)
            + ylim(-2.75, 2.75)
        )

        with open(f"{PathObtain.figure_dir()}/scatter.png", "w") as f:
            print(p, file=f)


def save_kernel_weights_logos(model):
    with open("kernel_weights/6", "w") as f:
        for layer_num in range(2, 3):
            layer = model.layers[layer_num]
            config = layer.get_config()
            print(config, file=f)
            weights = layer.get_weights()
            w = tf.transpose(weights[0], [2, 0, 1])
            alpha = 20
            # beta = 1 / alpha
            # bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
            # bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(
                lambda x: tf.math.exp(
                    tf.subtract(
                        tf.subtract(
                            tf.math.scalar_mul(alpha, x),
                            tf.expand_dims(
                                tf.math.reduce_max(
                                    tf.math.scalar_mul(alpha, x), axis=1
                                ),
                                axis=1,
                            ),
                        ),
                        tf.expand_dims(
                            tf.math.log(
                                tf.math.reduce_sum(
                                    tf.math.exp(
                                        tf.subtract(
                                            tf.math.scalar_mul(alpha, x),
                                            tf.expand_dims(
                                                tf.math.reduce_max(
                                                    tf.math.scalar_mul(alpha, x), axis=1
                                                ),
                                                axis=1,
                                            ),
                                        )
                                    ),
                                    axis=1,
                                )
                            ),
                            axis=1,
                        ),
                    )
                ),
                w,
            )

            npa = np.array(filt_list)
            print(npa, file=f)
            # print(npa.shape[0])
            for i in range(npa.shape[0]):
                df = pd.DataFrame(npa[i], columns=["A", "C", "G", "T"]).T
                # print(df.head())
                df.to_csv(
                    "kernel_weights/6.csv", mode="a", sep="\t", float_format="%.3f"
                )

            for i in range(npa.shape[0]):
                for rows in range(npa[i].shape[0]):
                    ownlog = np.array(npa[i][rows])
                    for cols in range(ownlog.shape[0]):
                        ownlog[cols] = ownlog[cols] * np.log2(ownlog[cols])
                    scalar = np.cumsum(ownlog, axis=0) + 2
                    npa[i][rows] *= scalar
                df = pd.DataFrame(npa[i], columns=["A", "C", "G", "T"])
                print(df.head())
                lm.Logo(
                    df,
                    font_name="Arial Rounded MT Bold",
                )
                # plt.show()
                plt.savefig(
                    "logos/l6/logo" + str(layer_num) + "_" + str(i) + ".png", dpi=50
                )

