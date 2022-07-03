from __future__ import annotations
from enum import Enum, auto
from datetime import datetime
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, LassoCV
from sklearn.svm import SVC, LinearSVC, SVR, LinearSVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from xgboost import XGBRegressor
import numpy as np
import tensorflow as tf
from util.util import FileSave, PathObtain
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from nptyping import NDArray

from .data_organizer import (
    DataOrganizer,
    TrainTestSequenceLibraries,
    SequenceLibrary,
    DataOrganizeOptions,
    FeatureSelector,
    ShapeOrganizerFactory,
)
from util.constants import RL, TL, GDataSubDir
from .feat_selector import FeatureSelectorFactory


class SKModels:
    regs = [
        ("Ridge_alpha_1e3", Ridge(alpha=1e3)),
        ("Linear_reg", LinearRegression()),
        (
            "HGB_reg",
            HistGradientBoostingRegressor(
                learning_rate=0.1,
                l2_regularization=1,
                max_iter=500,
                max_depth=64,
                min_samples_leaf=1,
                verbose=1
            ),
        ),
        (
            "RF_reg",
            RandomForestRegressor(
                n_estimators=50, max_depth=16, min_samples_leaf=2, n_jobs=-1, verbose=1
            ),
        ),
        ("LinSVR", make_pipeline(StandardScaler(), LinearSVR(C=5, max_iter=1000, verbose=1))),
        ("SVR_C_10", SVR(C=10, verbose=True)),
        ("XGBoost", XGBRegressor(verbosity=1, max_depth=32)),
        ("Lasso", LassoCV(verbose=1)),
        ("NN_reg", MLPRegressor(hidden_layer_sizes=(10,), verbose=True)),
    ]
    clfs = [
        ("LogisticRegression_C_1", LogisticRegression(C=1)),
        ("SVC", SVC()),
        ("GradientBoost", GradientBoostingClassifier()),
        ("NN_clf", MLPClassifier()),
        ("KNN_clf", KNeighborsClassifier()),
        ("RF_clf", RandomForestClassifier(n_estimators=5, max_depth=32)),
    ]

    @classmethod
    def regressors(
        cls, nums: list[int]
    ) -> list[tuple[str, sklearn.base.BaseEstimator]]:
        return list(map(lambda n: cls.regs[n], nums))

    @classmethod
    def classifiers(cls, nums: list[int]):
        return list(map(lambda n: cls.clfs[n], nums))


class Model:
    """
    A class to train model on DNA mechanics libraries.
    """

    @classmethod
    def run_seq_regression(
        self, X_train, X_test, y_train, y_test, nums: list[int] = [0]
    ) -> None:
        """
        Runs Scikit-learn regression models to classify C0 value with k-mer count & helical separation.
        """
        nums = [0, 4, 3, 6, 8]
        regressors = SKModels.regressors(nums)

        result_cols = [
            "Date",
            "Time",
            "Regression Model",
            "Test pearson",
            "Test R2",
            "Train pearson",
            "Train R2",
        ]
        reg_result = pd.DataFrame(columns=result_cols)
        for name, reg in regressors:
            reg.fit(X_train, y_train)
            test_p, _ = pearsonr(y_test, reg.predict(X_test))
            test_acc = reg.score(X_test, y_test)
            train_p, _ = pearsonr(y_train, reg.predict(X_train))
            train_acc = reg.score(X_train, y_train)

            print(
                f"model: {name}, train p: {train_p}, train r2: {train_acc}"
                f", test p: {test_p}, test r2: {test_acc}"
            )

            cur_date = datetime.now().strftime("%Y_%m_%d")
            cur_time = datetime.now().strftime("%H_%M")
            reg_result = pd.concat(
                [
                    reg_result,
                    pd.DataFrame(
                        [
                            [
                                cur_date,
                                cur_time,
                                name,
                                test_p,
                                test_acc,
                                train_p,
                                train_acc,
                            ]
                        ],
                        columns=result_cols,
                    ),
                ],
                ignore_index=True,
            )
            FileSave.append_tsv(
                reg_result,
                f"{PathObtain.gen_data_dir()}/{GDataSubDir.ML_MODEL}/regression.tsv",
            )

    @classmethod
    def run_seq_classifier(
        self,
        X_train: NDArray,
        X_test: NDArray,
        y_train: NDArray,
        y_test: NDArray,
        nums: list[int] = [0],
    ) -> None:
        """
        Runs Scikit-learn classifier to classify C0 value with k-mer count & helical separation.
        """
        classifiers = SKModels.classifiers(nums)

        result_cols = ["Date", "Time", "Classifier", "Test Accuracy", "Train Accuracy"]
        clf_result = pd.DataFrame(columns=result_cols)
        for name, clf in classifiers:
            clf.fit(X_train, y_train)
            test_acc = clf.score(X_test, y_test)
            train_acc = clf.score(X_train, y_train)
            print("Model:", name, "Train acc:", train_acc, ", Test acc:", test_acc)
            cur_date = datetime.now().strftime("%Y_%m_%d")
            cur_time = datetime.now().strftime("%H_%M")
            clf_result = pd.concat(
                [
                    clf_result,
                    pd.DataFrame(
                        [[cur_date, cur_time, name, test_acc, train_acc]],
                        columns=result_cols,
                    ),
                ],
                ignore_index=True,
            )
            FileSave.append_tsv(
                clf_result,
                f"{PathObtain.gen_data_dir()}/{GDataSubDir.ML_MODEL}/classification.tsv",
            )

    @classmethod
    def run_shape_cnn_classifier(
        self,
        X_train_valid: NDArray,
        X_test: NDArray,
        y_train_valid: NDArray,
        y_test: NDArray,
    ) -> None:
        """
        Run a CNN classifier on DNA shape values.

        Args:
            shape_name: Name of structural feature
            c0_range_split: A numpy 1D array denoting the point of split for classification

        """
        model, history = self._train_shape_cnn_classifier(X_train_valid, y_train_valid)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print("test_acc", test_acc)

        # Plot
        plt.plot(history.history["accuracy"], label="accuracy")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.show()

    @classmethod
    def _train_shape_cnn_classifier(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[models.Sequential, tf.keras.callbacks.History]:
        """
        Train a CNN model on data

        Args:
            X: training and validation samples
            y: training and validation targets

        Returns:
            A tuple containing
                * model: a tensorflow sequential model
                * history: history object of training
        """
        kernel_size = 8
        model = models.Sequential()
        model.add(
            layers.Conv2D(
                filters=64,
                kernel_size=(X.shape[1], kernel_size),
                strides=1,
                activation="relu",
                input_shape=(X.shape[1], X.shape[2], 1),
            )
        )
        model.add(layers.MaxPooling2D((2, 2), padding="same"))

        model.add(layers.Flatten())
        model.add(layers.Dense(50, activation="relu"))
        model.add(layers.Dense(np.unique(y).size))

        print(model.summary())

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

        history = model.fit(
            X_train, y_train, epochs=15, validation_data=(X_valid, y_valid)
        )

        return model, history

    @classmethod
    def run_shape_seq_classifier(self) -> None:
        pass


class LibrariesParam:
    tl_rl = TrainTestSequenceLibraries(
        train=[SequenceLibrary(name=TL, quantity=20000)],
        test=[SequenceLibrary(name=RL, quantity=5000)],
    )


class ModelCat(Enum):
    CLASSIFIER = auto()
    REGRESSOR = auto()
    SHAPE_CLASSIFIER = auto()


class ModelRunner:
    @classmethod
    def run_model(
        cls,
        libs: TrainTestSequenceLibraries,
        options: DataOrganizeOptions,
        featsel: FeatureSelector,
        cat: ModelCat,
    ):
        shape_organizer = None
        if cat == ModelCat.SHAPE_CLASSIFIER:
            shape_factory = ShapeOrganizerFactory("normal", "ProT")
            shape_organizer = shape_factory.make_shape_organizer(libs)

        organizer = DataOrganizer(libs, shape_organizer, featsel, options)

        if cat == ModelCat.CLASSIFIER:
            Model.run_seq_classifier(*organizer.get_seq_train_test(classify=True))
        elif cat == ModelCat.REGRESSOR:
            Model.run_seq_regression(*organizer.get_seq_train_test(classify=False))
        elif cat == ModelCat.SHAPE_CLASSIFIER:
            Model.run_shape_cnn_classifier(*organizer.get_shape_train_test())
