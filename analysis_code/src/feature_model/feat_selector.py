from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import boruta


class FeatureSelector:
    """
    Selects Features for classification

    Attributes
        support_: A numpy 1D array of boolean denoting which features were selected
        ranking: A numpy 1D array of integer denoting ranking of features
    """

    def __init__(self):
        """
        Selects features
        """
        self.support_ = None
        self.ranking_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise Exception("Subclass responsibility")

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise Exception("Subclass responsibility")


class ManualFeatureSelector(FeatureSelector):
    """
    Select features that are strictly increasing/decreasing for classes 0 to
    n-1.
    """

    def _check_increasing(self, arr: np.ndarray[float]) -> np.ndarray[bool]:
        return np.all(arr[1:, :] > arr[:-1, :], axis=0)

    def _check_decreasing(self, arr: np.ndarray[float]) -> np.ndarray[bool]:
        return np.all(arr[1:, :] < arr[:-1, :], axis=0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        mean_list = []
        for i in np.unique(y):
            mean_list.append(X[np.where(y == i)].mean(axis=0))

        mean_arr = np.array(mean_list)
        assert mean_arr.shape == (np.unique(y).size, X.shape[1])

        # Check which columns are either increasing or decreasing
        self.support_ = self._check_increasing(mean_arr) | self._check_decreasing(
            mean_arr
        )
        self.ranking_ = (~self.support_).astype(int) + 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.support_]


class BorutaFeatureSelector(FeatureSelector):
    """
    A wrapper class for Boruta feature selection algorithm
    """

    def __init__(self):
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)

        self._feat_selector = boruta.BorutaPy(
            rf, n_estimators="auto", verbose=0, random_state=1, perc=90, max_iter=50
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Select all relevant features with Boruta algorithm

        Args:
            X: feature array
            y: target

        Returns:
            None
        """
        self._feat_selector.fit(X, y)

        self.support_ = self._feat_selector.support_
        self.ranking_ = self._feat_selector.ranking_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._feat_selector.transform(X)


class AllFeatureSelector(FeatureSelector):
    """
    Selects all features
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.support_ = np.full(shape=(X.shape[1],), fill_value=True)
        self.ranking_ = np.full(shape=(X.shape[1],), fill_value=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.support_]


class CorrelationFeatureSelector(FeatureSelector):
    """
    Selects features whose correlation value with target are greater than
    threshold.
    """

    def __init__(self, threshold=0.05):
        self._threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        feat_corr = abs(np.corrcoef(X.transpose(), y)[-1, :-1])
        assert feat_corr.shape == (X.shape[1],)

        self.support_ = feat_corr > self._threshold
        self.ranking_ = (~self.support_).astype(int) + 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.support_]


class FeatureSelectorFactory:
    def __init__(
        self,
        selector_type: Literal["manual"]
        | Literal["boruta"]
        | Literal["corr"]
        | Literal["all"],
    ):
        self.selector_type = selector_type

    def make_feature_selector(self):
        """Creates feature selector instance"""
        if self.selector_type == "manual":
            return ManualFeatureSelector()
        elif self.selector_type == "boruta":
            return BorutaFeatureSelector()
        elif self.selector_type == "corr":
            return CorrelationFeatureSelector()
        elif self.selector_type == "all":
            return AllFeatureSelector()
        else:
            raise Exception("Selector type not recognized")
