from __future__ import annotations
import sys
import random
import inspect

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import History

from .data_preprocess import Preprocess
from .cnnmodel import CNNModel6
from .prediction import get_parameters
from util.constants import TL, RL
from util.reader import DNASequenceReader
from util.util import PathObtain


def train(save=False, train_lib=TL, val_lib=RL) -> tuple[keras.Model, History]:
    # TODO: Use kwargs to overwrite parameter config
    # Reproducibility
    seed = random.randint(1, 1000)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    params = get_parameters(
        f"{PathObtain.parent_dir(inspect.currentframe())}/parameter_model6.txt"
    )

    model = CNNModel6(dim_num=(-1, 50, 4), **params).create_model()

    train_prep = Preprocess(DNASequenceReader().get_processed_data()[train_lib][:2000])
    train_data = train_prep.one_hot_encode()  # Mono-nucleotide seqs
    # dict = prep.dinucleotide_encode()      # Di-nucleotide seqs

    val_prep = Preprocess(DNASequenceReader().get_processed_data()[val_lib][:1000])
    val_data = val_prep.one_hot_encode()  # Mono-nucleotide sequences
    # dict = prep.dinucleotide_encode()      # Di-nucleotide seqs

    np.set_printoptions(threshold=sys.maxsize)

    # Without early stopping
    history = model.fit(
        {"forward": train_data["forward"], "reverse": train_data["reverse"]},
        train_data["target"],
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_data=(
            {"forward": val_data["forward"], "reverse": val_data["reverse"]},
            val_data["target"],
        ),
    )

    # Early stopping
    # callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    # callback = EarlyStopping(monitor='val_spearman_fn', min_delta=0.0001, patience=3, verbose=0, mode='max', baseline=None, restore_best_weights=False)
    # history = model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1, callbacks = [callback])

    print("Seed number is {}".format(seed))

    if save:
        model.save_weights("model_weights/w6.h5", save_format="h5")

    return model, history
