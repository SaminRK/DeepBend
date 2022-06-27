import tensorflow as tf

from keras import backend as K
from scipy.stats import spearmanr, pearsonr


def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def spearman_fn(y_true, y_pred):
    return tf.py_function(
        spearmanr,
        [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)],
        Tout=tf.float32,
    )


def coeff_determination_loss(y_true, y_pred):
    return 1 - coeff_determination(y_true, y_pred)
