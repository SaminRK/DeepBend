import tensorflow as tf

from .functions import coeff_determination_loss


def get_loss(loss_name: str):
    if loss_name == "mse":
        return "mean_squared_error"
    elif loss_name == "coeff_determination":
        return coeff_determination_loss
    elif loss_name == "huber":
        return tf.keras.losses.Huber(delta=1)
    elif loss_name == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    elif loss_name == "rank_mse":
        return "rank_mse"
    elif loss_name == "poisson":
        return tf.keras.losses.Poisson()
    else:
        raise NameError("Unrecognized Loss Function")
