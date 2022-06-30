import tensorflow as tf
import keras

from keras.layers import Conv1D, Flatten, ReLU, Maximum
from keras import regularizers, optimizers

from .museum_layer import MultinomialConvolutionLayer
from .functions import coeff_determination, spearman_fn
from .utils import get_loss
from .custom_regularizers import LRange, LVariance, L1Variance, LEntropy

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


class nn_model:
    def __init__(
        self,
        dim_num=(50, 4),
        filters_conv_0=128,
        filters_conv_1=32,
        alpha=75.0,
        beta=1 / 75.0,
        kernel_size_0=8,
        kernel_size_1=48,
        regularizer_2="lrange",
        loss_func="coeff_determination",
        optimizer="Adam",
        hyperparameters=None,
    ):
        """initialize basic parameters"""

        if hyperparameters:
            self.filters_conv_0 = hyperparameters.get("filters_0", filters_conv_0)
            self.filters_conv_1 = hyperparameters.get("filters_1", filters_conv_1)
            self.kernel_size_0 = hyperparameters.get("kernel_size_0", kernel_size_0)
            self.kernel_size_1 = hyperparameters.get("kernel_size_1", kernel_size_1)
            self.regularizer_2 = hyperparameters.get("regularizer_2", regularizer_2)
            self.loss_func = hyperparameters.get("loss_func", loss_func)
            self.optimizer = hyperparameters.get("optimizer", optimizer)
            self.alpha = hyperparameters.get("alpha", alpha)
            self.beta = hyperparameters.get("beta", beta)
            self.multinomial_bkg = [
                hyperparameters.get("A", 0.25),
                hyperparameters.get("C", 0.25),
                hyperparameters.get("G", 0.25),
                hyperparameters.get("T", 0.25),
            ]
        else:
            self.filters_conv_0 = filters_conv_0
            self.filters_conv_1 = filters_conv_1
            self.kernel_size_0 = kernel_size_0
            self.kernel_size_1 = kernel_size_1
            self.regularizer_2 = regularizer_2
            self.loss_func = loss_func
            self.optimizer = optimizer
            self.alpha = alpha
            self.beta = beta
            self.multinomial_bkg = [0.25, 0.25, 0.25, 0.25]

        self.dim_num = dim_num

    def create_model(self):
        # building model
        # To build this model with the functional API,
        # you would start by creating an input node:
        forward = keras.Input(shape=self.dim_num, name="forward")
        reverse = keras.Input(shape=self.dim_num, name="reverse")

        first_layer_1 = MultinomialConvolutionLayer(
            alpha=self.alpha,
            beta=self.beta,
            filters=self.filters_conv_0,
            kernel_size=self.kernel_size_0,
            background=self.multinomial_bkg,
            strides=1,
            data_format="channels_last",
            use_bias=True,
            padding="same",
        )

        fw_1 = first_layer_1(forward)
        rc_1 = first_layer_1(reverse)

        fw_relu_1 = ReLU()(fw_1)
        rc_relu_1 = ReLU()(rc_1)

        conv_2 = Conv1D(
            filters=self.filters_conv_1,
            kernel_size=self.kernel_size_1,
            strides=1,
            data_format="channels_last",
            use_bias=True,
            kernel_initializer="normal",
            kernel_regularizer=regularizers.l2(0.0005),
            padding="same",
            use_bias=True,
            kernel_initializer="normal",
            kernel_regularizer=regularizers.l2(0.0005),
            padding="same",
        )

        fw_out_2 = conv_2(fw_relu_1)
        rc_out_2 = conv_2(rc_relu_1)

        out_2 = Maximum()([fw_out_2, rc_out_2])
        relu_2 = ReLU()(out_2)

        if self.regularizer_2 == "l1":
            reg = regularizers.l1(0.0005)
        elif self.regularizer_2 == "lrange":
            reg = LRange(lrange=relu_2.shape[1], limit=0.05)
        elif self.regularizer_2 == "lvariance":
            reg = LVariance(lvariance=relu_2.shape[1])
        elif self.regularizer_2 == "l1variance":
            reg = L1Variance(l1=0.0005, lvariance=relu_2.shape[1])
        elif self.regularizer_2 == "l2":
            reg = regularizers.l2(0.001)
        else:
            reg = None

        output_conv = Conv1D(
            filters=1,
            kernel_size=relu_2.shape[1],
            data_format="channels_last",
            use_bias=True,
            kernel_initializer="normal",
            kernel_regularizer=reg,
            activation="linear",
            use_bias=True,
            kernel_initializer="normal",
            kernel_regularizer=reg,
            activation="linear",
        )

        outputs = Flatten()(output_conv(relu_2))

        model = keras.Model(inputs=[forward, reverse], outputs=outputs)

        model.summary()

        model.compile(
            loss=get_loss(self.loss_func),
            optimizer=self.optimizer,
            metrics=[coeff_determination, spearman_fn],
        )

        return model
