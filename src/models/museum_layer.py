import tensorflow as tf

from keras.layers import Conv1D


class MultinomialConvolutionLayer(Conv1D):
    def __init__(
        self,
        alpha,
        beta,
        filters,
        kernel_size,
        background=[0.295, 0.205, 0.205, 0.295],
        data_format="channels_last",
        padding="valid",
        activation=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        __name__="ConvolutionLayer",
        **kwargs
    ):
        super(MultinomialConvolutionLayer, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            data_format=data_format,
            padding=padding,
            kernel_regularizer=kernel_regularizer,
            **kwargs
        )

        self.run_value = 1
        self.alpha = alpha
        self.beta = beta
        self.background = background

    def call(self, inputs):
        if self.run_value > 2:
            x_tf = tf.transpose(self.kernel, [2, 0, 1])

            bkg_tf = tf.constant(self.background, dtype=tf.float32)
            filt_list = tf.map_fn(
                lambda x: tf.math.scalar_mul(
                    self.beta,
                    tf.subtract(
                        tf.subtract(
                            tf.subtract(
                                tf.math.scalar_mul(self.alpha, x),
                                tf.expand_dims(
                                    tf.math.reduce_max(
                                        tf.math.scalar_mul(self.alpha, x), axis=1
                                    ),
                                    axis=1,
                                ),
                            ),
                            tf.expand_dims(
                                tf.math.log(
                                    tf.math.reduce_sum(
                                        tf.math.exp(
                                            tf.subtract(
                                                tf.math.scalar_mul(self.alpha, x),
                                                tf.expand_dims(
                                                    tf.math.reduce_max(
                                                        tf.math.scalar_mul(
                                                            self.alpha, x
                                                        ),
                                                        axis=1,
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
                        ),
                        tf.math.log(
                            tf.reshape(
                                tf.tile(bkg_tf, [tf.shape(x)[0]]),
                                [tf.shape(x)[0], tf.shape(bkg_tf)[0]],
                            )
                        ),
                    ),
                ),
                x_tf,
            )

            transf = tf.transpose(filt_list, [1, 2, 0])
            outputs = self.convolution_op(inputs, transf)
        else:
            outputs = self.convolution_op(inputs, self.kernel)
        self.run_value += 1
        return outputs

    def get_motifs(self):
        x_tf = tf.transpose(self.kernel, [2, 0, 1])

        bkg_tf = tf.constant(self.background, dtype=tf.float32)
        ll = tf.map_fn(
            lambda x: tf.subtract(
                tf.subtract(
                    tf.subtract(
                        tf.math.scalar_mul(self.alpha, x),
                        tf.expand_dims(
                            tf.math.reduce_max(
                                tf.math.scalar_mul(self.alpha, x), axis=1
                            ),
                            axis=1,
                        ),
                    ),
                    tf.expand_dims(
                        tf.math.log(
                            tf.math.reduce_sum(
                                tf.math.exp(
                                    tf.subtract(
                                        tf.math.scalar_mul(self.alpha, x),
                                        tf.expand_dims(
                                            tf.math.reduce_max(
                                                tf.math.scalar_mul(self.alpha, x),
                                                axis=1,
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
                ),
                tf.math.log(
                    tf.reshape(
                        tf.tile(bkg_tf, [tf.shape(x)[0]]),
                        [tf.shape(x)[0], tf.shape(bkg_tf)[0]],
                    )
                ),
            ),
            x_tf,
        )
        prob = tf.map_fn(
            lambda x: tf.multiply(
                tf.reshape(
                    tf.tile(bkg_tf, [tf.shape(x)[0]]),
                    [tf.shape(x)[0], tf.shape(bkg_tf)[0]],
                ),
                tf.exp(x),
            ),
            ll,
        )

        plp = tf.scalar_mul(1.442695041, tf.multiply(prob, ll))
        ic = tf.reduce_sum(plp, axis=2)
        ic_scaled_prob = tf.multiply(prob, tf.expand_dims(ic, axis=2))
        icpp = tf.reduce_mean(ic, axis=1)

        return prob, ic_scaled_prob

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_beta(self, beta):
        self.beta = beta
