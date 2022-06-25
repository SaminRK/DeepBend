import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Custom', name='lvariance')
class LVariance(tf.keras.regularizers.Regularizer):
    def __init__(self, lvariance=0.):
        self.lvariance = lvariance

    def __call__(self, x):
        return self.lvariance * tf.math.reduce_sum(tf.math.reduce_variance(x, axis=0))

    def get_config(self):
        return {'lvariance': float(self.lvariance)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='l1_variance')
class L1Variance(tf.keras.regularizers.Regularizer):
    def __init__(self, l1=.0, lvariance=0.):
        self.l1 = l1
        self.lvariance = lvariance

    def __call__(self, x):
        return self.l1 * tf.math.reduce_sum(tf.math.abs(x)) + self.lvariance * tf.math.reduce_sum(tf.math.reduce_variance(x, axis=0))

    def get_config(self):
        return {'l1': float(self.l1), 'lvariance': float(self.lvariance)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='lrange')
class LRange(tf.keras.regularizers.Regularizer):
    def __init__(self, lrange=0., limit=.05):
        self.lrange = lrange
        self.limit = limit

    def __call__(self, x):
        means = tf.math.reduce_mean(x, axis=0, keepdims=True)
        abs_diff = tf.math.abs(x - means)
        out_boundary = tf.maximum(abs_diff, self.limit) - self.limit
        out_bounary_variance = tf.reduce_mean(tf.square(out_boundary), axis=0)
        return self.lrange * tf.math.reduce_sum(tf.math.reduce_variance(out_bounary_variance, axis=0))

    def get_config(self):
        return {'lrange': float(self.lrange)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='l1_l2')
class L1_L2(tf.keras.regularizers.Regularizer):
    def __init__(self, l1_l2=0.):
        self.l1_l2 = l1_l2

    def __call__(self, x):
        return self.l1_l2 * tf.math.reduce_sum(tf.math.square(tf.math.reduce_sum(tf.math.abs(x), axis=1)))

    def get_config(self):
        return {'l1_l2': float(self.l1_l2)}


@tf.keras.utils.register_keras_serializable(package='Custom', name='lentropy')
class LEntropy(tf.keras.regularizers.Regularizer):
    def __init__(self, lentropy=.0, alpha=.0):
        self.lentropy = lentropy
        self.alpha = alpha

    def __call__(self, x_tf):
        x_tf = tf.transpose(x_tf, [2, 0, 1])

        bkg = tf.constant([0.295, 0.205, 0.205, 0.295])
        bkg_tf = tf.cast(bkg, tf.float32)
        ll = tf.map_fn(lambda x:
                       tf.subtract(tf.subtract(tf.subtract(tf.math.scalar_mul(self.alpha, x),
                                                           tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis=1), axis=1)),
                                               tf.expand_dims(tf.math.log(tf.math.reduce_sum(tf.math.exp(tf.subtract(tf.math.scalar_mul(self.alpha, x),
                                                                                                                     tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(self.alpha, x), axis=1), axis=1))), axis=1)), axis=1)),
                                   tf.math.log(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]]))), x_tf)
        prob = tf.map_fn(lambda x:
                         tf.multiply(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]]), tf.exp(x)), ll)

        return -self.lentropy * tf.math.reduce_sum(tf.multiply(prob, tf.math.log(prob)))

    def get_config(self):
        return {'lentropy': float(self.lentropy)}
