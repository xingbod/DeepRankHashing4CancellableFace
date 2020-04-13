import tensorflow as tf
import math
import tensorflow.keras.backend as K


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """
    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists

class MaxIndexLinearForeward(tf.keras.layers.Layer):

    def __init__(self, units=32, q=4):
        super(MaxIndexLinearForeward, self).__init__()
        self.units = units
        self.q = q
        self.iteratenum = tf.dtypes.cast(units / self.q, tf.int32)
        self.helpvector = tf.cast(tf.range(0, self.q, 1) + 1, tf.double)

    def call(self, inputs):
        myvar = []
        for i in range(0, self.iteratenum.numpy()):  # q=4
            my_variable1 = inputs[:, i * self.q + 0:i * self.q + self.q]
            init_index = K.argmax(my_variable1)

            myvar.append(init_index)
        myvar = tf.stack(myvar)
        myvar = tf.transpose(myvar)
        return myvar

class MaxIndexLinearTraining(tf.keras.layers.Layer):

  def __init__(self, units=32,q=4):
    super(MaxIndexLinearTraining, self).__init__()
    self.units = units
    self.q = q
    self.iteratenum = tf.dtypes.cast(units/self.q, tf.int32)
    self.helpvector =  tf.cast( tf.range(0, self.q , 1) + 1,tf.double)


  def call(self, inputs):
    myvar=[]
    for i in range(0,self.iteratenum.numpy()): # q=4
        my_variable1 = inputs[:,i*self.q+0:i*self.q+self.q]
        my_variable1 = tf.nn.softmax(my_variable1, axis=1)
        my_variable1 = tf.cast(my_variable1, tf.double)
        # init_index = K.argmax(res[:,0*self.q+0:0*self.q+self.q-1])
        softargmax2 = tf.multiply(self.helpvector, my_variable1)
        softargmax2 = tf.reduce_sum(softargmax2, axis=-1)
        myvar.append(softargmax2)
        #softargmax = tf.stack((softargmax, softargmax2), axis=-1)
    #print(init_index)
    #return  tf.stack(my_variable, axis=-1)
    myvar = tf.stack(myvar)
    myvar = tf.cast(myvar,tf.float64)
    myvar = tf.transpose(myvar)
    #print(myvar)
    return myvar
class PermLayer(tf.keras.layers.Layer):

    def __init__(self, permKey):
        super(PermLayer, self).__init__()
        self.permKey = permKey

    def call(self, inputs):
        return tf.matmul(inputs,self.permKey)

