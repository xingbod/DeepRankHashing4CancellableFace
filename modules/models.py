import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
)
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50
)
from .layers import (
    BatchNormalization,
    ArcMarginPenaltyLogists,
    MaxIndexLinearForeward,
    MaxIndexLinearTraining,
    PermLayer
)
from losses.sampling_matters.margin_loss import MarginLossLayer


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def Backbone(backbone_type='ResNet50', use_pretrain=True):
    """Backbone Model"""
    weights = None
    if use_pretrain:
        weights = 'imagenet'

    def backbone(x_in):
        if backbone_type == 'ResNet50':
            return ResNet50(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        else:
            raise TypeError('backbone_type error!')

    return backbone


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""

    def output_layer(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = BatchNormalization()(x)
        return Model(inputs, x, name=name)(x_in)

    return output_layer


def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""

    def arc_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))

    return arc_head


def MarginLossHead(num_classes=85742, margin=0.5, logist_scale=64, cfg=None, name='MlossHead'):
    """MarginLoss Head"""

    def mloss_head(x_in, y_in):
        x = inputs1 = Input(x_in.shape[1:])
        y = Input(y_in.shape[1:])
        x = MarginLossLayer(num_classes=num_classes, cfg=cfg)(x, y)
        return Model((inputs1, y), x, name=name)((x_in, y_in))

    return mloss_head


def IoMHead(m, q, isTraining=True, T=1, name='IoMHead'):
    """Arc Head"""

    def iom_head(x_in):
        x = inputs1 = Input(x_in.shape[1:])
        if isTraining:
            x = MaxIndexLinearTraining(units=m * q, q=q, T=T)(x)  # permutation
        else:
            x = MaxIndexLinearForeward(units=m * q, q=q)(x)  # permutation
        return Model(inputs1, x, name=name)(x_in)

    return iom_head


def NormHead(num_classes, w_decay=5e-4, name='NormHead'):
    """Norm Head"""

    def norm_head(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = Dense(num_classes, kernel_regularizer=_regularizer(w_decay))(x)
        return Model(inputs, x, name=name)(x_in)

    return norm_head


def ArcFaceModel(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False, cfg=None):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    if training:
        assert num_classes is not None
        labels = Input([], name='label')
        if head_type == 'ArcHead':
            logist = ArcHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        else:
            logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
        return Model((inputs, labels), logist, name=name)
    else:
        if head_type == 'IoMHead':
            labels = Input([], name='label')
            logist = IoMHead(m=embd_shape, q=cfg['q'], permKey=None, isTraining=training)(embds,
                                                                                          labels)  # loss need to change
            return Model(inputs, logist, name=name)
        else:
            return Model(inputs, embds, name=name)


'''
def ArcFaceModel2(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False,cfg=None):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)

    x = OutputLayer(embd_shape, w_decay=w_decay)(x)

    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
              name="IoMProjection")(x)  # extra connection layer
    embds = IoMHead(m=cfg['m'], q=cfg['q'], isTraining=training)(x)  # loss need to change
    if training:
        assert num_classes is not None
        labels = Input([], name='label')
        if head_type == 'ArcHead':
            logist = ArcHead(num_classes=num_classes, margin=margin,
                             logist_scale=logist_scale)(embds, labels)
        else:
            logist = NormHead(num_classes=num_classes, w_decay=w_decay)(embds)
        return Model((inputs, labels), logist, name=name)
    else:
        return Model(inputs, embds, name=name)
        '''
'''
def IoMFaceModel(size=None, channels=3, num_classes=None, name='IoMface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 head_type='ArcHead', backbone_type='ResNet50',
                 w_decay=5e-4, use_pretrain=True, training=False,permKey=None,cfg=None):
    """Arc Face Model"""
    x = inputs = Input([size, size, channels], name='input_image')

    x = Backbone(backbone_type=backbone_type, use_pretrain=use_pretrain)(x)
    x = OutputLayer(embd_shape, w_decay=w_decay)(x)
    if permKey is not None:
        x = PermLayer(permKey)(x)  # permutation before project to IoM hash code
    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),name="IoMProjection")(
        x)  # extra connection layer
    # if training:
    #     labels = Input([], name='label')
    #     logist = IoMHead(m=cfg['m'],q=cfg['q'] ,isTraining=training)(x, labels) # loss need to change
    #     return Model((inputs, labels), logist, name=name)
    # else:
    #     labels = Input([], name='label')
    #     logist = IoMHead(m=cfg['m'], q=cfg['q'], isTraining=training)(x,labels)  # loss need to change
    #     return Model(inputs, logist, name=name)

    if training:
        logist = IoMHead(m=cfg['m'],q=cfg['q'],isTraining=training)(x) # loss need to change
    else:
        logist = IoMHead(m=cfg['m'], q=cfg['q'], isTraining=training)(x)  # loss need to change
    return Model(inputs, logist, name=name)
'''


def IoMFaceModelFromArFace(size=None, channels=3, arcmodel=None, name='IoMface_model',
                           margin=0.5, logist_scale=64, embd_shape=512,
                           head_type='ArcHead', backbone_type='ResNet50',
                           w_decay=5e-4, use_pretrain=True, training=False, permKey=None, cfg=None):
    """IoMFaceModelFromArFace Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = arcmodel(x)
    if not (permKey is None):
        x = PermLayer(permKey)(x)  # permutation before project to IoM hash code
    # here I add one extra hidden layer
    # x = Dense(1024, kernel_regularizer=_regularizer(w_decay))(x)
    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
              name="IoMProjection")(x)  # extra connection layer
    logist = IoMHead(m=cfg['m'], q=cfg['q'], isTraining=training)(x)  # loss need to change
    return Model(inputs, logist, name=name)


def IoMFaceModelFromArFace2(size=None, channels=3, arcmodel=None, name='IoMface_model',
                            margin=0.5, logist_scale=64, embd_shape=512,
                            head_type='ArcHead', backbone_type='ResNet50',
                            w_decay=5e-4, use_pretrain=True, training=False, permKey=None, cfg=None):
    """IoMFaceModelFromArFace Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = arcmodel(x)
    if not (permKey is None):
        x = PermLayer(permKey)(x)  # permutation before project to IoM hash code
    # here I add one extra hidden layer
    x = Dense(1024, kernel_regularizer=_regularizer(w_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
              name="IoMProjection")(x)  # extra connection layer
    logist = IoMHead(m=cfg['m'], q=cfg['q'], isTraining=training)(x)  # loss need to change
    return Model(inputs, logist, name=name)


def IoMFaceModelFromArFace3(size=None, channels=3, arcmodel=None, name='IoMface_model',
                            margin=0.5, logist_scale=64, embd_shape=512,
                            head_type='ArcHead', backbone_type='ResNet50',
                            w_decay=5e-4, use_pretrain=True, training=False, permKey=None, cfg=None):
    """IoMFaceModelFromArFace Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = arcmodel(x)
    if not (permKey is None):
        x = PermLayer(permKey)(x)  # permutation before project to IoM hash code
    # here I add one extra hidden layer
    x = Dense(512, kernel_regularizer=_regularizer(w_decay), activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(512, kernel_regularizer=_regularizer(w_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
              name="IoMProjection")(x)  # extra connection layer
    logist = IoMHead(m=cfg['m'], q=cfg['q'], isTraining=training)(x)  # loss need to change
    return Model(inputs, logist, name=name)


def IoMFaceModelFromArFace_T(size=None, channels=3, arcmodel=None, name='IoMface_model',
                             margin=0.5, logist_scale=64, embd_shape=512,
                             head_type='ArcHead', backbone_type='ResNet50',
                             w_decay=5e-4, use_pretrain=True, training=False, permKey=None, cfg=None):
    """IoMFaceModelFromArFace Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = arcmodel(x)
    if not (permKey is None):
        x = PermLayer(permKey)(x)  # permutation before project to IoM hash code
    # here I add one extra hidden layer
    x = Dense(512, kernel_regularizer=_regularizer(w_decay), activation='relu')(x)
    # x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(512, kernel_regularizer=_regularizer(w_decay), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
              name="IoMProjection")(x)  # extra connection layer
    logist = IoMHead(m=cfg['m'], q=cfg['q'], T=0.01, isTraining=training)(x)  # loss need to change
    return Model(inputs, logist, name=name)


def IoMFaceModelFromArFaceMLossHead(size=None, channels=3, arcmodel=None, name='IoM_Mloss_Head_model',
                                    margin=0.5, logist_scale=64, embd_shape=512,
                                    head_type='ArcHead', backbone_type='ResNet50',
                                    w_decay=5e-4, use_pretrain=True, training=False, permKey=None, cfg=None):
    """IoMFaceModelFromArFace Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = arcmodel(x)
    if not (permKey is None):
        x = PermLayer(permKey)(x)  # permutation before project to IoM hash code
    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
              name="IoMProjection")(
        x)  # extra connection layer

    hashcodes = IoMHead(m=cfg['m'], q=cfg['q'], isTraining=training)(x)  # loss need to change

    if training:
        labels = Input([], name='label')
        logist = MarginLossHead(cfg=cfg)(hashcodes, labels)
        return Model((inputs, labels), logist, name=name)
    else:
        return Model(inputs, hashcodes, name=name)
