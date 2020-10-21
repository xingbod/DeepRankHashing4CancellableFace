import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Input,
    concatenate,
)
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    ResNet101,
    InceptionResNetV2,
    InceptionV3,
    Xception,
    VGG16,
    VGG19
)
from .layers import (
    BatchNormalization,
    ArcMarginPenaltyLogists,
    MaxIndexLinearForeward,
    MaxIndexLinearTraining,
    PermLayer
)
from losses.sampling_matters.margin_loss import MarginLossLayer
from modules.keras_resnet100 import KitModel
from modules.keras_resnet50 import KitModel_50


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
        elif backbone_type == 'ResNet101':
            return ResNet101(input_shape=x_in.shape[1:], include_top=False,
                            weights=weights)(x_in)
        elif backbone_type == 'MobileNetV2':
            return MobileNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'InceptionResNetV2':
            return InceptionResNetV2(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'InceptionV3':
            return InceptionV3(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'Xception':
            return Xception(input_shape=x_in.shape[1:], include_top=False,
                               weights=weights)(x_in)
        elif backbone_type == 'lresnet100e_ir':
            from .lresnet100e_ir import LResNet100E_IR
            return LResNet100E_IR()(x_in)
        elif backbone_type == 'VGG16':
            return VGG16(input_shape=x_in.shape[1:], include_top=False,
                         weights=weights)(x_in)
        elif backbone_type == 'VGG19':
            return VGG19(input_shape=x_in.shape[1:], include_top=False,
                         weights=weights)(x_in)
        elif backbone_type == 'Insight_ResNet100':# here use the pretrained model build by Insightface team
            print('[*] Loading Insightface pre-train model!')
            return KitModel('pre_models/resnet100/resnet100.npy')(x_in)
        elif backbone_type == 'Insight_ResNet50':# here use the pretrained model build by Insightface team
            return KitModel_50('pre_models/resnet50/resnet50.npy')(x_in)
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


def IoMProjectionLayer(cfg, name='IoMProjectionLayer'):
    """Output Later"""

    def output_layer(x_in):
        x_input = inputs = Input(x_in.shape[1:])
        new_emb = []
        for i in range(cfg['m']):
            x = Dense(cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
                      activation='tanh')(x_input)  # extra connection layer
            # x = Dense(cfg['q'],kernel_initializer=tf.keras.initializers.GlorotUniform())(x_input)  # extra connection layer
            # x = Dense(cfg['q'],kernel_initializer=tf.keras.initializers.GlorotNormal())(x_input)  # extra connection layer
            new_emb.append(x)

        hashcode = concatenate(new_emb)
        return Model(inputs, hashcode, name=name)(x_in)

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

    if backbone_type == 'Insight_ResNet100' or backbone_type == 'Insight_ResNet50':  # here use the pretrained model build by Insightface team, we don't need the output layer anymore
        embds = x
    else:
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


def IoMFaceModelFromArFace(size=None, channels=3, arcmodel=None, name='IoMface_model',
                           margin=0.5, logist_scale=64, embd_shape=512,
                           head_type='ArcHead', backbone_type='ResNet50',
                           w_decay=5e-4, use_pretrain=True, training=False, permKey=None, cfg=None):
    """IoMFaceModelFromArFace Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = arcmodel(x)
    # x = Dropout(0.2)(x)# 2020 07 09 add by xingbo
    if not (permKey is None):
        x = PermLayer(permKey)(x)  # permutation before project to IoM hash code
    # here I add one extra hidden layer
    # x = Dense(1024, kernel_regularizer=_regularizer(w_decay))(x)
    # x = IoMProjectionLayer(cfg)(x)
    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
              name="IoMProjection")(x)  # extra connection layer
    logist = IoMHead(m=cfg['m'], q=cfg['q'], isTraining=training)(x)  # loss need to change
    return Model(inputs, logist, name=name)


def IoMFaceModelFrom2ArcFace(size=None, channels=3, arcmodel=None, arcmodel2=None, name='IoMface_model',
                           margin=0.5, logist_scale=64, embd_shape=512,
                           head_type='ArcHead', backbone_type='ResNet50',
                           w_decay=5e-4, use_pretrain=True, training=False, permKey=None, cfg=None):
    """IoMFaceModelFromArFace Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x1 = arcmodel(x)
    x2 = arcmodel2(x)
    # x = Dropout(0.2)(x)# 2020 07 09 add by xingbo
    if not (permKey is None):
        x1 = PermLayer(permKey)(x1)  # permutation before project to IoM hash code
        x2 = PermLayer(permKey)(x2)  # permutation before project to IoM hash code
    # here I add one extra hidden layer
    # x = Dense(1024, kernel_regularizer=_regularizer(w_decay))(x)
    # x = IoMProjectionLayer(cfg)(x)

    x = concatenate([x1, x2])

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
    x = Dense(512, kernel_regularizer=_regularizer(w_decay))(x)  # , activation='relu'
    # x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(512, kernel_regularizer=_regularizer(w_decay))(x)  # , activation='relu'
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
    x = Dense(1024, kernel_regularizer=_regularizer(w_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    x = Flatten()(x)
    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
              name="IoMProjection")(x)  # extra connection layer
    logist = IoMHead(m=cfg['m'], q=cfg['q'], T=cfg['T'], isTraining=training)(x)  # loss need to change
    return Model(inputs, logist, name=name)


def IoMFaceModelFromArFace_T1(size=None, channels=3, arcmodel=None, name='IoMface_model',
                              margin=0.5, logist_scale=64, embd_shape=512,
                              head_type='ArcHead', backbone_type='ResNet50',
                              w_decay=5e-4, use_pretrain=True, training=False, permKey=None, cfg=None):
    """IoMFaceModelFromArFace Model"""
    x = inputs = Input([size, size, channels], name='input_image')
    x = arcmodel(x)
    if not (permKey is None):
        x = PermLayer(permKey)(x)  # permutation before project to IoM hash code
        # here I add one extra hidden layer
    x = Dense(cfg['m'] * cfg['q'], kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),
              name="IoMProjection")(x)  # extra connection layer
    logist = IoMHead(m=cfg['m'], q=cfg['q'], T=cfg['T'], isTraining=training)(x)  # loss need to change
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

def build_or_load_IoMmodel(arc_cfg=None, ckpt_epoch = '', is_only_arc=False):
    permKey = None
    if arc_cfg['head_type'] == 'IoMHead':  #
        # permKey = generatePermKey(cfg['embd_shape'])
        permKey = tf.eye(arc_cfg['embd_shape'])  # for training, we don't permutate, won't influence the performance

    arcmodel = ArcFaceModel(size=arc_cfg['input_size'],
                            embd_shape=arc_cfg['embd_shape'],
                            backbone_type=arc_cfg['backbone_type'],
                            head_type='ArcHead',
                            training=False,
                            cfg=arc_cfg)

    if is_only_arc:
        ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + arc_cfg['sub_name'])
        if ckpt_path is not None:
            print("[*] load ckpt from {}".format(ckpt_path))
            arcmodel.load_weights(ckpt_path)
        else:
            print("[*] Cannot find ckpt from {}.".format(ckpt_path), ', this means you will give a IoM weight later.')
        return arcmodel

    iom_cfg = arc_cfg
    # here I add the extra IoM layer and head
    if iom_cfg['hidden_layer_remark'] == '1':
        model = IoMFaceModelFromArFace(size=iom_cfg['input_size'],
                                       arcmodel=arcmodel, training=False,
                                       permKey=permKey, cfg=iom_cfg)
    elif iom_cfg['hidden_layer_remark'] == '2':
        model = IoMFaceModelFromArFace2(size=iom_cfg['input_size'],
                                        arcmodel=arcmodel, training=False,
                                        permKey=permKey, cfg=iom_cfg)
    elif iom_cfg['hidden_layer_remark'] == '3':
        model = IoMFaceModelFromArFace3(size=iom_cfg['input_size'],
                                        arcmodel=arcmodel, training=False,
                                        permKey=permKey, cfg=iom_cfg)
    elif iom_cfg['hidden_layer_remark'] == 'T':  # 2 layers
        model = IoMFaceModelFromArFace_T(size=iom_cfg['input_size'],
                                         arcmodel=arcmodel, training=False,
                                         permKey=permKey, cfg=iom_cfg)
    elif iom_cfg['hidden_layer_remark'] == 'T1':
        model = IoMFaceModelFromArFace_T1(size=iom_cfg['input_size'],
                                          arcmodel=arcmodel, training=False,
                                          permKey=permKey, cfg=iom_cfg)
    else:
        model = IoMFaceModelFromArFace(size=iom_cfg['input_size'],
                                       arcmodel=arcmodel, training=False,
                                       permKey=permKey, cfg=iom_cfg)
    if ckpt_epoch == '':
        ckpt_path_iom = tf.train.latest_checkpoint('./checkpoints/' + iom_cfg['sub_name'])
    else:
        ckpt_path_iom = './checkpoints/' + iom_cfg['sub_name'] + '/' + ckpt_epoch

    if ckpt_path_iom is not None:
        print("[*] load ckpt from {}".format(ckpt_path_iom))
        model.load_weights(ckpt_path_iom)
    else:
        print("[*] Warning!!!! Cannot find ckpt from {}.".format(ckpt_path_iom),'Using random layer')


    return model


def build_or_load_Random_IoMmodel(arc_cfg=None):
    permKey = None
    if arc_cfg['head_type'] == 'IoMHead':  #
        # permKey = generatePermKey(cfg['embd_shape'])
        permKey = tf.eye(arc_cfg['embd_shape'])  # for training, we don't permutate, won't influence the performance

    arcmodel = ArcFaceModel(size=arc_cfg['input_size'],
                            embd_shape=arc_cfg['embd_shape'],
                            backbone_type=arc_cfg['backbone_type'],
                            head_type='ArcHead',
                            training=False,
                            cfg=arc_cfg)
    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + arc_cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcmodel.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path), ', this means you will give a IoM weight later.')
    iom_cfg = arc_cfg
    # here I add the extra IoM layer and head
    if iom_cfg['hidden_layer_remark'] == '1':
        model = IoMFaceModelFromArFace(size=iom_cfg['input_size'],
                                       arcmodel=arcmodel, training=False,
                                       permKey=permKey, cfg=iom_cfg)
    elif iom_cfg['hidden_layer_remark'] == '2':
        model = IoMFaceModelFromArFace2(size=iom_cfg['input_size'],
                                        arcmodel=arcmodel, training=False,
                                        permKey=permKey, cfg=iom_cfg)
    elif iom_cfg['hidden_layer_remark'] == '3':
        model = IoMFaceModelFromArFace3(size=iom_cfg['input_size'],
                                        arcmodel=arcmodel, training=False,
                                        permKey=permKey, cfg=iom_cfg)
    elif iom_cfg['hidden_layer_remark'] == 'T':  # 2 layers
        model = IoMFaceModelFromArFace_T(size=iom_cfg['input_size'],
                                         arcmodel=arcmodel, training=False,
                                         permKey=permKey, cfg=iom_cfg)
    elif iom_cfg['hidden_layer_remark'] == 'T1':
        model = IoMFaceModelFromArFace_T1(size=iom_cfg['input_size'],
                                          arcmodel=arcmodel, training=False,
                                          permKey=permKey, cfg=iom_cfg)
    else:
        model = IoMFaceModelFromArFace(size=iom_cfg['input_size'],
                                       arcmodel=arcmodel, training=False,
                                       permKey=permKey, cfg=iom_cfg)

    return model

