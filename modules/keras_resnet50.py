import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Lambda
import tensorflow as tf


weights_dict = dict()
def load_weights_from_file(weight_file):
    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


def set_layer_weights(model, weights_dict):
    for layer in model.layers:
        if layer.name in weights_dict:
            cur_dict = weights_dict[layer.name]
            current_layer_parameters = list()
            if layer.__class__.__name__ == "BatchNormalization":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
                current_layer_parameters.extend([cur_dict['mean'], cur_dict['var']])
            elif layer.__class__.__name__ == "Scale":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            elif layer.__class__.__name__ == "SeparableConv2D":
                current_layer_parameters = [cur_dict['depthwise_filter'], cur_dict['pointwise_filter']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            elif layer.__class__.__name__ == "Embedding":
                current_layer_parameters.append(cur_dict['weights'])
            elif layer.__class__.__name__ == "PReLU":
                gamma =  np.ones(list(layer.input_shape[1:]))*cur_dict['gamma']
                current_layer_parameters.append(gamma)
            else:
                # rot
                if 'weights' in cur_dict:
                    current_layer_parameters = [cur_dict['weights']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            # print(layer.name,current_layer_parameters)
            model.get_layer(layer.name).set_weights(current_layer_parameters)

    return model


def KitModel_50(weight_file = None):
    global weights_dict
    weights_dict = load_weights_from_file(weight_file) if not weight_file == None else None

    minusscalar0_second = tf.constant(weights_dict['minusscalar0_second']['value'], dtype=tf.float32, name='minusscalar0_second')
    mulscalar0_second = tf.constant(weights_dict['mulscalar0_second']['value'], dtype=tf.float32, name='mulscalar0_second')

    # print(minusscalar0_second.numpy())
    # print(mulscalar0_second.numpy())
    data            = layers.Input(name = 'data', shape = (112, 112, 3,) )
    # minusscalar0    = my_sub()([data, minusscalar0_second])
    # mulscalar0      = mul_constant(weight_factor=mulscalar0_second, layer_name= minusscalar0)# not understand

    # minusscalar0    = data - minusscalar0_second
    # mulscalar0 = minusscalar0 * mulscalar0_second # here manually add

    conv0_input     = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(data)
    conv0           = convolution(weights_dict, name='conv0', input=conv0_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    bn0             = layers.BatchNormalization(name = 'bn0', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(conv0)
    relu0           = layers.PReLU(name='relu0')(bn0)
    stage1_unit1_bn1 = layers.BatchNormalization(name = 'stage1_unit1_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(relu0)
    stage1_unit1_conv1sc = convolution(weights_dict, name='stage1_unit1_conv1sc', input=relu0, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(1, 1), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage1_unit1_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage1_unit1_bn1)
    stage1_unit1_conv1 = convolution(weights_dict, name='stage1_unit1_conv1', input=stage1_unit1_conv1_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage1_unit1_sc = layers.BatchNormalization(name = 'stage1_unit1_sc', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage1_unit1_conv1sc)
    stage1_unit1_bn2 = layers.BatchNormalization(name = 'stage1_unit1_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage1_unit1_conv1)
    stage1_unit1_relu1 = layers.PReLU(name='stage1_unit1_relu1')(stage1_unit1_bn2)
    stage1_unit1_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage1_unit1_relu1)
    stage1_unit1_conv2 = convolution(weights_dict, name='stage1_unit1_conv2', input=stage1_unit1_conv2_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage1_unit1_bn3 = layers.BatchNormalization(name = 'stage1_unit1_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage1_unit1_conv2)
    plus0           = my_add()([stage1_unit1_bn3, stage1_unit1_sc])
    stage1_unit2_bn1 = layers.BatchNormalization(name = 'stage1_unit2_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus0)
    stage1_unit2_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage1_unit2_bn1)
    stage1_unit2_conv1 = convolution(weights_dict, name='stage1_unit2_conv1', input=stage1_unit2_conv1_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage1_unit2_bn2 = layers.BatchNormalization(name = 'stage1_unit2_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage1_unit2_conv1)
    stage1_unit2_relu1 = layers.PReLU(name='stage1_unit2_relu1')(stage1_unit2_bn2)
    stage1_unit2_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage1_unit2_relu1)
    stage1_unit2_conv2 = convolution(weights_dict, name='stage1_unit2_conv2', input=stage1_unit2_conv2_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage1_unit2_bn3 = layers.BatchNormalization(name = 'stage1_unit2_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage1_unit2_conv2)
    plus1           = my_add()([stage1_unit2_bn3, plus0])
    stage1_unit3_bn1 = layers.BatchNormalization(name = 'stage1_unit3_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus1)
    stage1_unit3_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage1_unit3_bn1)
    stage1_unit3_conv1 = convolution(weights_dict, name='stage1_unit3_conv1', input=stage1_unit3_conv1_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage1_unit3_bn2 = layers.BatchNormalization(name = 'stage1_unit3_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage1_unit3_conv1)
    stage1_unit3_relu1 = layers.PReLU(name='stage1_unit3_relu1')(stage1_unit3_bn2)
    stage1_unit3_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage1_unit3_relu1)
    stage1_unit3_conv2 = convolution(weights_dict, name='stage1_unit3_conv2', input=stage1_unit3_conv2_input, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage1_unit3_bn3 = layers.BatchNormalization(name = 'stage1_unit3_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage1_unit3_conv2)
    plus2           = my_add()([stage1_unit3_bn3, plus1])
    stage2_unit1_bn1 = layers.BatchNormalization(name = 'stage2_unit1_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus2)
    stage2_unit1_conv1sc = convolution(weights_dict, name='stage2_unit1_conv1sc', input=plus2, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(1, 1), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage2_unit1_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage2_unit1_bn1)
    stage2_unit1_conv1 = convolution(weights_dict, name='stage2_unit1_conv1', input=stage2_unit1_conv1_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage2_unit1_sc = layers.BatchNormalization(name = 'stage2_unit1_sc', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage2_unit1_conv1sc)
    stage2_unit1_bn2 = layers.BatchNormalization(name = 'stage2_unit1_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage2_unit1_conv1)
    stage2_unit1_relu1 = layers.PReLU(name='stage2_unit1_relu1')(stage2_unit1_bn2)
    stage2_unit1_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage2_unit1_relu1)
    stage2_unit1_conv2 = convolution(weights_dict, name='stage2_unit1_conv2', input=stage2_unit1_conv2_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage2_unit1_bn3 = layers.BatchNormalization(name = 'stage2_unit1_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage2_unit1_conv2)
    plus3           = my_add()([stage2_unit1_bn3, stage2_unit1_sc])
    stage2_unit2_bn1 = layers.BatchNormalization(name = 'stage2_unit2_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus3)
    stage2_unit2_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage2_unit2_bn1)
    stage2_unit2_conv1 = convolution(weights_dict, name='stage2_unit2_conv1', input=stage2_unit2_conv1_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage2_unit2_bn2 = layers.BatchNormalization(name = 'stage2_unit2_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage2_unit2_conv1)
    stage2_unit2_relu1 = layers.PReLU(name='stage2_unit2_relu1')(stage2_unit2_bn2)
    stage2_unit2_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage2_unit2_relu1)
    stage2_unit2_conv2 = convolution(weights_dict, name='stage2_unit2_conv2', input=stage2_unit2_conv2_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage2_unit2_bn3 = layers.BatchNormalization(name = 'stage2_unit2_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage2_unit2_conv2)
    plus4           = my_add()([stage2_unit2_bn3, plus3])
    stage2_unit3_bn1 = layers.BatchNormalization(name = 'stage2_unit3_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus4)
    stage2_unit3_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage2_unit3_bn1)
    stage2_unit3_conv1 = convolution(weights_dict, name='stage2_unit3_conv1', input=stage2_unit3_conv1_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage2_unit3_bn2 = layers.BatchNormalization(name = 'stage2_unit3_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage2_unit3_conv1)
    stage2_unit3_relu1 = layers.PReLU(name='stage2_unit3_relu1')(stage2_unit3_bn2)
    stage2_unit3_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage2_unit3_relu1)
    stage2_unit3_conv2 = convolution(weights_dict, name='stage2_unit3_conv2', input=stage2_unit3_conv2_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage2_unit3_bn3 = layers.BatchNormalization(name = 'stage2_unit3_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage2_unit3_conv2)
    plus5           = my_add()([stage2_unit3_bn3, plus4])
    stage2_unit4_bn1 = layers.BatchNormalization(name = 'stage2_unit4_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus5)
    stage2_unit4_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage2_unit4_bn1)
    stage2_unit4_conv1 = convolution(weights_dict, name='stage2_unit4_conv1', input=stage2_unit4_conv1_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage2_unit4_bn2 = layers.BatchNormalization(name = 'stage2_unit4_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage2_unit4_conv1)
    stage2_unit4_relu1 = layers.PReLU(name='stage2_unit4_relu1')(stage2_unit4_bn2)
    stage2_unit4_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage2_unit4_relu1)
    stage2_unit4_conv2 = convolution(weights_dict, name='stage2_unit4_conv2', input=stage2_unit4_conv2_input, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage2_unit4_bn3 = layers.BatchNormalization(name = 'stage2_unit4_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage2_unit4_conv2)
    plus6           = my_add()([stage2_unit4_bn3, plus5])
    stage3_unit1_bn1 = layers.BatchNormalization(name = 'stage3_unit1_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus6)
    stage3_unit1_conv1sc = convolution(weights_dict, name='stage3_unit1_conv1sc', input=plus6, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(1, 1), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit1_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit1_bn1)
    stage3_unit1_conv1 = convolution(weights_dict, name='stage3_unit1_conv1', input=stage3_unit1_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit1_sc = layers.BatchNormalization(name = 'stage3_unit1_sc', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit1_conv1sc)
    stage3_unit1_bn2 = layers.BatchNormalization(name = 'stage3_unit1_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit1_conv1)
    stage3_unit1_relu1 = layers.PReLU(name='stage3_unit1_relu1')(stage3_unit1_bn2)
    stage3_unit1_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit1_relu1)
    stage3_unit1_conv2 = convolution(weights_dict, name='stage3_unit1_conv2', input=stage3_unit1_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit1_bn3 = layers.BatchNormalization(name = 'stage3_unit1_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit1_conv2)
    plus7           = my_add()([stage3_unit1_bn3, stage3_unit1_sc])
    stage3_unit2_bn1 = layers.BatchNormalization(name = 'stage3_unit2_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus7)
    stage3_unit2_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit2_bn1)
    stage3_unit2_conv1 = convolution(weights_dict, name='stage3_unit2_conv1', input=stage3_unit2_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit2_bn2 = layers.BatchNormalization(name = 'stage3_unit2_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit2_conv1)
    stage3_unit2_relu1 = layers.PReLU(name='stage3_unit2_relu1')(stage3_unit2_bn2)
    stage3_unit2_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit2_relu1)
    stage3_unit2_conv2 = convolution(weights_dict, name='stage3_unit2_conv2', input=stage3_unit2_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit2_bn3 = layers.BatchNormalization(name = 'stage3_unit2_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit2_conv2)
    plus8           = my_add()([stage3_unit2_bn3, plus7])
    stage3_unit3_bn1 = layers.BatchNormalization(name = 'stage3_unit3_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus8)
    stage3_unit3_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit3_bn1)
    stage3_unit3_conv1 = convolution(weights_dict, name='stage3_unit3_conv1', input=stage3_unit3_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit3_bn2 = layers.BatchNormalization(name = 'stage3_unit3_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit3_conv1)
    stage3_unit3_relu1 = layers.PReLU(name='stage3_unit3_relu1')(stage3_unit3_bn2)
    stage3_unit3_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit3_relu1)
    stage3_unit3_conv2 = convolution(weights_dict, name='stage3_unit3_conv2', input=stage3_unit3_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit3_bn3 = layers.BatchNormalization(name = 'stage3_unit3_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit3_conv2)
    plus9           = my_add()([stage3_unit3_bn3, plus8])
    stage3_unit4_bn1 = layers.BatchNormalization(name = 'stage3_unit4_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus9)
    stage3_unit4_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit4_bn1)
    stage3_unit4_conv1 = convolution(weights_dict, name='stage3_unit4_conv1', input=stage3_unit4_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit4_bn2 = layers.BatchNormalization(name = 'stage3_unit4_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit4_conv1)
    stage3_unit4_relu1 = layers.PReLU(name='stage3_unit4_relu1')(stage3_unit4_bn2)
    stage3_unit4_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit4_relu1)
    stage3_unit4_conv2 = convolution(weights_dict, name='stage3_unit4_conv2', input=stage3_unit4_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit4_bn3 = layers.BatchNormalization(name = 'stage3_unit4_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit4_conv2)
    plus10          = my_add()([stage3_unit4_bn3, plus9])
    stage3_unit5_bn1 = layers.BatchNormalization(name = 'stage3_unit5_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus10)
    stage3_unit5_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit5_bn1)
    stage3_unit5_conv1 = convolution(weights_dict, name='stage3_unit5_conv1', input=stage3_unit5_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit5_bn2 = layers.BatchNormalization(name = 'stage3_unit5_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit5_conv1)
    stage3_unit5_relu1 = layers.PReLU(name='stage3_unit5_relu1')(stage3_unit5_bn2)
    stage3_unit5_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit5_relu1)
    stage3_unit5_conv2 = convolution(weights_dict, name='stage3_unit5_conv2', input=stage3_unit5_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit5_bn3 = layers.BatchNormalization(name = 'stage3_unit5_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit5_conv2)
    plus11          = my_add()([stage3_unit5_bn3, plus10])
    stage3_unit6_bn1 = layers.BatchNormalization(name = 'stage3_unit6_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus11)
    stage3_unit6_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit6_bn1)
    stage3_unit6_conv1 = convolution(weights_dict, name='stage3_unit6_conv1', input=stage3_unit6_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit6_bn2 = layers.BatchNormalization(name = 'stage3_unit6_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit6_conv1)
    stage3_unit6_relu1 = layers.PReLU(name='stage3_unit6_relu1')(stage3_unit6_bn2)
    stage3_unit6_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit6_relu1)
    stage3_unit6_conv2 = convolution(weights_dict, name='stage3_unit6_conv2', input=stage3_unit6_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit6_bn3 = layers.BatchNormalization(name = 'stage3_unit6_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit6_conv2)
    plus12          = my_add()([stage3_unit6_bn3, plus11])
    stage3_unit7_bn1 = layers.BatchNormalization(name = 'stage3_unit7_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus12)
    stage3_unit7_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit7_bn1)
    stage3_unit7_conv1 = convolution(weights_dict, name='stage3_unit7_conv1', input=stage3_unit7_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit7_bn2 = layers.BatchNormalization(name = 'stage3_unit7_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit7_conv1)
    stage3_unit7_relu1 = layers.PReLU(name='stage3_unit7_relu1')(stage3_unit7_bn2)
    stage3_unit7_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit7_relu1)
    stage3_unit7_conv2 = convolution(weights_dict, name='stage3_unit7_conv2', input=stage3_unit7_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit7_bn3 = layers.BatchNormalization(name = 'stage3_unit7_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit7_conv2)
    plus13          = my_add()([stage3_unit7_bn3, plus12])
    stage3_unit8_bn1 = layers.BatchNormalization(name = 'stage3_unit8_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus13)
    stage3_unit8_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit8_bn1)
    stage3_unit8_conv1 = convolution(weights_dict, name='stage3_unit8_conv1', input=stage3_unit8_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit8_bn2 = layers.BatchNormalization(name = 'stage3_unit8_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit8_conv1)
    stage3_unit8_relu1 = layers.PReLU(name='stage3_unit8_relu1')(stage3_unit8_bn2)
    stage3_unit8_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit8_relu1)
    stage3_unit8_conv2 = convolution(weights_dict, name='stage3_unit8_conv2', input=stage3_unit8_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit8_bn3 = layers.BatchNormalization(name = 'stage3_unit8_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit8_conv2)
    plus14          = my_add()([stage3_unit8_bn3, plus13])
    stage3_unit9_bn1 = layers.BatchNormalization(name = 'stage3_unit9_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus14)
    stage3_unit9_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit9_bn1)
    stage3_unit9_conv1 = convolution(weights_dict, name='stage3_unit9_conv1', input=stage3_unit9_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit9_bn2 = layers.BatchNormalization(name = 'stage3_unit9_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit9_conv1)
    stage3_unit9_relu1 = layers.PReLU(name='stage3_unit9_relu1')(stage3_unit9_bn2)
    stage3_unit9_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit9_relu1)
    stage3_unit9_conv2 = convolution(weights_dict, name='stage3_unit9_conv2', input=stage3_unit9_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit9_bn3 = layers.BatchNormalization(name = 'stage3_unit9_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit9_conv2)
    plus15          = my_add()([stage3_unit9_bn3, plus14])
    stage3_unit10_bn1 = layers.BatchNormalization(name = 'stage3_unit10_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus15)
    stage3_unit10_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit10_bn1)
    stage3_unit10_conv1 = convolution(weights_dict, name='stage3_unit10_conv1', input=stage3_unit10_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit10_bn2 = layers.BatchNormalization(name = 'stage3_unit10_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit10_conv1)
    stage3_unit10_relu1 = layers.PReLU(name='stage3_unit10_relu1')(stage3_unit10_bn2)
    stage3_unit10_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit10_relu1)
    stage3_unit10_conv2 = convolution(weights_dict, name='stage3_unit10_conv2', input=stage3_unit10_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit10_bn3 = layers.BatchNormalization(name = 'stage3_unit10_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit10_conv2)
    plus16          = my_add()([stage3_unit10_bn3, plus15])
    stage3_unit11_bn1 = layers.BatchNormalization(name = 'stage3_unit11_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus16)
    stage3_unit11_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit11_bn1)
    stage3_unit11_conv1 = convolution(weights_dict, name='stage3_unit11_conv1', input=stage3_unit11_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit11_bn2 = layers.BatchNormalization(name = 'stage3_unit11_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit11_conv1)
    stage3_unit11_relu1 = layers.PReLU(name='stage3_unit11_relu1')(stage3_unit11_bn2)
    stage3_unit11_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit11_relu1)
    stage3_unit11_conv2 = convolution(weights_dict, name='stage3_unit11_conv2', input=stage3_unit11_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit11_bn3 = layers.BatchNormalization(name = 'stage3_unit11_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit11_conv2)
    plus17          = my_add()([stage3_unit11_bn3, plus16])
    stage3_unit12_bn1 = layers.BatchNormalization(name = 'stage3_unit12_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus17)
    stage3_unit12_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit12_bn1)
    stage3_unit12_conv1 = convolution(weights_dict, name='stage3_unit12_conv1', input=stage3_unit12_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit12_bn2 = layers.BatchNormalization(name = 'stage3_unit12_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit12_conv1)
    stage3_unit12_relu1 = layers.PReLU(name='stage3_unit12_relu1')(stage3_unit12_bn2)
    stage3_unit12_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit12_relu1)
    stage3_unit12_conv2 = convolution(weights_dict, name='stage3_unit12_conv2', input=stage3_unit12_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit12_bn3 = layers.BatchNormalization(name = 'stage3_unit12_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit12_conv2)
    plus18          = my_add()([stage3_unit12_bn3, plus17])
    stage3_unit13_bn1 = layers.BatchNormalization(name = 'stage3_unit13_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus18)
    stage3_unit13_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit13_bn1)
    stage3_unit13_conv1 = convolution(weights_dict, name='stage3_unit13_conv1', input=stage3_unit13_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit13_bn2 = layers.BatchNormalization(name = 'stage3_unit13_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit13_conv1)
    stage3_unit13_relu1 = layers.PReLU(name='stage3_unit13_relu1')(stage3_unit13_bn2)
    stage3_unit13_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit13_relu1)
    stage3_unit13_conv2 = convolution(weights_dict, name='stage3_unit13_conv2', input=stage3_unit13_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit13_bn3 = layers.BatchNormalization(name = 'stage3_unit13_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit13_conv2)
    plus19          = my_add()([stage3_unit13_bn3, plus18])
    stage3_unit14_bn1 = layers.BatchNormalization(name = 'stage3_unit14_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus19)
    stage3_unit14_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit14_bn1)
    stage3_unit14_conv1 = convolution(weights_dict, name='stage3_unit14_conv1', input=stage3_unit14_conv1_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit14_bn2 = layers.BatchNormalization(name = 'stage3_unit14_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit14_conv1)
    stage3_unit14_relu1 = layers.PReLU(name='stage3_unit14_relu1')(stage3_unit14_bn2)
    stage3_unit14_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage3_unit14_relu1)
    stage3_unit14_conv2 = convolution(weights_dict, name='stage3_unit14_conv2', input=stage3_unit14_conv2_input, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage3_unit14_bn3 = layers.BatchNormalization(name = 'stage3_unit14_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage3_unit14_conv2)
    plus20          = my_add()([stage3_unit14_bn3, plus19])
    stage4_unit1_bn1 = layers.BatchNormalization(name = 'stage4_unit1_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus20)
    stage4_unit1_conv1sc = convolution(weights_dict, name='stage4_unit1_conv1sc', input=plus20, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(1, 1), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage4_unit1_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage4_unit1_bn1)
    stage4_unit1_conv1 = convolution(weights_dict, name='stage4_unit1_conv1', input=stage4_unit1_conv1_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage4_unit1_sc = layers.BatchNormalization(name = 'stage4_unit1_sc', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage4_unit1_conv1sc)
    stage4_unit1_bn2 = layers.BatchNormalization(name = 'stage4_unit1_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage4_unit1_conv1)
    stage4_unit1_relu1 = layers.PReLU(name='stage4_unit1_relu1')(stage4_unit1_bn2)
    stage4_unit1_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage4_unit1_relu1)
    stage4_unit1_conv2 = convolution(weights_dict, name='stage4_unit1_conv2', input=stage4_unit1_conv2_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(2, 2), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage4_unit1_bn3 = layers.BatchNormalization(name = 'stage4_unit1_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage4_unit1_conv2)
    plus21          = my_add()([stage4_unit1_bn3, stage4_unit1_sc])
    stage4_unit2_bn1 = layers.BatchNormalization(name = 'stage4_unit2_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus21)
    stage4_unit2_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage4_unit2_bn1)
    stage4_unit2_conv1 = convolution(weights_dict, name='stage4_unit2_conv1', input=stage4_unit2_conv1_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage4_unit2_bn2 = layers.BatchNormalization(name = 'stage4_unit2_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage4_unit2_conv1)
    stage4_unit2_relu1 = layers.PReLU(name='stage4_unit2_relu1')(stage4_unit2_bn2)
    stage4_unit2_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage4_unit2_relu1)
    stage4_unit2_conv2 = convolution(weights_dict, name='stage4_unit2_conv2', input=stage4_unit2_conv2_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage4_unit2_bn3 = layers.BatchNormalization(name = 'stage4_unit2_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage4_unit2_conv2)
    plus22          = my_add()([stage4_unit2_bn3, plus21])
    stage4_unit3_bn1 = layers.BatchNormalization(name = 'stage4_unit3_bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus22)
    stage4_unit3_conv1_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage4_unit3_bn1)
    stage4_unit3_conv1 = convolution(weights_dict, name='stage4_unit3_conv1', input=stage4_unit3_conv1_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage4_unit3_bn2 = layers.BatchNormalization(name = 'stage4_unit3_bn2', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage4_unit3_conv1)
    stage4_unit3_relu1 = layers.PReLU(name='stage4_unit3_relu1')(stage4_unit3_bn2)
    stage4_unit3_conv2_input = layers.ZeroPadding2D(padding = ((1, 1), (1, 1)))(stage4_unit3_relu1)
    stage4_unit3_conv2 = convolution(weights_dict, name='stage4_unit3_conv2', input=stage4_unit3_conv2_input, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='valid', use_bias=False)
    stage4_unit3_bn3 = layers.BatchNormalization(name = 'stage4_unit3_bn3', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(stage4_unit3_conv2)
    plus23          = my_add()([stage4_unit3_bn3, plus22])
    bn1             = layers.BatchNormalization(name = 'bn1', axis = 3, epsilon = 1.9999999494757503e-05, center = True, scale = True)(plus23)
    pre_fc1_flatten= layers.Flatten()(bn1)# need to add manually
    dropout0        = layers.Dropout(name = 'dropout0', rate = 0.4000000059604645, seed = None)(pre_fc1_flatten)
    pre_fc1         = layers.Dense(name = 'pre_fc1', units = 512, use_bias = True)(dropout0)
    fc1             = layers.BatchNormalization(name = 'fc1', axis = 1, epsilon = 1.9999999494757503e-05, center = True, scale = False)(pre_fc1)
    model           = Model(inputs = [data], outputs = [fc1])
    set_layer_weights(model, weights_dict)
    return model

def mul_constant(weight_factor, layer_name):
    weight = Lambda(lambda x: x*weight_factor)
    weight(layer_name)
    return weight.output


class my_add(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(my_add, self).__init__(**kwargs)
    def call(self, inputs):
        res = inputs[0] + inputs[1]
        self.output_shapes = K.int_shape(res)
        return res
    
    def compute_output_shape(self, input_shape):
        return self.output_shapes


def convolution(weights_dict, name, input, group, conv_type, filters=None, **kwargs):
    if not conv_type.startswith('layer'):
        layer = tensorflow.keras.applications.mobilenet.DepthwiseConv2D(name=name, **kwargs)(input)
        return layer
    elif conv_type == 'layers.DepthwiseConv2D':
        layer = layers.DepthwiseConv2D(name=name, **kwargs)(input)
        return layer
    
    inp_filters = K.int_shape(input)[-1]
    inp_grouped_channels = int(inp_filters / group)
    out_grouped_channels = int(filters / group)
    group_list = []
    if group == 1:
        func = getattr(layers, conv_type.split('.')[-1])
        layer = func(name = name, filters = filters, **kwargs)(input)
        return layer
    weight_groups = list()
    if not weights_dict == None:
        w = np.array(weights_dict[name]['weights'])
        weight_groups = np.split(w, indices_or_sections=group, axis=-1)
    for c in range(group):
        x = layers.Lambda(lambda z: z[..., c * inp_grouped_channels:(c + 1) * inp_grouped_channels])(input)
        x = layers.Conv2D(name=name + "_" + str(c), filters=out_grouped_channels, **kwargs)(x)
        weights_dict[name + "_" + str(c)] = dict()
        weights_dict[name + "_" + str(c)]['weights'] = weight_groups[c]
        group_list.append(x)
    layer = layers.concatenate(group_list, axis = -1)
    if 'bias' in weights_dict[name]:
        b = K.variable(weights_dict[name]['bias'], name = name + "_bias")
        layer = layer + b
    return layer

class my_sub(tensorflow.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(my_sub, self).__init__(**kwargs)
    def call(self, inputs):
        res = inputs[0] - inputs[1]
        self.output_shapes = K.int_shape(res)
        return res
    
    def compute_output_shape(self, input_shape):
        return self.output_shapes


