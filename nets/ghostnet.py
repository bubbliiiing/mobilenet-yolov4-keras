import math
import warnings

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.initializers import random_normal
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, 
                          Lambda, Multiply, Reshape)


def slices(dw, n):
    return dw[:,:,:,:n]

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
    
def _squeeze(inputs, hidden_channel, ratio, block_id, sub_block_id):
    x = GlobalAveragePooling2D()(inputs)
    x = Reshape((1,1,-1))(x)

    x = Conv2D(_make_divisible(hidden_channel/ratio, 4), (1,1), strides=(1,1), padding='same', kernel_initializer=random_normal(stddev=0.02),
            name="blocks."+str(block_id)+"."+str(sub_block_id)+".se.conv_reduce")(x)
    x = Activation('relu')(x)

    x = Conv2D(hidden_channel, (1,1),strides=(1,1), padding='same', kernel_initializer=random_normal(stddev=0.02),
            name="blocks."+str(block_id)+"."+str(sub_block_id)+".se.conv_expand")(x)
    x = Activation('hard_sigmoid')(x)

    x = Multiply()([inputs, x])    # inputs和x逐元素相乘
    return x

def _ghost_module(inputs, exp, ratio, block_id, sub_block_id, part, kernel_size=1, dw_size=3, stride=1, relu=True):
    output_channels = math.ceil(exp * 1.0 / ratio)

    x = Conv2D(output_channels, kernel_size, strides=stride, padding="same", use_bias=False, kernel_initializer=random_normal(stddev=0.02),
            name="blocks."+str(block_id)+"."+str(sub_block_id)+".ghost"+str(part)+".primary_conv.0")(inputs)
    x = BatchNormalization(name="blocks."+str(block_id)+"."+str(sub_block_id)+".ghost"+str(part)+".primary_conv.1")(x)
    if relu:
        x = Activation('relu')(x)

    dw = DepthwiseConv2D(dw_size, 1, padding="same", depth_multiplier=ratio-1, use_bias=False, depthwise_initializer=random_normal(stddev=0.02), 
            name="blocks."+str(block_id)+"."+str(sub_block_id)+".ghost"+str(part)+".cheap_operation.0")(x)
    dw = BatchNormalization(name="blocks."+str(block_id)+"."+str(sub_block_id)+".ghost"+str(part)+".cheap_operation.1")(dw)
    if relu:
        dw = Activation('relu')(dw)

    x = Concatenate(axis=-1)([x,dw])
    x = Lambda(slices, arguments={'n':exp})(x)
    return x

def _ghost_bottleneck(inputs, output_channel, hidden_channel, kernel, strides, ratio, squeeze, block_id, sub_block_id):
    input_shape = K.int_shape(inputs)

    x = _ghost_module(inputs, hidden_channel, ratio, block_id, sub_block_id, 1)

    if strides > 1:
        x = DepthwiseConv2D(kernel, strides, padding='same', depth_multiplier=1, use_bias=False, depthwise_initializer=random_normal(stddev=0.02),
                name="blocks."+str(block_id)+"."+str(sub_block_id)+".conv_dw")(x)
        x = BatchNormalization(name="blocks."+str(block_id)+"."+str(sub_block_id)+".bn_dw")(x)

    if squeeze:
        x = _squeeze(x, hidden_channel, 4, block_id, sub_block_id)

    x = _ghost_module(x, output_channel, ratio, block_id, sub_block_id, 2, relu=False)

    if strides == 1 and input_shape[-1] == output_channel:
        res = inputs
    else:
        res = DepthwiseConv2D(kernel, strides=strides, padding='same', depth_multiplier=1, use_bias=False, depthwise_initializer=random_normal(stddev=0.02),
                name="blocks."+str(block_id)+"."+str(sub_block_id)+".shortcut.0")(inputs)
        res = BatchNormalization(name="blocks."+str(block_id)+"."+str(sub_block_id)+".shortcut.1")(res)
        res = Conv2D(output_channel, (1, 1), padding='same', strides=(1, 1), use_bias=False, kernel_initializer=random_normal(stddev=0.02),
                name="blocks."+str(block_id)+"."+str(sub_block_id)+".shortcut.2")(res)
        res = BatchNormalization(name="blocks."+str(block_id)+"."+str(sub_block_id)+".shortcut.3")(res)
    x = Add()([res, x])

    return x

def Ghostnet(inputs):
    x = Conv2D(16, (3, 3), padding="same", strides=(2, 2), use_bias=False, kernel_initializer=random_normal(stddev=0.02), name="conv_stem")(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = Activation('relu')(x)

    x = _ghost_bottleneck(x, 16, 16, (3, 3), strides=1, ratio=2, squeeze=False, block_id=0, sub_block_id=0)
    
    x = _ghost_bottleneck(x, 24, 48, (3, 3), strides=2, ratio=2, squeeze=False, block_id=1, sub_block_id=0)
    x = _ghost_bottleneck(x, 24, 72, (3, 3), strides=1, ratio=2, squeeze=False, block_id=2, sub_block_id=0)

    x = _ghost_bottleneck(x, 40, 72, (5, 5), strides=2, ratio=2, squeeze=True, block_id=3, sub_block_id=0)
    x = _ghost_bottleneck(x, 40, 120, (5, 5), strides=1, ratio=2, squeeze=True, block_id=4, sub_block_id=0)
    feat1 = x

    x = _ghost_bottleneck(x, 80, 240, (3, 3), strides=2, ratio=2, squeeze=False, block_id=5, sub_block_id=0)
    x = _ghost_bottleneck(x, 80, 200, (3, 3), strides=1, ratio=2, squeeze=False, block_id=6, sub_block_id=0)
    x = _ghost_bottleneck(x, 80, 184, (3, 3), strides=1, ratio=2, squeeze=False, block_id=6, sub_block_id=1)
    x = _ghost_bottleneck(x, 80, 184, (3, 3), strides=1, ratio=2, squeeze=False, block_id=6, sub_block_id=2)
    x = _ghost_bottleneck(x, 112, 480, (3, 3), strides=1, ratio=2, squeeze=True, block_id=6, sub_block_id=3)
    x = _ghost_bottleneck(x, 112, 672, (3, 3), strides=1, ratio=2, squeeze=True, block_id=6, sub_block_id=4)
    feat2 = x

    x = _ghost_bottleneck(x, 160, 672, (5, 5), strides=2, ratio=2, squeeze=True, block_id=7, sub_block_id=0)
    x = _ghost_bottleneck(x, 160, 960, (5, 5), strides=1, ratio=2, squeeze=False, block_id=8, sub_block_id=0)
    x = _ghost_bottleneck(x, 160, 960, (5, 5), strides=1, ratio=2, squeeze=True, block_id=8, sub_block_id=1)
    x = _ghost_bottleneck(x, 160, 960, (5, 5), strides=1, ratio=2, squeeze=False, block_id=8, sub_block_id=2)
    x = _ghost_bottleneck(x, 160, 960, (5, 5), strides=1, ratio=2, squeeze=True, block_id=8, sub_block_id=3)
    feat3 = x

    return feat1,feat2,feat3

