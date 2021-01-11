#-------------------------------------------------------------#
#   MobileNet的网络部分
#-------------------------------------------------------------#
import keras.backend as backend
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Activation, Add, Concatenate, Conv2D,
                          DepthwiseConv2D, MaxPooling2D, UpSampling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    
    # 深度可分离卷积
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # 1x1卷积
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
                      padding='same',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def relu6(x):
    return K.relu(x, max_value=6)


def MobileNetV1(inputs,alpha=1,depth_multiplier=1):
    if alpha not in [0.25, 0.5, 0.75, 1.0]:
        raise ValueError('Unsupported alpha - `{}` in MobilenetV1, Use 0.25, 0.5, 0.75, 1.0'.format(alpha))

    # 416,416,3 -> 208,208,32
    x = _conv_block(inputs, 32, alpha, strides=(2, 2))
    # 208,208,32 -> 208,208,64
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    # 208,208,64 -> 104,104,128
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    # 104,104.128 -> 64,64,256
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    feat1 = x

    # 64,64,256 -> 32,32,512
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    feat2 = x

    # 32,32,512 -> 16,16,1024
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    feat3 = x

    return feat1,feat2,feat3

if __name__ == "__main__":
    from keras.layers import Input
    from keras.models import Model
    alpha   = 0.25
    inputs  = Input([None,None,3])
    outputs = MobileNetV1(inputs,alpha=alpha)
    model   = Model(inputs,outputs)
    model.summary()
