from keras import backend
from keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                          DepthwiseConv2D, GlobalAveragePooling2D, Multiply,
                          Reshape)


def _activation(x, name='relu'):
    if name == 'relu':
        return Activation('relu')(x)
    elif name == 'hardswish':
        return hard_swish(x)

def hard_sigmoid(x):
    return backend.relu(x + 3.0, max_value=6.0) / 6.0

def hard_swish(x):
    return Multiply()([Activation(hard_sigmoid)(x), x])

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _bneck(inputs, expansion, alpha, out_ch, kernel_size, stride, se_ratio, activation,
                        block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    out_channels = _make_divisible(out_ch * alpha, 8)
    exp_size = _make_divisible(in_channels * expansion, 8)
    x = inputs
    prefix = 'expanded_conv/'
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = Conv2D(exp_size,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis,
                                      name=prefix + 'expand/BatchNorm')(x)
        x = _activation(x, activation)

    x = DepthwiseConv2D(kernel_size,
                               strides=stride,
                               padding='same',
                               dilation_rate=1,
                               use_bias=False,
                               name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                                  name=prefix + 'depthwise/BatchNorm')(x)
    x = _activation(x, activation)

    if se_ratio:
        reduced_ch = _make_divisible(exp_size * se_ratio, 8)
        y = GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(x)
        y = Reshape([1, 1, exp_size], name=prefix + 'reshape')(y)
        y = Conv2D(reduced_ch,
                          kernel_size=1,
                          padding='same',
                          use_bias=True,
                          name=prefix + 'squeeze_excite/Conv')(y)
        y = Activation("relu", name=prefix + 'squeeze_excite/Relu')(y)
        y = Conv2D(exp_size,
                          kernel_size=1,
                          padding='same',
                          use_bias=True,
                          name=prefix + 'squeeze_excite/Conv_1')(y)
        x = Multiply(name=prefix + 'squeeze_excite/Mul')([Activation(hard_sigmoid)(y), x])

    x = Conv2D(out_channels,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis,
                                  name=prefix + 'project/BatchNorm')(x)

    if in_channels == out_channels and stride == 1:
        x = Add(name=prefix + 'Add')([inputs, x])
    return x

def MobileNetV3(inputs, alpha=1.0, kernel=5, se_ratio=0.25):
    if alpha not in [0.75, 1.0]:
        raise ValueError('Unsupported alpha - `{}` in MobilenetV3, Use 0.75, 1.0.'.format(alpha))
    # 416,416,3 -> 208,208,16
    x = Conv2D(16, kernel_size=3, strides=(2, 2), padding='same',
                      use_bias=False,
                      name='Conv')(inputs)
    x = BatchNormalization(axis=-1,
                            epsilon=1e-3,
                            momentum=0.999,
                            name='Conv/BatchNorm')(x)
    x = Activation(hard_swish)(x)

    # 208,208,16 -> 208,208,16
    x = _bneck(x, 1, 16, alpha, 3, 1, None, 'relu', 0)

    # 208,208,16 -> 104,104,24
    x = _bneck(x, 4, 24, alpha, 3, 2, None, 'relu', 1)
    x = _bneck(x, 3, 24, alpha, 3, 1, None, 'relu', 2)
    
    # 104,104,24 -> 52,52,40
    x = _bneck(x, 3, 40, alpha, kernel, 2, se_ratio, 'relu', 3)
    x = _bneck(x, 3, 40, alpha, kernel, 1, se_ratio, 'relu', 4)
    x = _bneck(x, 3, 40, alpha, kernel, 1, se_ratio, 'relu', 5)
    feat1 = x

    # 52,52,40 -> 26,26,112
    x = _bneck(x, 6, 80, alpha, 3, 2, None, 'hardswish', 6)
    x = _bneck(x, 2.5, 80, alpha, 3, 1, None, 'hardswish', 7)
    x = _bneck(x, 2.3, 80, alpha, 3, 1, None, 'hardswish', 8)
    x = _bneck(x, 2.3, 80, alpha, 3, 1, None, 'hardswish', 9)
    x = _bneck(x, 6, 112, alpha, 3, 1, se_ratio, 'hardswish', 10)
    x = _bneck(x, 6, 112, alpha, 3, 1, se_ratio, 'hardswish', 11)
    feat2 = x

    # 26,26,112 -> 13,13,160
    x = _bneck(x, 6, 160, alpha, kernel, 2, se_ratio, 'hardswish', 12)
    x = _bneck(x, 6, 160, alpha, kernel, 1, se_ratio, 'hardswish', 13)
    x = _bneck(x, 6, 160, alpha, kernel, 1, se_ratio, 'hardswish', 14)
    feat3 = x

    return feat1, feat2, feat3
