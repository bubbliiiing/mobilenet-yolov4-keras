from keras import layers
from keras import backend 

def conv_block(x, growth_rate, name):
    x1 = layers.BatchNormalization(name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)

    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate( name=name + '_concat')([x, x1])
    return x

def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x
    
def transition_block(x, reduction, name):
    x = layers.BatchNormalization(name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[-1] * reduction), 1, use_bias=False, name=name + '_conv')(x)
    feat = x
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return feat, x

def DenseNet(inputs, backbone):
    blocks = {
        'densenet121' : [6, 12, 24, 16],
        'densenet169' : [6, 12, 32, 32],
        'densenet201' : [6, 12, 48, 32],
    }[backbone]

    # 416, 416, 3 -> 208, 208, 64
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization( epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    
    # 208, 208, 64 -> 104, 104, 64
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    # Densenet121 104, 104, 64 -> 104, 104, 64 + 32 * 6 == 104, 104, 256
    x = dense_block(x, blocks[0], name='conv2')

    # Densenet121 104, 104, 256 -> 52, 52, 32 + 16 * 6 == 52, 52, 128
    _, x = transition_block(x, 0.5, name='pool2')

    # Densenet121 52, 52, 128 -> 52, 52, 128 + 32 * 12 == 52, 52, 512
    x = dense_block(x, blocks[1], name='conv3')
    
    # Densenet121 52, 52, 512 -> 26, 26, 256
    feat1, x = transition_block(x, 0.5, name='pool3')

    # Densenet121 26, 26, 256 -> 26, 26, 256 + 32 * 24 == 26, 26, 1024
    x = dense_block(x, blocks[2], name='conv4')

    # Densenet121 26, 26, 1024 -> 13, 13, 512
    feat2, x  = transition_block(x, 0.5, name='pool4')

    # Densenet121 13, 13, 512 -> 13, 13, 512 + 32 * 16 == 13, 13, 1024
    x = dense_block(x, blocks[3], name='conv5')

    x = layers.BatchNormalization(name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)
    feat3 = x

    return feat1, feat2, feat3
