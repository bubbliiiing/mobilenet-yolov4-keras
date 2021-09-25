from functools import wraps

from keras import backend as K
from keras.initializers import random_normal
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          DepthwiseConv2D, Input, Lambda, MaxPooling2D,
                          UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from utils.utils import compose

from nets.ghostnet import Ghostnet
from nets.mobilenet_v1 import MobileNetV1
from nets.mobilenet_v2 import MobileNetV2
from nets.mobilenet_v3 import MobileNetV3
from nets.yolo_training import yolo_loss


def relu6(x):
    return K.relu(x, max_value=6)
    
#------------------------------------------------------#
#   单次卷积DarknetConv2D
#   如果步长为2则自己设定padding方式。
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02), 'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + Relu6
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Activation(relu6))

#---------------------------------------------------#
#   深度可分离卷积块
#   DepthwiseConv2D + BatchNormalization + Relu6
#---------------------------------------------------#
def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha = 1,
                          depth_multiplier=1, strides=(1, 1)):

    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    
    x = DepthwiseConv2D((3, 3), depthwise_initializer=random_normal(stddev=0.02),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False)(inputs)

    x = BatchNormalization()(x)
    x = Activation(relu6)(x)

    x = DarknetConv2D(pointwise_conv_filters, (1, 1), 
                    padding='same',
                    use_bias=False,
                    strides=(1, 1))(x)
    x = BatchNormalization()(x)
    return Activation(relu6)(x)
    
#---------------------------------------------------#
#   进行五次卷积
#---------------------------------------------------#
def make_five_convs(x, num_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = _depthwise_conv_block(x, num_filters*2,alpha=1)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = _depthwise_conv_block(x, num_filters*2,alpha=1)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    return x

#---------------------------------------------------#
#   Panet网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes, backbone="mobilenetv1", alpha=1):
    inputs = Input(input_shape)
    #---------------------------------------------------#   
    #   生成mobilnet的主干模型，获得三个有效特征层。
    #---------------------------------------------------#
    if backbone=="mobilenetv1":
        #---------------------------------------------------#   
        #   52,52,256；26,26,512；13,13,1024
        #---------------------------------------------------#
        feat1,feat2,feat3 = MobileNetV1(inputs, alpha=alpha)
    elif backbone=="mobilenetv2":
        #---------------------------------------------------#   
        #   52,52,32；26,26,92；13,13,320
        #---------------------------------------------------#
        feat1,feat2,feat3 = MobileNetV2(inputs, alpha=alpha)
    elif backbone=="mobilenetv3":
        #---------------------------------------------------#   
        #   52,52,40；26,26,112；13,13,160
        #---------------------------------------------------#
        feat1,feat2,feat3 = MobileNetV3(inputs, alpha=alpha)
    elif backbone=="ghostnet":
        #---------------------------------------------------#   
        #   52,52,40；26,26,112；13,13,160
        #---------------------------------------------------#
        feat1,feat2,feat3 = Ghostnet(inputs)
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenetv1, mobilenetv2, mobilenetv3, ghostnet.'.format(backbone))
    
    P5 = DarknetConv2D_BN_Leaky(int(512* alpha), (1,1))(feat3)
    P5 = _depthwise_conv_block(P5, int(1024* alpha))
    P5 = DarknetConv2D_BN_Leaky(int(512* alpha), (1,1))(P5)
    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(int(512* alpha), (1,1))(P5)
    P5 = _depthwise_conv_block(P5, int(1024* alpha))
    P5 = DarknetConv2D_BN_Leaky(int(512* alpha), (1,1))(P5)

    P5_upsample = compose(DarknetConv2D_BN_Leaky(int(256* alpha), (1,1)), UpSampling2D(2))(P5)
    
    P4 = DarknetConv2D_BN_Leaky(int(256* alpha), (1,1))(feat2)
    P4 = Concatenate()([P4, P5_upsample])
    P4 = make_five_convs(P4,int(256* alpha))

    P4_upsample = compose(DarknetConv2D_BN_Leaky(int(128* alpha), (1,1)), UpSampling2D(2))(P4)
    
    P3 = DarknetConv2D_BN_Leaky(int(128* alpha), (1,1))(feat1)
    P3 = Concatenate()([P3, P4_upsample])
    P3 = make_five_convs(P3,int(128* alpha))

    #---------------------------------------------------#
    #   第三个特征层
    #   y3=(batch_size,52,52,3,85)
    #---------------------------------------------------#
    P3_output = _depthwise_conv_block(P3, int(256* alpha))
    P3_output = DarknetConv2D(len(anchors_mask[0])*(num_classes+5), (1,1))(P3_output)

    P3_downsample = _depthwise_conv_block(P3, int(256* alpha), strides=(2,2))
    P4 = Concatenate()([P3_downsample, P4])
    P4 = make_five_convs(P4,int(256* alpha))
    
    #---------------------------------------------------#
    #   第二个特征层
    #   y2=(batch_size,26,26,3,85)
    #---------------------------------------------------#
    P4_output = _depthwise_conv_block(P4, int(512* alpha))
    P4_output = DarknetConv2D(len(anchors_mask[1])*(num_classes+5), (1,1))(P4_output)

    P4_downsample = _depthwise_conv_block(P4, int(512* alpha), strides=(2,2))
    P5 = Concatenate()([P4_downsample, P5])
    P5 = make_five_convs(P5,int(512* alpha))
    
    #---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,13,13,3,85)
    #---------------------------------------------------#
    P5_output = _depthwise_conv_block(P5, int(1024* alpha))
    P5_output = DarknetConv2D(len(anchors_mask[2])*(num_classes+5), (1,1))(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])


def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {'input_shape' : input_shape, 'anchors' : anchors, 'anchors_mask' : anchors_mask, 
                           'num_classes' : num_classes, 'label_smoothing' : label_smoothing}
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
