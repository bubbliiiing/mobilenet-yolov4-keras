from functools import wraps

import tensorflow as tf
from keras import backend as K
from keras.initializers import random_normal
from keras.layers import (Activation, Concatenate, Conv2D, DepthwiseConv2D,
                          MaxPooling2D, UpSampling2D)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from utils.utils import compose

from nets.ghostnet import Ghostnet
from nets.mobilenet_v1 import MobileNetV1
from nets.mobilenet_v2 import MobileNetV2
from nets.mobilenet_v3 import MobileNetV3


def relu6(x):
    return K.relu(x, max_value=6)

#--------------------------------------------------#
#   单次卷积DarknetConv2D
#   正则化系数为5e-4
#   如果步长为2则自己设定padding方式。
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
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

    x = Conv2D(pointwise_conv_filters, (1, 1), kernel_initializer=random_normal(stddev=0.02),
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
def yolo_body(inputs, num_anchors, num_classes, backbone="mobilenetv1", alpha=1):
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
    P3_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(P3_output)

    P3_downsample = _depthwise_conv_block(P3, int(256* alpha), strides=(2,2))
    P4 = Concatenate()([P3_downsample, P4])
    P4 = make_five_convs(P4,int(256* alpha))
    
    #---------------------------------------------------#
    #   第二个特征层
    #   y2=(batch_size,26,26,3,85)
    #---------------------------------------------------#
    P4_output = _depthwise_conv_block(P4, int(512* alpha))
    P4_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(P4_output)

    P4_downsample = _depthwise_conv_block(P4, int(512* alpha), strides=(2,2))
    P5 = Concatenate()([P4_downsample, P5])
    P5 = make_five_convs(P5,int(512* alpha))
    
    #---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,13,13,3,85)
    #---------------------------------------------------#
    P5_output = _depthwise_conv_block(P5, int(1024* alpha))
    P5_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1))(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])

#---------------------------------------------------#
#   将预测值的每个特征层调成真实值
#---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    #---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    #---------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    #---------------------------------------------------#
    #   获得x，y的网格
    #   (13, 13, 1, 2)
    #---------------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    #---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    #---------------------------------------------------#
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    #---------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy对应框的中心点
    #   box_wh对应框的宽和高
    #---------------------------------------------------#
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    #---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    #-----------------------------------------------------------------#
    #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
    #   new_shape指的是宽高缩放情况
    #-----------------------------------------------------------------#
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   获取每个box和它的得分
#---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   将预测值调成真实值
    #   box_xy : -1,13,13,3,2; 
    #   box_wh : -1,13,13,3,2; 
    #   box_confidence : -1,13,13,3,1; 
    #   box_class_probs : -1,13,13,3,80;
    #-----------------------------------------------------------------#
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    #-----------------------------------------------------------------#
    #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
    #   因此生成的box_xy, box_wh是相对于有灰条的图像的
    #   我们需要对齐进行修改，去除灰条的部分。
    #   将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    #-----------------------------------------------------------------#
    if letterbox_image:
        boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    else:
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))

        boxes =  K.concatenate([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ])
    #-----------------------------------------------------------------#
    #   获得最终得分和框的位置
    #-----------------------------------------------------------------#
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

#---------------------------------------------------#
#   图片预测
#---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              letterbox_image=True):
    #---------------------------------------------------#
    #   获得特征层的数量，有效特征层的数量为3
    #---------------------------------------------------#
    num_layers = len(yolo_outputs)
    #-----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    #-----------------------------------------------------------#
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    
    #-----------------------------------------------------------#
    #   这里获得的是输入图片的大小，一般是416x416
    #-----------------------------------------------------------#
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    #-----------------------------------------------------------#
    #   对每个特征层进行处理
    #-----------------------------------------------------------#
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape, letterbox_image)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    #-----------------------------------------------------------#
    #   将每个特征层的结果进行堆叠
    #-----------------------------------------------------------#
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    #-----------------------------------------------------------#
    #   判断得分是否大于score_threshold
    #-----------------------------------------------------------#
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成绩
        #-----------------------------------------------------------#
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        #-----------------------------------------------------------#
        #   非极大抑制
        #   保留一定区域内得分最大的框
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        #-----------------------------------------------------------#
        #   获取非极大抑制后的结果
        #   下列三个分别是
        #   框的位置，得分与种类
        #-----------------------------------------------------------#
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


