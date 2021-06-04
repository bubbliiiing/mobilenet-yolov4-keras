## YOLOV4：You Only Look Once目标检测模型-修改mobilenet系列主干网络-在Keras当中的实现
---

**2021年2月8日更新：**   
**加入letterbox_image的选项，关闭letterbox_image后网络的map一般可以得到提升。**

**2021年6月4日更新：**   
**加入ghostnet主干网络。**

## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [注意事项 Attention](#注意事项)
4. [小技巧的设置 TricksSet](#小技巧的设置)
5. [文件下载 Download](#文件下载)
6. [预测步骤 How2predict](#预测步骤)
7. [训练步骤 How2train](#训练步骤)
8. [评估步骤 How2eval](#评估步骤)
9. [参考资料 Reference](#Reference)

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12 | [yolov4_mobilenet_v1_025_voc.h5](https://github.com/bubbliiiing/mobilenet-yolov4-lite-keras/releases/download/v1.0/yolov4_mobilenet_v1_025_voc.h5) | VOC-Test07 | 416x416 | - | 66.29
| VOC07+12 | [yolov4_mobilenet_v1_voc.h5](https://github.com/bubbliiiing/mobilenet-yolov4-lite-keras/releases/download/v1.0/yolov4_mobilenet_v1_voc.h5) | VOC-Test07 | 416x416 | - | 80.18
| VOC07+12 | [yolov4_mobilenet_v2_voc.h5](https://github.com/bubbliiiing/mobilenet-yolov4-lite-keras/releases/download/v1.0/yolov4_mobilenet_v2_voc.h5) | VOC-Test07 | 416x416 | - | 79.72
| VOC07+12 | [yolov4_mobilenet_v3_voc.h5](https://github.com/bubbliiiing/mobilenet-yolov4-lite-keras/releases/download/v1.0/yolov4_mobilenet_v3_voc.h5) | VOC-Test07 | 416x416 | - | 78.45
| VOC07+12 | [yolov4_ghostnet_voc.h5](https://github.com/bubbliiiing/mobilenet-yolov4-lite-keras/releases/download/v1.0/yolov4_ghostnet_voc.h5) | VOC-Test07 | 416x416 | - | 78.64

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5

## 注意事项
提供的四个训练好的权重分别是基于mobilenetv1-025、mobilenetv1、mobilenetv2、mobilenetv3主干网络训练而成的。使用的时候注意backbone和权重的对应。   
训练前注意修改alpha、model_path和backbone使得三者对应。  
预测前注意修改alhpa、model_path和backbone使得三者对应。  

## 小技巧的设置
在train.py文件下：   
1、mosaic参数可用于控制是否实现Mosaic数据增强。    
2、Cosine_scheduler可用于控制是否使用学习率余弦退火衰减。    
3、label_smoothing可用于控制是否Label Smoothing平滑。  

## 文件下载
训练所需的各个权值可在百度网盘中下载。    
链接: https://pan.baidu.com/s/1CA4d9gVBI5JObnhK_xRPxQ 提取码: 9am5   
四个已经训练好的权重均为VOC数据集的权重。  
  
VOC数据集下载地址如下：  
VOC2007+2012训练集    
链接: https://pan.baidu.com/s/16pemiBGd-P9q2j7dZKGDFA 提取码: eiw9    

VOC2007测试集   
链接: https://pan.baidu.com/s/1BnMiFwlNwIWG9gsd4jHLig 提取码: dsda   

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载权重，放入model_data，运行predict.py，输入  
```python
img/street.jpg
``` 
2. 利用video.py可进行摄像头检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path、alpha和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，alpha是通道的缩放比例，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path"        : 'model_data/yolov4_mobilenet_v1_voc.h5',
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "classes_path"      : 'model_data/voc_classes.txt',
    "backbone"          : 'mobilenetv1',
    "alpha"             : 1,
    "score"             : 0.5,
    "iou"               : 0.3,
    "max_boxes"         : 100,
    # 显存比较小可以使用416x416
    # 显存比较大可以使用608x608
    "model_image_size"  : (416, 416)
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 利用video.py可进行摄像头检测。  

## 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在训练前利用voc2yolo4.py文件生成对应的txt。  
5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6. 此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：   
```python
classes_path = 'model_data/new_classes.txt'    
```
model_data/new_classes.txt文件内容为：   
```python
cat
dog
...
```
8. 运行train.py即可开始训练。

## 评估步骤
评估过程可参考视频https://www.bilibili.com/video/BV1zE411u7Vw  
步骤是一样的，不需要自己再建立get_dr_txt.py、get_gt_txt.py等文件。  
1. 本文使用VOC格式进行评估。  
2. 评估前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 评估前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。  
4. 在评估前利用voc2yolo4.py文件生成对应的txt，评估用的txt为VOCdevkit/VOC2007/ImageSets/Main/test.txt，需要注意的是，如果整个VOC2007里面的数据集都是用于评估，那么直接将trainval_percent设置成0即可。  
5. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
6. 运行get_dr_txt.py和get_gt_txt.py，在./input/detection-results和./input/ground-truth文件夹下生成对应的txt。  
7. 运行get_map.py即可开始计算模型的mAP。

## mAP目标检测精度计算更新
更新了get_gt_txt.py、get_dr_txt.py和get_map.py文件。  
get_map文件克隆自https://github.com/Cartucho/mAP  
具体mAP计算过程可参考：https://www.bilibili.com/video/BV1zE411u7Vw

## Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4  
