#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
from keras.layers import Input

from nets.yolo4 import yolo_body

if __name__ == "__main__":
    inputs = Input([416,416,3])
    
    model = yolo_body(inputs, 3, 80, backbone='ghostnet', alpha=1)
    model.summary()
    # mobilenetv1 41,005,757
    # mobilenetv2 39,124,541
    # mobilenetv3 40,043,389

    # 修改了PANET的mobilenetv1  12,754,109
    # 修改了PANET的mobilenetv2  10,872,893
    # 修改了PANET的mobilenetv3  11,791,741

    # 修改了PANET的mobilenetv1-0.25     680,381
    # 修改了PANET的mobilenetv2-0.5      2,370,541
    # 修改了PANET的mobilenetv3-0.75     6,315,309

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
