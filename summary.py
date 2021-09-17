#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.yolo import yolo_body

if __name__ == "__main__":
    input_shape     = [416, 416, 3]
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_classes     = 80

    model = yolo_body(input_shape, anchors_mask, num_classes)
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
    
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
