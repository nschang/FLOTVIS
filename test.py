# -----------------------------------------------------------
# test network structure
# -----------------------------------------------------------
from keras.layers import Input
from nets.yolo4 import yolo_body

if __name__ == "__main__":
    input_shape     = Input([416, 416, 3])
    #input_shape     = [416, 416, 3]
    #anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors_mask    = 3
    num_classes     = 80
    model           = yolo_body(input_shape, anchors_mask, num_classes)
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
