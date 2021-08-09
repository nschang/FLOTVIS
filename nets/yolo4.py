from functools import wraps

import keras
import tensorflow as tf
from keras import backend as K
from keras.initializers import random_normal
from keras.layers import (Concatenate, Conv2D, MaxPooling2D, UpSampling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from utils.utils import compose

from nets.CSPdarknet53 import darknet_body

'''
# single convolution with DarknetConv2D
'''
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs = {'kernel_initializer' : random_normal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

'''
# convolution blocks -> convolution + standardization + activation function
# DarknetConv2D + BatchNormalization + LeakyReLU
'''
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

'''
# five convolutions
'''
def make_five_convs(x, num_filters):
    # five times convolution
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    return x

'''
# build Panet network and get prediction result
'''
def yolo_body(inputs, num_anchors, num_classes):
    '''   
    # generate the CSPdarknet53 backbone
    # with three feature layers of shape
    # 52,52,256
    # 26,26,512
    # 13,13,1024
    '''
    feat1,feat2,feat3 = darknet_body(inputs)

    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = DarknetConv2D_BN_Leaky(512, (1,1))(feat3)
    P5 = DarknetConv2D_BN_Leaky(1024, (3,3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1,1))(P5)
    # using SPP structure: stacking after max pooling at various scales
    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(512, (1,1))(P5)
    P5 = DarknetConv2D_BN_Leaky(1024, (3,3))(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1,1))(P5)

    # 13,13,512 -> 13,13,256 -> 26,26,256
    P5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1,1)), UpSampling2D(2))(P5)
    # 26,26,512 -> 26,26,256
    P4 = DarknetConv2D_BN_Leaky(256, (1,1))(feat2)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P4, P5_upsample])
    
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4,256)

    # 26,26,256 -> 26,26,128 -> 52,52,128
    P4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1,1)), UpSampling2D(2))(P4)
    # 52,52,256 -> 52,52,128
    P3 = DarknetConv2D_BN_Leaky(128, (1,1))(feat1)
    # 52,52,128 + 52,52,128 -> 52,52,256
    P3 = Concatenate()([P3, P4_upsample])

    # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    P3 = make_five_convs(P3,128)
    
    '''
    # third feature layer
    # y3=(batch_size,52,52,3,85)
    '''
    P3_output = DarknetConv2D_BN_Leaky(256, (3,3))(P3)
    P3_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1), kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01))(P3_output)

    # 52,52,128 -> 26,26,256
    P3_downsample = ZeroPadding2D(((1,0),(1,0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2))(P3_downsample)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P3_downsample, P4])
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4,256)
    
    '''
    # second feature layer
    # y2=(batch_size,26,26,3,85)
    '''
    P4_output = DarknetConv2D_BN_Leaky(512, (3,3))(P4)
    P4_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1), kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01))(P4_output)
    
    # 26,26,256 -> 13,13,512
    P4_downsample = ZeroPadding2D(((1,0),(1,0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2))(P4_downsample)
    # 13,13,512 + 13,13,512 -> 13,13,1024
    P5 = Concatenate()([P4_downsample, P5])
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = make_five_convs(P5,512)
    
    '''
    # first feature layer
    # y1=(batch_size,13,13,3,85)
    '''
    P5_output = DarknetConv2D_BN_Leaky(1024, (3,3))(P5)
    P5_output = DarknetConv2D(num_anchors*(num_classes+5), (1,1), kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01))(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])

'''
# turn feature layer into ground truth
'''
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    '''
    # [1, 1, 1, num_anchors, 2]
    '''
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    '''
    # get x,y grid
    # (13, 13, 1, 2)
    '''
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    '''
    # adjust predict result to (batch_size,13,13,3,85)
    # 85 = 4 + 1 + 80
    # where 4  is the parameter for width and height adjustment
    # 1 is the confidence level of the box
    # 80 is the confidence level of the class
    '''
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    '''
    # adjust predict value to true value
    # box_xy = center point of box
    # box_wh = width and height of box
    '''
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    #box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    '''
    # return grid, feats, box_xy, box_wh for loss calculation
    # return box_xy, box_wh, box_confidence, box_class_probs at prediction
    '''
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

'''
# adjust box size relative to image
'''
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''
    # put y_axis in front for convenient multiplication of width and height of box and image
    '''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    '''
    # offset of the effective area relative to the top left corner of the image
    # new_shape is the scale of width and height
    '''
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

'''
# get box position and score
'''
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, letterbox_image):
    '''
    # adjust predict value to true value
    # box_xy = center point of box = -1,13,13,3,2; 
    # box_wh = width and height of box = -1,13,13,3,2; 
    # box_confidence : -1,13,13,3,1; 
    # box_class_probs : -1,13,13,3,80;
    '''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    '''
    # if letterbox_image is True, gray bars are added around image
    # thus box_xy, box_wh are relative to image with gray bar
    # here we realign the image to remove the gray bars
    # turn box_xy and box_wh into y_min,y_max,xmin,xmax
    '''
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
    '''
    # get final score and box position
    '''
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

'''
# predict image
'''
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              letterbox_image=True):
    '''
    # obtain number of feature layers
    # number of valid layers = 3
    '''
    num_layers = len(yolo_outputs)
    '''
    # 13x13 sized feature map has anchor [142, 110], [192, 243], [459, 401]
    # 26x26 sized feature map has anchor [36, 75], [76, 55], [72, 146]
    # 52x52 sized feature map has anchor [12, 16], [19, 36], [40, 28]
    '''
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    
    '''
    # get size of input image (416x416 or 608x608)
    '''
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    '''
    # process each feature layer
    '''
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape, letterbox_image)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    '''
    # stack result of each feature layer
    '''
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    '''
    # compare score to score_threshold
    '''
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        '''
        # get box_scores >= score_threshold
        '''
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        '''
        # non-maximal suppression
        # retain boxes with best score
        '''
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        '''
        # get result after non-maximal suppression
        # including box position, score and class
        '''
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
