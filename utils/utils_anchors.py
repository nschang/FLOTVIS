# -*- coding: utf-8 -*-
import tensorflow as tf
from keras import backend as K

# ------------------------------
# adjust ground truth bounding box size relative to image size
# ------------------------------
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # ------------------------------
    # put y_axis above to multiply width and height of boxes and image
    # ------------------------------
    box_yx      = box_xy[..., ::-1]
    box_hw      = box_wh[..., ::-1]
    
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    
    if letterbox_image:
        # ------------------------------
        # offset of effective area relative to the top left corner of image
        # new_shape is the scale of width and height
        # ------------------------------
        new_shape   = K.round(image_shape * K.min(input_shape / image_shape))
        offset      = (input_shape - new_shape) / 2. / input_shape
        scale       = input_shape / new_shape

        box_yx      = (box_yx - offset) * scale
        box_hw     *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes       =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1], # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

# ------------------------------
# get anchor and decode
# ------------------------------
def yolo_anchor_decode(feats, anchors, num_classes, input_shape, calc_loss=False): # same as yolo_anchor_decode
    num_anchors = len(anchors)

    # ------------------------------
    # get coordinates of each feature point
    # shape is  (13, 13, num_anchors, 2)
    # grid_y    [13, 13, 3, 1]
    # grid_x    [13, 13, 3, 1]
    # grid      [13, 13, 3, 2]
    # ------------------------------
    grid_shape = K.shape(feats)[1:3]
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], num_anchors, 1])
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))
    # ------------------------------
    # extend anchor tensor to shape [1, 1, 1, num_anchors, 2]
    # ------------------------------
    anchors_tensor = K.tile(K.reshape(K.constant(anchors), [1, 1, num_anchors, 2]), [grid_shape[0], grid_shape[1], 1, 1])
    # ------------------------------
    # decode predicted box as (batch_size,13,13,3,85)
    # 85 = 4 + 1 + 80
    # where: 4  = parameter for width and height adjustment
    #           = center of box and width, height
    #        1  = confidence score of boxes
    #        80 = confidence score of class
    # ------------------------------
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # ------------------------------
    # decode anchor box and normalize
    # box_xy = center of boxes
    # box_wh = width and height of boxes
    # ------------------------------    
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    #box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))
    # ------------------------------
    # return confidence score of prediction box
    # ------------------------------
    box_scores  = K.sigmoid(feats[..., 4:5])
    box_class_scores = K.sigmoid(feats[..., 5:])

    # ------------------------------
    # return grid, feats, box_xy, box_wh for loss calculation
    # return box_xy, box_wh, box_scores, box_class_scores at prediction
    # ------------------------------
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_scores, box_class_scores

# ------------------------------
# predict image
# ------------------------------
# yolo_process handles post-processing of detection result
# which includes Decoding, Non-Maximum Suppression (NMS),
# Thresholding, etc.
# ------------------------------
def yolo_process(yolo_outputs,
            anchors,
            num_classes,
            image_shape,
            input_shape,
            # ------------------------------
            # anchor of feature layer 13x13 is [142, 110], [192, 243], [459, 401]
            # anchor of feature layer 26x26 is [36, 75], [76, 55], [72, 146]
            # anchor of feature layer 52x52 is [12, 16], [19, 36], [40, 28]
            # ------------------------------
            anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            max_boxes       = 100,
            score_threshold = .5,
            iou_threshold   = .3,
            letterbox_image = True):
    # ------------------------------
    # number of valid layers of feature map = 3
    # ------------------------------
    # num_layers = len(yolo_outputs)
    # ------------------------------
    # size of input image = 416x416 or 608x608
    # ------------------------------
    # input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    # ------------------------------
    # process each feature layer and stack results
    box_xy           = []
    box_wh           = []
    box_scores       = []
    box_class_scores = []
    for l in range(len(yolo_outputs)):
        sub_box_xy, sub_box_wh, sub_box_scores, sub_box_class_scores = \
            yolo_anchor_decode(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_scores.append(K.reshape(sub_box_scores, [-1, 1]))
        box_class_scores.append(K.reshape(sub_box_class_scores, [-1, num_classes]))
    box_xy           = K.concatenate(box_xy, axis = 0)
    box_wh           = K.concatenate(box_wh, axis = 0)
    box_scores       = K.concatenate(box_scores, axis=0)
    box_class_scores = K.concatenate(box_class_scores, axis = 0)
    # ------------------------------
    # letterbox_image adds gray bars to sides of image
    # box_xy, box_wh are relative to image with gray bars
    # the gray bars need to be removed in order to
    # convert box_xy, box_wh to y_min,y_max,xmin,xmax
    # ------------------------------
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    box_scores  = box_scores * box_class_scores
    # ------------------------------
    # compare if score >= score_threshold
    # ------------------------------
    mask             = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out        = []
    scores_out       = []
    classes_out      = []
    for c in range(num_classes):
        # ------------------------------
        # get boxes and scores where box_scores >= score_threshold
        # ------------------------------
        class_boxes         = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores    = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # ------------------------------
        # non-maximal suppression
        # keep boxes with best score
        # ------------------------------
        nms_index = tf.image.non_max_suppression(class_boxes, 
        class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        # ------------------------------
        # get result after non-maximal suppression
        # including box position, score and class
        # ------------------------------
        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c
        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out   = K.concatenate(boxes_out, axis=0)
    scores_out  = K.concatenate(scores_out, axis=0)
    classes_out = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out

# ------------------------------
# vision_for_anchors
# ------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    # 13x13
    def yolo_anchor_decode(feats, anchors, num_classes):
        # ------------------------------
        # feats         [batch_size, 13, 13, 3 * (5 + num_classes)]
        # anchors       [3, 2]
        # num_classes   1
        # ------------------------------
        num_anchors = len(anchors)
        # ------------------------------
        # grid_shape [13, 13] is width, height of feature layer
        # ------------------------------
        grid_shape = np.shape(feats)[1:3] # height, width
        print(grid_shape)
        # ------------------------------
        # get coordinates of each feature point
        # shape is  (13, 13, num_anchors, 2)
        # grid_y    [13, 13, 3, 1]
        # grid_x    [13, 13, 3, 1]
        # grid      [13, 13, 3, 2]
        # ------------------------------
        grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
        grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
        grid = np.concatenate([grid_x, grid_y], -1)
        print(np.shape(grid))
        # ------------------------------
        # expand anchor box
        # shape is  (13, 13, num_anchors, 2)
        # ------------------------------
        anchors_tensor = np.reshape(anchors, [1, 1, num_anchors, 2])
        anchors_tensor = np.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])
        # ------------------------------
        # decode predicted box as (batch_size,13,13,3,85)
        # 85 = 4 + 1 + 80
        # where: 4  = parameter for width and height adjustment
        #           = center of box and width, height
        #        1  = confidence score of boxes
        #        80 = confidence score of class
        # ------------------------------
        feats = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
        # ------------------------------
        # decode anchor box and normalize
        # box_xy = center of boxes
        # box_wh = width and height of boxes
        # ------------------------------
        box_xy = (sigmoid(feats[..., :2]) + grid)
        box_wh = np.exp(feats[..., 2:4]) * anchors_tensor
        
        # confidence score of prediction box
        box_scores  = sigmoid(feats[..., 4:5])
        box_class_scores = sigmoid(feats[..., 5:])

        box_wh = box_wh / 32
        anchors_tensor = anchors_tensor / 32

        fig = plt.figure()
        ax = fig.add_subplot(121)
        plt.ylim(-2,15)
        plt.xlim(-2,15)
        plt.scatter(grid_x,grid_y)
        plt.scatter(5,5,c='black')
        plt.gca().invert_yaxis()

        anchor_left = grid_x - anchors_tensor/2 
        anchor_top = grid_y - anchors_tensor/2 
        print(np.shape(anchors_tensor))
        print(np.shape(box_xy))

        rect1 = plt.Rectangle([anchor_left[5,5,0,0],anchor_top[5,5,0,1]],anchors_tensor[0,0,0,0],anchors_tensor[0,0,0,1],color="r",fill=False)
        rect2 = plt.Rectangle([anchor_left[5,5,1,0],anchor_top[5,5,1,1]],anchors_tensor[0,0,1,0],anchors_tensor[0,0,1,1],color="r",fill=False)
        rect3 = plt.Rectangle([anchor_left[5,5,2,0],anchor_top[5,5,2,1]],anchors_tensor[0,0,2,0],anchors_tensor[0,0,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax = fig.add_subplot(122)
        plt.ylim(-2,15)
        plt.xlim(-2,15)
        plt.scatter(grid_x,grid_y)
        plt.scatter(5,5,c='black')
        plt.scatter(box_xy[0,5,5,:,0],box_xy[0,5,5,:,1],c='r')
        plt.gca().invert_yaxis()

        pre_left = box_xy[...,0] - box_wh[...,0]/2 
        pre_top = box_xy[...,1] - box_wh[...,1]/2 

        rect1 = plt.Rectangle([pre_left[0,5,5,0],pre_top[0,5,5,0]],box_wh[0,5,5,0,0],box_wh[0,5,5,0,1],color="r",fill=False)
        rect2 = plt.Rectangle([pre_left[0,5,5,1],pre_top[0,5,5,1]],box_wh[0,5,5,1,0],box_wh[0,5,5,1,1],color="r",fill=False)
        rect3 = plt.Rectangle([pre_left[0,5,5,2],pre_top[0,5,5,2]],box_wh[0,5,5,2,0],box_wh[0,5,5,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()
        #
    feat = np.random.normal(0,0.5,[4,13,13,75])
    anchors = [[142, 110],[192, 243],[459, 401]]
    yolo_anchor_decode(feat,anchors,20)
