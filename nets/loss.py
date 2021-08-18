# -*- coding: utf-8 -*-
import math
import tensorflow as tf
from keras import backend as K
from utils.utils_anchors import yolo_anchor_decode
from nets.ious import box_ciou

# --------------------------------------------------------------
# label smoothing
# --------------------------------------------------------------
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    
# --------------------------------------------------------------
# convert layers of feature map to grount truth bounding box
# --------------------------------------------------------------
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)

    # --------------------------------------------------------------
    # get coordinates of each feature point
    # shape is  (13, 13, num_anchors, 2)
    # grid_x    [13, 13, 3, 1]
    # grid_y    [13, 13, 3, 1]
    # grid      [13, 13, 3, 2]
    # --------------------------------------------------------------
    grid_shape = K.shape(feats)[1:3]
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], num_anchors, 1])
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))
    # --------------------------------------------------------------
    # extend anchor tensor to shape [1, 1, 1, num_anchors, 2]
    # --------------------------------------------------------------
    anchors_tensor = K.tile(K.reshape(K.constant(anchors), [1, 1, num_anchors, 2]), [grid_shape[0], grid_shape[1], 1, 1])
    # --------------------------------------------------------------
    # adjust prediction to (batch_size,13,13,3,85)
    # 85 = 4 + 1 + 80
    # where: 4  = parameter for width and height adjustment
    #           = center of box and width, height
    #        1  = confidence score of boxes
    #        80 = confidence score of class
    # --------------------------------------------------------------
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # --------------------------------------------------------------
    # decode anchor box and normalize
    # box_xy = center of boxes
    # box_wh = width and height of boxes
    # --------------------------------------------------------------
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    #box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))
    # --------------------------------------------------------------
    # return confidence score of prediction box
    # --------------------------------------------------------------
    box_scores  = K.sigmoid(feats[..., 4:5])
    box_class_scores = K.sigmoid(feats[..., 5:])

    # --------------------------------------------------------------
    # return grid, feats, box_xy, box_wh for loss calculation
    # return box_xy, box_wh, box_scores, box_class_scores at prediction
    # --------------------------------------------------------------
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_scores, box_class_scores

# --------------------------------------------------------------
# compute iou between anchor box and ground truth bounding box
# --------------------------------------------------------------
def box_iou(b1, b2):
    # 13,13,3,1,4
    # coordinate of top left and bottom right corners
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # 1,n,4
    # coordinate of top left and bottom right corners
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # overlapping area
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou

# --------------------------------------------------------------
# compute loss
# --------------------------------------------------------------
def yolo_loss(args, 
            anchors, 
            num_classes, 
            ignore_thresh   = .5, 
            label_smoothing = 0.1, 
            print_loss      = False, 
            normalize       = True):
    # three layers
    num_layers = len(anchors)//3 

    # --------------------------------------------------------------
    # separate prediction and ground truth
    # args = [*model_body.output, *y_true]
    # y_true is a list with three layers of feature map of size (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)
    # yolo_outputs is a list with three layers of feature map of size (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)
    # --------------------------------------------------------------
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    # --------------------------------------------------------------
    # anchor of feature layer 13x13 is [142, 110], [192, 243], [459, 401]
    # anchor of feature layer 26x26 is [36, 75], [76, 55], [72, 146]
    # anchor of feature layer 52x52 is [12, 16], [19, 36], [40, 28]
    # --------------------------------------------------------------
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    # --------------------------------------------------------------
    # get input image = (416,416)
    # --------------------------------------------------------------
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    # --------------------------------------------------------------
    # extract each frame of image, where
    # m = batch_size
    # --------------------------------------------------------------
    m = K.shape(yolo_outputs[0])[0]

    loss = 0
    num_pos = 0
    # --------------------------------------------------------------
    # y_true is a list with three layers of feature map of shape (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)
    # yolo_outputs is a list with three layers of feature map of shape (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)
    # --------------------------------------------------------------
    for l in range(num_layers):
        # --------------------------------------------------------------
        # example: in feature layer (m,13,13,3,85)
        # retrieve coordinate (m,13,13,3,1) of overlapping points with target in layer of feature map 
        # --------------------------------------------------------------
        object_mask = y_true[l][..., 4:5]
        # --------------------------------------------------------------
        # retrieve respective class (m,13,13,3,80)
        # --------------------------------------------------------------
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        # --------------------------------------------------------------
        # process output of yolo_outputs and return four values:
        # grid        (13,13,1,2)     grid coordinate
        # raw_pred    (m,13,13,3,85)  unprocessed prediction
        # pred_xy     (m,13,13,3,2)   decoded coordinate of center of box
        # pred_wh     (m,13,13,3,2)   decoded coordinate of width and height
        # --------------------------------------------------------------
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        
        # --------------------------------------------------------------
        # pred_box is the predicted box position after decoding
        # (m,13,13,3,4)
        # --------------------------------------------------------------
        pred_box = K.concatenate([pred_xy, pred_wh])

        # --------------------------------------------------------------
        # generate array of negative samples
        # by first creating an empty array []
        # --------------------------------------------------------------
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        
        # --------------------------------------------------------------
        # compute ignore_mask for each frame of image
        # --------------------------------------------------------------
        def loop_body(b, ignore_mask):
            # --------------------------------------------------------------
            # extract n ground truth bounding boxes: n,4
            # --------------------------------------------------------------
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            # --------------------------------------------------------------
            # iou between anchor box and Ground Truth Bounding Box
            # pred_box    13,13,3,4   # Coordinate or anchor box
            # true_box    n,4         # coordinate of Ground Truth Bounding Box
            # iou         13,13,3,n   # iou between anchor box and Ground Truth Bounding Box
            # --------------------------------------------------------------
            iou = box_iou(pred_box[b], true_box)

            # --------------------------------------------------------------
            # max overlap of each feature point with ground truth
            # best_iou    13,13,3
            # --------------------------------------------------------------
            best_iou = K.max(iou, axis=-1)

            # --------------------------------------------------------------
            # if best_iou < ignore_thresh then this anchor box does not
            # have corresponding ground truth bounding box
            # this ignores feature points with good match to ground truth for negative samples
            # --------------------------------------------------------------
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask

        # --------------------------------------------------------------
        # loop for each frame of image
        # --------------------------------------------------------------
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])

        # --------------------------------------------------------------
        # ignore_mask extracts feature points to represent negative samples
        # (m,13,13,3)
        # --------------------------------------------------------------
        ignore_mask = ignore_mask.stack()
        # (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # --------------------------------------------------------------
        # larger the ground truth bounding box, lower the weight
        # --------------------------------------------------------------
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # --------------------------------------------------------------
        # compute Ciou loss
        # --------------------------------------------------------------
        raw_true_box = y_true[l][...,0:4]
        ciou         = box_ciou(pred_box, raw_true_box)
        ciou_loss    = object_mask * box_loss_scale * (1 - ciou)
        
        # --------------------------------------------------------------
        # if anchor box is present at position, compute cross-entropy of 1 and confidence score
        # if no anchor box present at position, compute cross-entropy of 0 and confidence score
        # anchors where best_iou < ignore_thresh will be ignored
        # this ignores feature points with good match to ground truth for negative samples
        # --------------------------------------------------------------
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        location_loss   = K.sum(ciou_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss      = K.sum(class_loss)
        # --------------------------------------------------------------
        # compute number of positive samples
        # --------------------------------------------------------------
        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        loss    += location_loss + confidence_loss + class_loss
        # if print_loss:
        #     loss     = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
        
    loss = loss / num_pos

