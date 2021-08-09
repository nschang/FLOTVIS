from keras import backend as K
import tensorflow as tf
import math
def box_ciou(b1, b2):
    '''
    input
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    returns
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    '''
    '''
    upper left and lower right of anchor box
    b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    '''
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    '''
    upper left and lower right of ground truth bounding box
    b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    '''
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    '''
    return iou tensor of anchor box and ground truth bounding box
    iou         (batch, feat_w, feat_h, anchor_num)
    '''
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    '''
    compute distance to center
    center_distance (batch, feat_w, feat_h, anchor_num)
    '''
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    '''
    compute diagonal distance
    enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    '''
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal ,K.epsilon())
    
    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
    alpha = v /  K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v

    ciou = K.expand_dims(ciou, -1)
    ciou = tf.where(tf.is_nan(ciou), tf.zeros_like(ciou), ciou)
    return ciou
