import numpy as np
import tensorflow as tf
from keras import backend as K

from nets.ious import box_ciou

'''
label smoothing
'''
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
    
'''
convert feature layers to grount truth bounding box
'''
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    '''
    [1, 1, 1, num_anchors, 2]
    '''
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    '''
    get x,y grid
    (13, 13, 1, 2)
    '''
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    '''
    adjust prediction to (batch_size,13,13,3,85)
    85 = 4 + 1 + 80
    4 = parameters of center width and height
    1 = confidence level of box
    80 = confidence level of class
    '''
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    '''
    convert anchor box to ground truth bounding box
    box_xy = center point of box
    box_wh = width and height of box
    '''
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    '''
    return grid, feats, box_xy, box_wh at loss calculation
    return box_xy, box_wh, box_confidence, box_class_probs at prediction
    '''
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


'''
calculate iou between anchor box and ground truth bounding box
'''
def box_iou(b1, b2):
    13,13,3,1,4
    # coordinate of upper left and lower right corner
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    1,n,4
    # coordinate of upper left and lower right corner
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

'''
calculate loss
'''
def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0.1, print_loss=False, normalize=True):
    # three layers
    num_layers = len(anchors)//3 

    '''
    separate prediction and ground truth
    args = [*model_body.output, *y_true]
    y_true is a list with three feature layers with shape (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)
    yolo_outputs is a list with three feature layers with shape (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)
    '''
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]

    '''
    anchor of 13x13 feature layer = [142, 110], [192, 243], [459, 401]
    anchor of 26x26 feature layer = [36, 75], [76, 55], [72, 146]
    anchor of 52x52 feature layer = [12, 16], [19, 36], [40, 28]
    '''
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    # input_shpae = 416,416 
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))

    loss = 0
    num_pos = 0
    '''
    extract frame
    m = batch_size
    '''
    m = K.shape(yolo_outputs[0])[0]
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    '''
    y_true is a list with three feature layers with shape (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    yolo_outputs is a list with three feature layers with shape (m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    '''
    for l in range(num_layers):
        '''
        extract target position in feature layer (m,13,13,3,85)
        (m,13,13,3,1)
        '''
        object_mask = y_true[l][..., 4:5]
        '''
        get class (m,13,13,3,80)
        '''
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        '''
        process output of yolo_outputs and return four values:
        grid        (13,13,1,2)     grid coordinate
        raw_pred    (m,13,13,3,85)  unprocessed prediction
        pred_xy     (m,13,13,3,2)   decoded centerpoint coordinate
        pred_wh     (m,13,13,3,2)   decoded width and height
        '''
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        
        '''
        pred_box is decoded position of prediction
        (m,13,13,3,4)
        '''
        pred_box = K.concatenate([pred_xy, pred_wh])

        '''
        get negative sample array
        first create an empty array []
        '''
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        
        '''
        calculate ignore_mask for each frame
        '''
        def loop_body(b, ignore_mask):
            '''
            extract n anchor boxes: n,4
            '''
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            '''
            iou between anchor box and Ground Truth Bounding Box
            pred_box    13,13,3,4   # Coordinate or anchor box
            true_box    n,4         # coordinate of Ground Truth Bounding Box
            iou         13,13,3,n   # iou between anchor box and Ground Truth Bounding Box
            '''
            iou = box_iou(pred_box[b], true_box)

            '''
            best_iou    13,13,3
            overlap of each feature point with GTBB
            '''
            best_iou = K.max(iou, axis=-1)

            '''
            ignore anchor box with best_iou<ignore_thresh
            '''
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask

        '''
        loop for each frame
        '''
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])

        '''
        ignore_mask extracts feature point of negative sample
        (m,13,13,3)
        '''
        ignore_mask = ignore_mask.stack()
        (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        '''
        larger the ground truth bounding box, lower the weight
        '''
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        '''
        calculate Ciou loss
        '''
        raw_true_box = y_true[l][...,0:4]
        ciou = box_ciou(pred_box, raw_true_box)
        ciou_loss = object_mask * box_loss_scale * (1 - ciou)
        
        '''
        if anchor box present at position, calculate cross-entropy between 1 and confidence level
        if no anchor box present at position, calculate cross entropy between 0 and confidence level
        while ignoring samples with best_iou<ignore_thresh
        '''
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        location_loss = K.sum(ciou_loss)
        confidence_loss = K.sum(confidence_loss)
        class_loss = K.sum(class_loss)
        '''
        Calculate positive samples
        '''
        num_pos += tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        loss += location_loss + confidence_loss + class_loss
        if print_loss:
        loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
        
    if normalize:
        loss = loss / num_pos
    else:
        loss = loss / mf
    return loss
