# -*- coding: utf-8 -*-
import numpy as np
from random import shuffle, sample
import cv2
import keras
from PIL import Image
from utils.utils import cvtColor, preprocess_input

class YoloDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, mosaic, train):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.anchors_mask       = anchors_mask
        self.mosaic             = mosaic
        self.train              = train

    def __len__(self):
        return np.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        image_data  = []
        box_data    = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.length
            # -----------------------------------------------------------
            # random image augmentation for training set only
            # no augmentation for validation set
            # -----------------------------------------------------------
            if self.mosaic:
                if self.rand() < 0.5:
                    lines = sample(self.annotation_lines, 3)
                    lines.append(self.annotation_lines[i])
                    shuffle(lines)
                    image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
                else:
                    image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
            else:
                image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
            image_data.append(preprocess_input(np.array(image)))
            box_data.append(box)

        image_data  = np.array(image_data)
        box_data    = np.array(box_data)
        y_true      = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size)

    def on_epoch_begin(self):
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, 
                        max_boxes=100, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        # random preprocessing for real-time data augmentation
        line    = annotation_line.split()
        # cv2 convert color space
        image   = Image.open(line[0])
        image   = cvtColor(image)
        # width, height of image
        iw, ih  = image.size
        # height, width of image
        h, w    = input_shape
        # prediction box
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            # resize image
            scale = min(w / iw, h / ih)
            nw    = int(iw * scale)
            nh    = int(ih * scale)
            dx    = (w-nw) // 2
            dy    = (h-nh) // 2

            # resize, add gray bar to sides of image
            image      = image.resize((nw,nh), Image.BICUBIC)
            new_image  = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # adjust ground truth bounding box
            box_data = np.zeros((max_boxes,5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]] * nw / iw + dx
                box[:, [1,3]] = box[:, [1,3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w]     = w
                box[:, 3][box[:, 3]>h]     = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box   = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
                if len(box)>max_boxes: box = box[:max_boxes]
                box_data[:len(box)] = box

            return image_data, box_data
            
        # zoom image and adjust width and length
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # resize, add gray bar to sides of image
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # cv2 convert color space to HSV
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        # adjust ground truth bounding box
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box
        
        return image_data, box_data

    def merge_bboxes(self, bboxes, cutx, cuty): #def merge_bboxes(bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                        # if y2-y1 < 5:
                        #     continue
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                        # if x2-x1 < 5:
                        #     continue
                    
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                        # if y2-y1 < 5:
                        #     continue
                    
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                        # if x2-x1 < 5:
                        #     continue

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                        # if y2-y1 < 5:
                        #     continue

                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                        # if x2-x1 < 5:
                        #     continue

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                        # if y2-y1 < 5:
                        #     continue

                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                        # if x2-x1 < 5:
                        #     continue

                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, 
                                    max_boxes=100, hue=.1, sat=1.5, val=1.5):
        # random preprocessing for real-time data augmentation
        h, w = input_shape
        min_offset_x = self.rand(0.25, 0.75)
        min_offset_y = self.rand(0.25, 0.75)
        
        nws     = [ int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), 
                    int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1))]
        nhs     = [ int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), 
                    int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1))]
        
        image_datas = [] 
        box_datas = []
        index = 0

        place_x = [int(w * min_offset_x) - nws[0], 
                   int(w * min_offset_x) - nws[1], 
                   int(w * min_offset_x), 
                   int(w * min_offset_x)]
        place_y = [int(h * min_offset_y) - nhs[0], 
                   int(h * min_offset_y), 
                   int(h * min_offset_y), 
                   int(h * min_offset_y) - nhs[3]]

        for line in annotation_line:
            # split each line
            line_content = line.split()
            # open image
            image = Image.open(line_content[0])
            image = cvtColor(image)
            # define image size
            iw, ih = image.size
            # save position of box
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            # flip image or not
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]
            
            nw = nws[index] 
            nh = nhs[index] 

            # -----------------------------------------------------------
            # replaced
            # -----------------------------------------------------------
            # # zoom input image
            # new_ar = w/h
            # scale = rand(scale_low, scale_high)
            # if new_ar < 1:
            #     nh = int(scale * h)
            #     nw = int(nh * new_ar)
            # else:
            #     nw = int(scale * w)
            #     nh = int(nw / new_ar)
            # image = image.resize((nw,nh), Image.BICUBIC)
            # -----------------------------------------------------------

            # place image according to four split images
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255

            index = index + 1
            box_data = []
            # process box
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        # cut image and place together
        # -----------------------------------------------------------
        # replaced
        # -----------------------------------------------------------
        # cutx = np.random.randint(int(w*min_offset_x), 
        #                          int(w*(1 - min_offset_x)))
        # cuty = np.random.randint(int(h*min_offset_y), 
        #                          int(h*(1 - min_offset_y)))
        # -----------------------------------------------------------
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
        
        # cv2 color-space conversion
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(new_image/255,np.float32), cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

        # -----------------------------------------------------------
        # replaced
        # -----------------------------------------------------------
        # new_image = Image.fromarray((image*255).astype(np.uint8))
        # -----------------------------------------------------------
        
        # process box
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        # adjust ground truth bounding box
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes)>0:
            if len(new_boxes)>max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes
        return new_image, box_data

    # -----------------------------------------------------------
    # read .xml files and return y_true
    # -----------------------------------------------------------
    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
        # -----------------------------------------------------------
        # coordinate of box and size of image
        # -----------------------------------------------------------
        true_boxes  = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        # -----------------------------------------------------------
        # three layers of feature map:
        # anchor of feature layer 13x13 is [142, 110], [192, 243], [459, 401]
        # anchor of feature layer 26x26 is [36, 75], [76, 55], [72, 146]
        # anchor of feature layer 52x52 is [12, 16], [19, 36], [40, 28]
        # -----------------------------------------------------------
        num_layers  = len(self.anchors_mask)
        # anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
        # -----------------------------------------------------------
        # m is the number of images
        # -----------------------------------------------------------
        m = true_boxes.shape[0]
        # -----------------------------------------------------------
        # grid_shapes is the shape of grid
        # -----------------------------------------------------------
        grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        # -----------------------------------------------------------
        # y_true has shape (m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
        # -----------------------------------------------------------
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]),
                           5 + num_classes), dtype='float32') for l in range(num_layers)]
        # -----------------------------------------------------------
        # compute center and width, height  of ground truth bounding box
        # center (m, n, 2) width,height (m, n, 2)
        # -----------------------------------------------------------
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh =  true_boxes[..., 2:4] - true_boxes[..., 0:2]
        # -----------------------------------------------------------
        # normalize ground truth to decimal
        # -----------------------------------------------------------
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
        # -----------------------------------------------------------
        # [9,2] -> [1,9,2]
        # -----------------------------------------------------------
        anchors      = np.expand_dims(anchors, 0)
        anchor_max = anchors / 2.
        anchor_min  = - anchor_max
        # width, height valid only when width, height > 0
        valid_mask  = boxes_wh[..., 0] > 0

        for b in range(m):
            # process each frame of image
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            # -----------------------------------------------------------
            # [n,2] -> [n,1,2]
            # -----------------------------------------------------------
            wh        = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins  = - box_maxes

            # -----------------------------------------------------------
            # Intersection over Union (IoU) of ground truth bounding box and anchor box
            # intersect_area  [n,9]
            # box_area        [n,1]
            # anchor_area     [1,9]
            # iou             [n,9]
            # -----------------------------------------------------------
            intersect_mins  = np.maximum(box_mins, anchor_min)
            intersect_maxes = np.minimum(box_maxes, anchor_max)
            intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area    = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)
            # -----------------------------------------------------------
            # [n,] dimension
            # -----------------------------------------------------------
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                # -----------------------------------------------------------
                # find layer of feature map of ground truth
                # -----------------------------------------------------------
                for l in range(num_layers):
                    if n in self.anchor_mask[l]:
                        # ------------------------------
                        # return the floor of the elements of array
                        # find x,y coordinate of the layer of feature map of ground truth bounding box
                        # ------------------------------
                        i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')
                        # kth anchor box of the feature point
                        k = self.anchor_mask[l].index(n)
                        # class of the ground truth bounding box
                        c = true_boxes[b, t, 4].astype('int32')
                        # ------------------------------
                        # y_true has shape (m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                        # where 85 = 4 + 1 + 80
                        #        4 = parameter for width and height adjustment
                        #          = center of box and width, height
                        #        1 = confidence score of boxes
                        #       80 = confidence score of class
                        # ------------------------------
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1

        return y_true
