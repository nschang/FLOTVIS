# -*- coding: utf-8 -*-

import colorsys
import os
import time

import numpy as np
from keras import backend as K
from PIL import Image, ImageFont, ImageDraw

from nets.yolo_net import yolo_body
from utils.utils_anchors import yolo_process
from utils.utils import (cvtColor, get_anchors, get_classes, 
                        preprocess_input, resize_image)

class YOLO(object):
    _defaults = {
        # ------------------------------
        # Both model_path and classes_path need to be updated!
        # check model_path and classes_path if shape mismatch error 
        # ------------------------------
        #"model_path"        : 'train/trained_weights_stage_1.h5',
        "model_path"        : 'model_data/trained_weights_stage_1.h5',
        "classes_path"      : 'model_data/class.txt',
        # text file with anchor box data
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        # point to respective anchor box
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        # only keep achor boxes with confidence higher than this score
        "score"             : 0.1,
        # size of iOU in non-maximal suppression
        "iou"               : 0.3,
        # max number of boxes
        "max_boxes"         : 100,
        # choose between (416,416) or (608,608) depending on RAM size
        "input_shape"  : (608, 608), # must be multiple of 32
        # ------------------------------
        # toggle letterbox_image to resize input without distortion
        # effect UNTESTED
        # ------------------------------
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ------------------------------
    # initialize yolo
    # ------------------------------
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # get class
        self.class_names, self.num_classes = get_classes(self.classes_path)
        # get anchors
        self.anchors, self.num_anchors        = get_anchors(self.anchors_path)

        # assign colors to boxes
        hsv_tuples  = [(x / self.num_classes, 1., 1.)
                      for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), 
                                          int(x[1] * 255), 
                                          int(x[2] * 255)),
                                          self.colors
                                          ))
        self.input_image_shape = K.placeholder(shape=(2, ))

        self.sess                             = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
    # ------------------------------
    # load class and pre-trained model
    # ------------------------------
    def generate(self):
        model_path  = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # get class and number of anchor boxes
        # num_anchors = len(self.anchors)
        # num_classes = self.num_classes

        # load model
        self.yolo_model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # randomize color
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        # ------------------------------
        # yolo_process handles post-processing of detection result
        # which includes Decoding, Non-Maximum Suppression (NMS),
        # Thresholding, etc.
        # ------------------------------
        boxes, scores, classes  = yolo_process(
            self.yolo_model.output, 
            self.anchors,
            self.num_classes,
            self.input_image_shape,
            self.input_shape,
            max_boxes       = self.max_boxes,
            score_threshold = self.score, 
            iou_threshold   = self.iou, 
            letterbox_image = self.letterbox_image
        )
        return boxes, scores, classes

# ------------------------------
# detect image
# ------------------------------
    def detect_image(self, image):
        # ------------------------------
        # # BATCH DETECT MODULE 
        # write list of target images to a new txt file
        # f = open("./test/"+ image_id +".txt","w")
        # ------------------------------
        # cv2 convert to RGB to prevent grayscale error at prediction
        image = cvtColor(image)
        # resize, add gray bar to sides of image
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # add batch_size dimension and normalize
        image_data      = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        # ------------------------------
        # load image to grid and predict
        # ------------------------------
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict = {
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # initial font size
        font_size = 1
        font_path = 'model_data/ArialUnicode.ttf'
        # set font
        font = ImageFont.truetype(font = font_path,
                     size=np.floor(3e-2 * image.size[1] + font_size).astype('int32'))
        # set thickness of prediction box
        thickness = max((image.size[0] + image.size[1]) // 666, 1)
        # draw image
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]

            # coordinate of prediction box
            top, left, bottom, right = box
            # ------------------------------
            # BATCH DETECT MODULE
            # f.write("%s %s %s %s %s %s\n" % (predicted_class, str(score)[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
            # ------------------------------
            top    = top - 5
            left   = left - 5
            bottom = bottom + 5
            right  = right + 5

            top    = max(0, np.floor(top + 0.5).astype('int32'))
            left   = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right  = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # draw boxes
            label = '{} {:.2f}'.format(predicted_class, score)
            global draw
            draw  = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            # draw prediction on each object label
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        # ------------------------------
        # draw sum of detected objects above original image
        # ------------------------------
        # plastic count:
        # (0,0)------(w,0)
        # |              |
        # |              |
        # |              |
        # |              |
        # (0,h)------(w,h)
        # ------------------------------
        image_w, image_h = image.size # width and height of original image
        image = image.crop((0, - 5e-2 * image_h, image_w, image_h))
        draw2 = ImageDraw.Draw(image) 
        # draw2.text((5, 0), "plastic count: " + str(out_boxes.shape[0]), (255, 255, 255), font=font)
        # ------------------------------
        # draw FPS
        # ------------------------------
        # includes inference, score filtering, non-maximal suppression
        # does NOT include pre-processing (batch normalization and resizing) and draw
        # default test using test/test.jpg
        # note that the FPS when using camera will be lower than this value
        # due to frame limites of camera and preprocessing and draw process
        # ------------------------------
        test_interval = 1
        t3 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0})
        t4 = time.time()
        tact_time = (t4 - t3)
        inference_time = "{:.2f}".format(tact_time)
        frame_per_second = "{:.2f}".format(1/tact_time)
        draw2.text((5, 0), 
        "plastic count: " + str(out_boxes.shape[0])
        + "    FPS: " + frame_per_second
        + "    Predicted in " + inference_time + " seconds"
        , (255, 255, 255), font=font)
        # ------------------------------
        # IoU threshold = 50 %, used Area-Under-Curve for each unique Recall
        # mean average precision (MAP@0.50) = 0.756655, or 75.67 %
        # Total Detection Time: Seconds
        # ------------------------------
        # beautify output
        # ------------------------------
        # from random import randint
        # make_color = lambda : (randint(50, 255), randint(50, 255), randint(50,255))
        # z = 300
        # for c in "plastic count: ":
        #     draw2.text((z, 5), c, make_color())
        #     z = z + 12
        # ------------------------------
        # draw translucent box to top left corner of image and 
        # draw total object count on to the box
        # ------------------------------
        # draw1 = ImageDraw.Draw(image, "RGBA")
        # draw1.rectangle(((0, 0), (158, 36)), fill=(200, 200, 200, 66))
        # draw.text((5, 5), "plastic count: " + str(out_boxes.shape[0]), (0, 0, 0), font=font)
        # del draw1
        #del draw2
        #del draw
        return image
        # ------------------------------
        # BATCH DETECT MODULE
        # f.close()
        # return
        # ------------------------------

    # ------------------------------
    # get FPS
    # ------------------------------
    # includes inference, score filtering, non-maximal suppression
    # does NOT include pre-processing (batch normalization and resizing) and draw
    # default test using test/test.jpg
    # note that the FPS when using camera will be lower than this value
    # due to frame limites of camera and preprocessing and draw process
    # ------------------------------
    def get_FPS(self, image, test_interval):
        # cv2 convert to RGB to prevent grayscale error at prediction
        image = cvtColor(image)
        # resize, add gray bar to sides of image
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # add batch_size dimension and normalize
        image_data      = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        # ------------------------------
        # load image to grid and predict
        # ------------------------------
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0})
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
    
    # ------------------------------
    # get mAP
    # ------------------------------
    def get_map_txt(self, image_id, image, class_names, MAP_OUT_PATH):
        f = open(os.path.join(MAP_OUT_PATH, "detection-results/" + image_id + ".txt"),"w") 
        # cv2 convert to RGB to prevent grayscale error at prediction
        image       = cvtColor(image)
        # resize, add gray bar to sides of image
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        # ------------------------------
        # add batch_size dimension and normalize
        # ------------------------------
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        for i, c in enumerate(out_classes):
            predicted_class             = self.class_names[int(c)]
            score                       = str(out_scores[i])
            top, left, bottom, right    = out_boxes[i]
            if predicted_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
        f.close()
        return 

    def close_session(self):
        self.sess.close()
