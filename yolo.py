import colorsys
import os
import time

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from nets.yolo4 import yolo_body, yolo_eval
from utils.utils import letterbox_image

from random import randint


################################
# Both model_path and classes_path need to be updated!
# in case of shape mismatch error during training,
# check model_path and classes_path .
################################
#class C_YOLO(YOLO):
class YOLO(object):
    _defaults = {
        #"model_path"        : 'train/trained_weights_stage_1.h5',
        "model_path"        : 'model_data/trained_weights_stage_1.h5',
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "classes_path"      : 'model_data/class.txt',
        "score"             : 0.1,
        "iou"               : 0.3,
        "max_boxes"         : 100,
        "model_image_size"  : (608, 608), # use (416,416) or (608,608) depending on RAM size
        ################################
        # toggle letterbox_image to resize input without distortion
        ################################
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    ################################
    # init yolo
    ################################
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    ################################
    # get class
    ################################
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    ################################
    # get anchor boxes
    ################################
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    ################################
    # load classes and pre-trained model
    ################################
    def generate(self):
        #self.score = 0.01
        #self.iou = 0.5
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # get class and number of anchor boxes
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        ################################
        # load model if present
        # otherwise create model first
        ################################
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # assign box colors
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # randomize color
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        self.input_image_shape = K.placeholder(shape=(2, ))

        ################################
        # yolo_eval handles post-processing of detection result
        # which includes Decoding, Non-Maximum Suppression (NMS),
        # Thresholding, etc.
        ################################
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                num_classes, self.input_image_shape, max_boxes = self.max_boxes,
                score_threshold = self.score, iou_threshold = self.iou, letterbox_image = self.letterbox_image)
        return boxes, scores, classes

    ################################
    # detect image
    ################################
    def detect_image(self, image):
        # convert to RGB image to prevent error from grayscale image
        image = image.convert('RGB')

        # resize by adding gray bar to image
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # add batch_size dimension
        image_data = np.expand_dims(image_data, 0)

        # load image to grid and detect
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # set font size
        font_size = 8
        # set font
        font = ImageFont.truetype(font='model_data/ArialUnicode.ttf',
                    size=np.floor(3e-2 * image.size[1] + font_size).astype('int32'))

        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            # coordinate of predict box
            top, left, bottom, right = box
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # draw boxes
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
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
            ################################
            # draw box
            ################################
            draw1 = ImageDraw.Draw(image, "RGBA")
            draw1.rectangle(((0, 0), (158, 36)), fill=(200, 200, 200, 66))
            #draw1.rectangle(((280, 10), (1010, 706)), outline=(0, 0, 0, 127), width=3)
            ################################
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            draw.text((5, 5), "plastic count: " + str(out_boxes.shape[0]), (0, 0, 0), font=font) # comment out to beautify
            ################################
            # beautify output (optional)
            ################################
            # make_color = lambda : (randint(50, 255), randint(50, 255), randint(50,255))
            # z = 5
            # for c in "plastic count: ":
            #     draw.text((z, 5), c, make_color())
            #     z = z + 12
            ################################
            del draw

        return image
        img.save('prediction.jpg')

    ################################
    # batch detect image in folder
    ################################
    def detect_batch(self, image_id, image):
        # write classes to a new txt file
        f = open("./test/"+image_id+".txt","w")
        # convert to RGB image to prevent error from grayscale image
        image = image.convert('RGB')

        # resize by adding gray bar to image
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # add batch_size dimension
        image_data = np.expand_dims(image_data, 0)

        # load image to grid and detect
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        # set font
        font = ImageFont.truetype(font='model_data/ArialUnicode.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))+1

        thickness = max((image.size[0] + image.size[1]) // 300, 1)

        #for i, c in enumerate(out_classes):
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box = out_boxes[i]
            score = out_scores[i]

            # coordinate of predict box
            top, left, bottom, right = box
            f.write("%s %s %s %s %s %s\n" % (predicted_class, str(score)[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # draw boxes
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
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
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            draw.text((5, 5), "plastic count: " + str(out_boxes.shape[0]), (255, 255, 255), font=font)
            del draw

        return image
        img.save('prediction.jpg')
        f.close()
        return

    ################################
    # get FPS
    ################################
    def get_FPS(self, image, test_interval):
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

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

    def close_session(self):
        self.sess.close()
