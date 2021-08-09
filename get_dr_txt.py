'''
# test trained model with validation set and
# save in ./results
'''
import colorsys
import os

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from PIL import Image
from tqdm import tqdm

from nets.yolo4 import yolo_body, yolo_eval
from utils.utils import letterbox_image
from yolo import YOLO


class mAP_YOLO(YOLO):
    '''
    # get all classes
    '''
    def generate(self):
        self.score = 0.01
        self.iou = 0.5
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        '''
        # get class and number of anchor boxes
        '''
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        '''
        # load model if present, otherwise create model then load
        '''
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

        # assign different colors to boxes
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

        '''
        # yolo_eval handles post-processing of detection result
        # which includes Decoding, Non-Maximum Suppression (NMS), 
        # Thresholding, etc.
        '''
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                num_classes, self.input_image_shape, max_boxes = self.max_boxes,
                score_threshold = self.score, iou_threshold = self.iou, letterbox_image = self.letterbox_image)
        return boxes, scores, classes

    '''
    # predict
    '''
    def detect_image(self, image_id, image):
        f = open("./val/detection-results/"+image_id+".txt","w") 
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        '''
        # add batch_size dimension
        '''
        image_data = np.expand_dims(image_data, 0)

        '''
        # load image to grid and predict
        '''
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})
        for i, c in enumerate(out_classes):
            predicted_class = self.class_names[int(c)]
            score = str(out_scores[i])

            top, left, bottom, right = out_boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

yolo = mAP_YOLO()

image_ids = open('VOCdevkit/VOC2007/ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./val"):
    os.makedirs("./val")
if not os.path.exists("./val/detection-results"):
    os.makedirs("./val/detection-results")
if not os.path.exists("./val/images-optional"):
    os.makedirs("./val/images-optional")

for image_id in tqdm(image_ids):
    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # visualize mean Average Precision (mAP) and save to val/images-optional
    image.save("./val/images-optional/"+image_id+".jpg")
    yolo.detect_image(image_id,image)
    
print("Conversion completed!")
