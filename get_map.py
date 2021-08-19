# -----------------------------------------------------------
# modified from https://github.com/Cartucho/mAP
# -----------------------------------------------------------
import os
import argparse
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

if __name__ == "__main__":
    # -----------------------------------------------------------
    # set MINOVERLAP = 0.75 for mAP0.75
    # -----------------------------------------------------------
    MINOVERLAP = 0.5 
    # -----------------------------------------------------------
    # set variables
    # -----------------------------------------------------------
    MAP_VISUALIZE   = True         # toggle visualization of VOC_map
    VOCDEVKIT_PATH  = 'VOCdevkit'   # folder with annotated images
    MAP_OUT_PATH    = 'mAP_out'     # folder to save mAP output
    # -----------------------------------------------------------
    # define object classes
    # -----------------------------------------------------------
    classes_path = 'model_data/class.txt'
    class_names, _ = get_classes(classes_path)

    image_ids = open(os.path.join(VOCDEVKIT_PATH, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(MAP_OUT_PATH):
        os.makedirs(MAP_OUT_PATH)
    if not os.path.exists(os.path.join(MAP_OUT_PATH, 'ground-truth')):
        os.makedirs(os.path.join(MAP_OUT_PATH, 'ground-truth'))
    if not os.path.exists(os.path.join(MAP_OUT_PATH, 'detection-results')):
        os.makedirs(os.path.join(MAP_OUT_PATH, 'detection-results'))
    if not os.path.exists(os.path.join(MAP_OUT_PATH, 'images-optional')):
        os.makedirs(os.path.join(MAP_OUT_PATH, 'images-optional'))

    parser = argparse.ArgumentParser()
    parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    # argparse receiving list of classes to be ignored
    parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    # argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
    args = parser.parse_args()

    #    0,0 ------> x (width)
    #    |
    #    |  (Left,Top)
    #    |      *_________
    #    |      |         |
    #           |         |
    #    y      |_________|
    # (height)            *
    #                 (Right,Bottom)
    
    # -----------------------------------------------------------
    # get prediction
    # -----------------------------------------------------------
    print("Load model.")
    yolo = YOLO(confidence = 0.001, nms_iou = 0.5)
    print("Load model done.")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path  = os.path.join(VOCDEVKIT_PATH, "VOC2007/JPEGImages/" + image_id + ".jpg")
        image       = Image.open(image_path)
        if MAP_VISUALIZE:
            image.save(os.path.join(MAP_OUT_PATH, "images-optional/" + image_id + ".jpg"))
        yolo.get_map_txt(image_id, image, class_names, MAP_OUT_PATH)
    print("Got prediction.")

    # -----------------------------------------------------------
    # get ground truth
    # -----------------------------------------------------------
    print("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(MAP_OUT_PATH, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCDEVKIT_PATH, "VOC2007/Annotations/" + image_id + ".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                # if other pre-defined labels are present in annotation file
                if obj_name not in class_names:
                    continue
                bndbox  = obj.find('bndbox')
                left    = bndbox.find('xmin').text
                top     = bndbox.find('ymin').text
                right   = bndbox.find('xmax').text
                bottom  = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Got ground truth.")

    # -----------------------------------------------------------
    # get mAP
    # -----------------------------------------------------------
    print("Get map.")
    get_map(MINOVERLAP, True, path = MAP_OUT_PATH)
    print("Got mAP.")

    # -----------------------------------------------------------
    # get mAP using pycocotools
    # -----------------------------------------------------------
    print("Get COCO map.")
    get_coco_map(class_names = class_names, path = MAP_OUT_PATH)
    print("Got COCO mAP.")