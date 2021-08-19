# -----------------------------------------------------------
# modified from https://github.com/Cartucho/mAP
# -----------------------------------------------------------
import os
import sys
import glob
import json
import shutil
import operator
import argparse
import numpy as np
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
    map_visualize   = False         # toggle visualization of VOC_map
    VOCdevkit_PATH = 'VOCdevkit'    # Folder with annotated images
    MAP_OUT_PATH    = 'mAP_out'     # where to save mAP output
    # -----------------------------------------------------------

    image_ids = open(os.path.join(VOCdevkit_PATH, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(MAP_OUT_PATH):
        os.makedirs(MAP_OUT_PATH)
    if not os.path.exists("./val"):
        os.makedirs("./val")
    if not os.path.exists("./val/ground-truth"):
        os.makedirs("./val/ground-truth")

    parser = argparse.ArgumentParser()
    parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    # argparse receiving list of classes to be ignored
    parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    # argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
    args = parser.parse_args()

    """
        0,0 ------> x (width)
        |
        |  (Left,Top)
        |      *_________
        |      |         |
                |         |
        y      |_________|
    (height)            *
                    (Right,Bottom)
    """
    # if there are no classes to ignore then replace None by empty list
    if args.ignore is None:
        args.ignore = []

    specific_iou_flagged = False
    if args.set_class_iou is not None:
        specific_iou_flagged = True

    # make sure that the cwd() is the location of the python script (so that every path makes sense)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # -----------------------------------------------------------
    # set path to results on validation set
    # -----------------------------------------------------------
    GT_PATH = os.path.join(os.getcwd(), 'val', 'ground-truth')
    DR_PATH = os.path.join(os.getcwd(), 'val', 'detection-results')
    # if there are no images then no animation can be shown
    IMG_PATH = os.path.join(os.getcwd(), 'val', 'images-optional')

    if os.path.exists(IMG_PATH): 
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                # no image files found
                args.no_animation = True
    else:
        args.no_animation = True

    # try to import OpenCV if the user didn't choose the option --no-animation
    show_animation = False
    if not args.no_animation:
        try:
            import cv2
            show_animation = True
        except ImportError:
            print("\"opencv-python\" not found, please install to visualize the results.")
            args.no_animation = True

    # try to import Matplotlib if the user didn't choose the option --no-plot
    draw_plot = False
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            draw_plot = True
        except ImportError:
            print("\"matplotlib\" not found, please install it to get the resulting plots.")
            args.no_plot = True

    # -----------------------------------------------------------
    #  Create a ".temp_files/" and "results/" directory
    # -----------------------------------------------------------
    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)
    RESULTS_FILES_PATH = "results"
    if os.path.exists(RESULTS_FILES_PATH): # if it exist already
        # reset the results directory
        shutil.rmtree(RESULTS_FILES_PATH)

    os.makedirs(RESULTS_FILES_PATH)
    if draw_plot:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "AP"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))
    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "images", "detections_one_by_one"))

    # -----------------------------------------------------------
    #  ground-truth
    #      Load each of the ground-truth files into a temporary ".json" file.
    #      Create a list of all the class names present in the ground-truth (gt_classes).
    # -----------------------------------------------------------
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}




        # dump bounding_boxes into a ".json" file
        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    #print(gt_classes)
    #print(gt_counter_per_class)

    
    for image_id in image_ids:
        with open(os.path.join(MAP_OUT_PATH, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_PATH , "VOC2007/Annotations/" + image_id + ".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                # -----------------------------------------------------------
                # uncomment the following section if other pre-defined labels 
                # are present in annotation file
                # -----------------------------------------------------------
                classes_path = 'model_data/class.txt'
                class_names = get_classes(classes_path)
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
    print("Got ground truth result.")