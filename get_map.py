# -----------------------------------------------------------
# modified from https://github.com/Cartucho/mAP
# -----------------------------------------------------------
import os
import glob
import json
import shutil
import operator
import sys
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -----------------------------------------------------------
# set MINOVERLAP = 0.75 for mAP0.75
# -----------------------------------------------------------
MINOVERLAP = 0.5 

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

for txt_file in ground_truth_files_list:
    #print(txt_file)
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    # check if there is a correspondent detection-results file
    temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
    if not os.path.exists(temp_path):
        error_msg = "Error. File not found: {}\n".format(temp_path)
        error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
        error(error_msg)
    lines_list = file_lines_to_list(txt_file)
    # create ground-truth dictionary
    bounding_boxes = []
    is_difficult = False
    already_seen_classes = []
    for line in lines_list:
        try:
            if "difficult" in line:
                class_name, left, top, right, bottom, _difficult = line.split()
                is_difficult = True
            else:
                class_name, left, top, right, bottom = line.split()
                    
        except:
            if "difficult" in line:
                line_split = line.split()
                _difficult = line_split[-1]
                bottom = line_split[-2]
                right = line_split[-3]
                top = line_split[-4]
                left = line_split[-5]
                class_name = ""
                for name in line_split[:-5]:
                    class_name += name + " "
                class_name = class_name[:-1]
                is_difficult = True
            else:
                line_split = line.split()
                bottom = line_split[-1]
                right = line_split[-2]
                top = line_split[-3]
                left = line_split[-4]
                class_name = ""
                for name in line_split[:-4]:
                    class_name += name + " "
                    # check if class is in the ignore list, if yes skip
                class_name = class_name[:-1]
        if class_name in args.ignore:
            continue
        bbox = left + " " + top + " " + right + " " +bottom
        if is_difficult:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
        else:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if no class exists yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if no class exists yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)


    # dump bounding_boxes into a ".json" file
    with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
        json.dump(bounding_boxes, outfile)

gt_classes = list(gt_counter_per_class.keys())
# sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)
#print(gt_classes)
#print(gt_counter_per_class)

# # -----------------------------------------------------------
# #  Check format of the flag --set-class-iou (if used)
# #     e.g. check if class exists
# # -----------------------------------------------------------
# if specific_iou_flagged:
#     n_args = len(args.set_class_iou)
#     error_msg = \
#         '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
#     if n_args % 2 != 0:
#         error('Error, missing arguments. Flag usage:' + error_msg)
#     # [class_1] [IoU_1] [class_2] [IoU_2]
#     # specific_iou_classes = ['class_1', 'class_2']
#     specific_iou_classes = args.set_class_iou[::2] # even
#     # iou_list = ['IoU_1', 'IoU_2']
#     iou_list = args.set_class_iou[1::2] # odd
#     if len(specific_iou_classes) != len(iou_list):
#         error('Error, missing arguments. Flag usage:' + error_msg)
#     for tmp_class in specific_iou_classes:
#         if tmp_class not in gt_classes:
#                     error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
#     for num in iou_list:
#         if not is_float_between_0_and_1(num):
#             error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

    # -----------------------------------------------------------
    #  detection-results
    #  Load each of the detection-results files into a temporary ".json" file
    # -----------------------------------------------------------
    # get a list with the detection-results files
    # dr_files_list = glob.glob(DR_PATH + '/*.txt')
    # dr_files_list.sort()

    # for class_index, class_name in enumerate(gt_classes):
        # bounding_boxes = []
        # for txt_file in dr_files_list:
        #     #print(txt_file)
        #     # the first time it checks if all the corresponding ground-truth files exist
        #     file_id = txt_file.split(".txt",1)[0]
        #     file_id = os.path.basename(os.path.normpath(file_id))
        #     temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
        #     if class_index == 0:
        #         if not os.path.exists(temp_path):
        #             error_msg = "Error. File not found: {}\n".format(temp_path)
        #             error(error_msg)
        #     lines = file_lines_to_list(txt_file)
        #     for line in lines:
        #         try:
        #             tmp_class_name, confidence, left, top, right, bottom = line.split()
        #         except:
        #             line_split     = line.split()
        #             bottom         = line_split[-1]
        #             right          = line_split[-2]
        #             top            = line_split[-3]
        #             left           = line_split[-4]
        #             confidence     = line_split[-5]
        #             tmp_class_name = ""
        #             for name in line_split[:-5]:
        #                 tmp_class_name += name + " "
        #             tmp_class_name = tmp_class_name[:-1]

        #         if tmp_class_name == class_name:
        #             #print("match")
        #             bbox = left + " " + top + " " + right + " " +bottom
        #             bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
        #             #print(bounding_boxes)
        # # sort detection-results by decreasing confidence
        # bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        # with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
        #     json.dump(bounding_boxes, outfile)

# # -----------------------------------------------------------
# #  Calculate the AP for each class
# # -----------------------------------------------------------
# sum_AP = 0.0
# ap_dictionary = {}
# lamr_dictionary = {}
# # open file to store the results
# with open(RESULTS_FILES_PATH + "/results.txt", 'w') as results_file:
#     results_file.write("# AP and precision/recall per class\n")
#     count_true_positives = {}

# # -----------------------------------------------------------
# #  Count total of detection-results
# # -----------------------------------------------------------
# # iterate through all the files
# det_counter_per_class = {}
# for txt_file in dr_files_list:
#     # get lines to list
#     lines_list = file_lines_to_list(txt_file)
#     for line in lines_list:
#         class_name = line.split()[0]
#         # check if class is in the ignore list, if yes skip
#         if class_name in args.ignore:
#             continue
#         # count that object
#         if class_name in det_counter_per_class:
#             det_counter_per_class[class_name] += 1
#         else:
#             # if class didn't exist yet
#             det_counter_per_class[class_name] = 1
# #print(det_counter_per_class)
# dr_classes = list(det_counter_per_class.keys())

# # -----------------------------------------------------------
# #  Write number of ground-truth objects per class to results.txt
# # -----------------------------------------------------------
# with open(RESULTS_FILES_PATH + "/results.txt", 'a') as results_file:
#     results_file.write("\n# Number of ground-truth objects per class\n")
#     for class_name in sorted(gt_counter_per_class):
#         results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")
