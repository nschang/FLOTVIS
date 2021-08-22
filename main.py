# -*- coding: utf-8 -*-
##############################
# python version check
import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3.6.9 is required.")
##############################
# start measurement of execution time
import utils.timing as timing
##############################
import os
import time
from pathlib import Path
# import glob
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from yolo import YOLO

# from keras import backend as K
# from keras.layers import Input
# from keras.models import load_model
# from nets.yolo_net import yolo_body, yolo_process
# from utils.utils import letterbox_image

parser = argparse.ArgumentParser()

parser.add_argument('--img', type=str, default='./test/test.jpg', help='specify image path')
parser.add_argument('--imgdir', type=str, default='./test/', help='specify path to directory with images')
parser.add_argument('--vid', type=str, default='./test/test.mp4',help='specify video path')
parser.add_argument('--saveto', type=str, default='./video-predict/', help='path to save the video. No save if left blank.')
parser.add_argument('--mode', type=str, default='image',
                    choices=['summary','image','batch','video','camera','fps'],
                    help='''
                    "summary" produces summary of network architecture, 
                    "image" predicts single image, 
                    "batch" predicts all images in specified directory, 
                    "video" predicts video, 
                    "camera" predicts using camera feed, 
                    "fps" returns the FPS value''')
args = parser.parse_args()

# this is the folder to save results
if not os.path.exists("./prediction"):
    os.makedirs("./prediction")

if __name__ == "__main__":
    yolo = YOLO()
    #print("Dependencies loaded in --- %s seconds ---" % (time.time() - start_time)) # measurement
    timing.log
    mode = args.mode
    # ------------------------------
    # '--mode=':
    # 'summary' returns summary of network architecture
    # 'image'   to predict single image
    # 'batch'   to predict all images in folder
    # 'video'   for video prediction
    # 'camera'  to predict using camera
    # 'fps'     returns the FPS value
    # ------------------------------
#############SUMMARY##############
    if mode == "summary":
        if __name__ == "__main__":
            from nets.yolo_net import yolo_body
            input_shape     = [416, 416, 3]
            anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            num_classes     = 1
            model           = yolo_body(input_shape, anchors_mask, num_classes)
            model.summary()
            # ------------------------------
            # print out name of each layer
            # ------------------------------
            # for i,layer in enumerate(model.layers):
            #     print(i,layer.name)
#############IMAGE##############
    elif mode == "image":
       while True:
           img     = args.img
           imgpath = Path(args.img)
           #imgpath = Path(r"test/test1.jpg")
           try:
               image = Image.open(img)
           except:
               print('Open Error! Try again!')
               break
           else:
               r_image = yolo.detect_image(image)
               r_image.show()
               r_image.save("./prediction/" + imgpath.stem + '-predict.jpg')
           break
#############BATCH##############
    elif mode == "batch":
        image_ids = [f for f in os.listdir(args.imgdir) if f.endswith(".jpg")]
        for image_id in tqdm(image_ids):
            if image_id.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path  = os.path.join(image_ids, image_id)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                r_image.save(os.path.join("./prediction/", image_id))
#############VIDEO##############   
    elif mode == "video":
        # ------------------------------
        # specify video path
        # ------------------------------
        video_path      = args.vid      # path of the video
        video_save_path = args.saveto   # path to save the video. video_save_path = "" means no save
        video_fps       = 60.0          # set fps of the output video
        #############
        capture=cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            ref,frame=capture.read()                      # read frame
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # BGRtoRGB
            frame = Image.fromarray(np.uint8(frame))      # get img
            frame = np.array(yolo.detect_image(frame))    # predict
            # convert RGB to BGR to work with OpenCV
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        capture.release()
        out.release()
        cv2.destroyAllWindows()
#############CAMERA##############   
    elif mode == "camera":
        # ------------------------------
        # use ctrl+c to save video
        # DO NOT exit directly after prediction
        # ------------------------------
        # specify video path
        video_path      = 0  # video_path=0 means detect using camera
        video_save_path = args.saveto     # path to save the video 
        # video_save_path = "" means no save
        video_fps       = 60.0          # fps of the output video
        #############
        capture=cv2.VideoCapture('test/' + video_path + '.mp4')
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        fps = 0.0
        while(True):
            t1 = time.time()
            ref,frame=capture.read()                      # read frame
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) # BGRtoRGB
            frame = Image.fromarray(np.uint8(frame))      # get img
            frame = np.array(yolo.detect_image(frame))    # predict
            # convert RGB to BGR to work with OpenCV
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()
#############FPS##############
    elif mode == "fps":
        # ------------------------------
        # includes inference, score filtering, non-maximal suppression
        # does NOT include pre-processing (batch normalization and resizing) and draw
        # default test using test/test.jpg
        # note that the FPS when using camera will be lower than this value
        # due to frame limites of camera and preprocessing and draw process
        # ------------------------------
        test_interval = 100
        img = args.img
        image = Image.open(img)
        tact_time = yolo.get_FPS(image, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode:'image','batch','video','camera','fps'")
