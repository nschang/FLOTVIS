#!/usr/bin/env python
# coding=utf-8
#import colorsys
#import os
#from os import listdir
#from os.path import isfile, join
#from pathlib import Path
#import numpy as np
import glob
import time
import argparse
import cv2
import numpy as np
#from tqdm import tqdm
from PIL import Image
#from keras import backend as K
#from keras.layers import Input
#from keras.models import load_model
from yolo import YOLO
#from nets.yolo4 import yolo_body, yolo_eval
#from utils.utils import letterbox_image

parser = argparse.ArgumentParser()

parser.add_argument('--img', type=str, default='./test/test.jpg')
parser.add_argument('--imgdir', type=str, default='./test/')
parser.add_argument('--vid', type=str, default='./test/test.mp4')
parser.add_argument('--saveto', type=str, default='./video-predict/', help='path to save the video. Leave blank for no save.')
parser.add_argument('--mode', type=str, default='image',
                    choices=['image','batch','video','camera','fps'],
                    help='mode: "image" to predict single image, "batch" to predict all images in folder, "video" for video prediction, "fps" returns the FPS value')
args = parser.parse_args()

# this is the folder to save results
if not os.path.exists("./prediction"):
    os.makedirs("./prediction")

if __name__ == "__main__":
    yolo = YOLO()
    mode = args.mode
    ################################
    # '--mode=':
    # 'image'   to predict single image
    # 'batch'   to predict all images in folder
    # 'video'   for video prediction
    # 'camera'  to predict using camera
    # 'fps'     returns the FPS value
#############IMAGE##############
    if mode == "image":
       while True:
           img = args.img
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
        # return a list of images in test folder
        image_ids = [f for f in os.listdir('./test/') if f.endswith(".jpg")]
        for image_id in tqdm(image_ids):
            image_path = "./test/"+image_id
            image = Image.open(image_path)
            yolo.detect_batch(image_id,image)
            r_image = yolo.detect_image(image_id,image)
            image.save("./prediction/detected-"+image_id)
            r_image.save("./prediction/detected-"+image_id)
        print("Conversion completed!")

        imdir = args.imgdir
        ext = ['jpg', 'png']
        files = []
        [files.extend(glob.glob(imdir + '*.' + t)) for t in ext]
        images = [cv2.imread(file) for file in files]

#############VIDEO##############   
    elif mode == "video":
        ################################
        # use ctrl+c to save video
        # DO NOT exit directly after prediction
        ################################
        # specify video path
        video_path      = args.vid    # path of the video. 
        video_save_path = args.saveto # path to save the video. video_save_path="" means no save
        video_fps       = 60.0        # fps of the saved video
        ################################
        capture=cv2.VideoCapture(video_path)
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
#############CAMERA##############   
    elif mode == "camera":
        ################################
        # use ctrl+c to save video
        # DO NOT exit directly after prediction
        ################################
        # specify video path
        video_path      = 0  # path of the video. video_path=0 means detect using camera
        video_save_path = "./video"     # path to save the video. video_save_path="" means no save
        video_fps       = 60.0          # fps of the saved video
        ################################
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
        test_interval = 100
        img = args.img
        image = Image.open('test/' + img + '.jpg')
        tact_time = yolo.get_FPS(image, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode:'image','batch','video','camera','fps'")
