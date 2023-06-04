# process a test video and prints FPS
import math
import random
import os
import cv2
import numpy as np
import time
import argparse  
import configparser
from ultralytics import YOLO
import draw_boxes as Boxes

video_file = 'vid.mp4'
video_file_out = 'vid_res.mp4'
model = YOLO('best_m.pt')

vid = cv2.VideoCapture(video_file)
fps = int(round(vid.get(cv2.CAP_PROP_FPS)))
w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
out_w = w
out_h = h  
out = cv2.VideoWriter(video_file_out, fourcc, fps, (out_w, out_h), True)

print( "image size: " , w, " x ", h)

np.set_printoptions(suppress=True)

frame = 0
average_fps = 0.0
while True:
    ret, img = vid.read()
    if not ret:
        break
        
    start = time.time()
    # output = model.track(img)
    output = model.predict(img, conf=0.25)
    end = time.time()
    
    frame = frame + 1
    fps = 1.0/(end-start)
    print("%.1f" % fps)
    
    average_fps += fps
    
    Boxes.drawBoxes( img, output[0].boxes.data, score=True)

    out.write(img)
    
print("Average FPS: %.1f" % (average_fps / frame) )

vid.release()
out.release()
