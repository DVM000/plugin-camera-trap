#https://github.com/waggle-sensor/plugin-imagesampler/blob/main/app.py

from datetime import datetime, timezone
import logging
import time
import os
import argparse
from multiprocessing import Process
import shutil

from waggle.plugin import Plugin
from waggle.data.vision import Camera


import numpy as np
from PIL import Image



from config import *
from py_functions_TFLite import ModelTFLiteClass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')

def capture_stream(plugin, stream, fps, nframes, out_dir=""):
    os.makedirs(out_dir, exist_ok=True)
    
    # use case 2
    with Camera() as camera:
        i=0
        for sample in camera.stream():
            # Save image
            sample_path = os.path.join(out_dir, datetime.now().astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S%z')+f"-{i:02d}.jpg")
            sample.save(sample_path.replace(':',''))

            time.sleep(1.0/fps)
            i = i+1
            
            if i > nframes: 
                break
	    
    print(f"Captured {nframes} images")
   
   
def capture(plugin, stream, fps, nframes, out_dir=""):
    os.makedirs(out_dir, exist_ok=True)
   
    # use case 1
    for i in range(nframes):
	    # Capture image
	    try:
	        sample = Camera().snapshot()
	    except:
	        print(f"Error capturing image. Simulating.")
	        sample = np.random.rand(100,100,3) * 255
	        sample = Image.fromarray(sample.astype('uint8'))
	    
	    # Save image
	    sample_path = os.path.join(out_dir, datetime.now().astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S%z')+f"-{i:02d}.jpg")
	    sample.save(sample_path)
    
	    time.sleep(1.0/fps)
	    
    print(f"Captured {nframes} images")
     
            
        
def main(args):

    print(f'[INFO] Starting date {datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}')
    FRAMES_FOLDER = "OUTPUT/" #args.out_dir
    NFRAMES = args.n
    
    os.makedirs(FRAMES_FOLDER, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Load CNN
    # ------------------------------------------------------------------
    print(f'[INFO] Loading model')
    modelObj = ModelTFLiteClass()
    modelObj.A_load_CNN(CNN_FILE)

    with open(LABEL_FILE, "r") as file:
      classes = [line.strip() for line in file]

    
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    
    # 1- Capture frames
    with Plugin() as plugin:
            capture_stream(plugin, args.stream, FPS, NFRAMES, FRAMES_FOLDER)
            #capture(plugin, args.stream, FPS, NFRAMES, FRAMES_FOLDER)
                     
    # 2- CNN classification
    dt_s = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    modelObj.B_execute_inference(FRAMES_FOLDER, classes, 1000, target_classes = TARGET_LABELS)
    result_classes = modelObj.result_vector
    print(result_classes)

    
    # 3- Publish detections 
    meta = {"camera":  f"{args.stream}"}
    with Plugin() as plugin:
        for c,detections in modelObj.detections.items(): # tuple (image_path,confidence) indexed by class-idx
            for det in detections:
                plugin.publish(f'env.detection.animal', float(det[1]), timestamp=int(os.path.getmtime(det[0])*1e9), meta=meta) #timestamp=det[0].timestamp
                plugin.upload_file(det[0], timestamp=int(os.path.getmtime(det[0])*1e9), meta=meta)
	
    shutil.rmtree(FRAMES_FOLDER)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument(
    #    '--out-dir', dest='out_dir',
    #    action='store', default="OUTPUT/", type=str,
    #    help='Path to save images locally in %%Y-%%m-%%dT%%H:%%M:%%S%%z.jpg format')
    parser.add_argument(
        '--n', 
        action='store', default=5, type=int,
        help='Number of frames to capture')
    parser.add_argument(
        '--stream', dest='stream',
        help='ID or name of a stream', default='node-cam')

       
    args = parser.parse_args()
    exit(main(args))
