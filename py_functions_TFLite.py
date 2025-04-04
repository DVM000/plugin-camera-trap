import sys
import os
import numpy as np
import shutil
try:
    import tflite_runtime.interpreter as tflite
except:
    from tensorflow import lite as tflite
import cv2
import time
import logging as log
from config import *

log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO)

class ModelTFLiteClass:
    
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.mu = [103.939, 116.779, 123.68]
        self.std = 1.0
        self.input_shape = None
        self.result_vector = None

    def A_load_CNN(self, model_file):
        """
        Load TFLite model.
        """
        print("[INFO] Loading CNN...")
        
        self.interpreter = tflite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        print("[INFO] Model loaded successfully.")
        
    def process_results(self, results, output_details, labels_file='', N=10):
        """
        Process inference results
        """
        datatype = output_details['dtype']
        #print(datatype)

        # Read labels
        if len(labels_file):
            labels = labels_file
        elif os.path.exists(labels_file):
            labels = np.loadtxt(labels_file, str, delimiter='\t')
        else:
            labels = [str(i) for i in range(1001)]

        # Process results
        if datatype == np.uint8:
            scale, zero_point = output_details['quantization']
            results = scale * (results - zero_point)
            log.info('Quantization. Scale {:.10f} and zero-point {:.10f}'.format(scale, zero_point))

        top_k = results.argsort()[-N:][::-1]

        for i in top_k:
            if datatype == np.float32:
                print('{:.3f} %: {}'.format(float(results[i]) * 100, labels[i]))
            else:
                print('{:.3f} %: {}'.format(float(results[i] / 255.0) * 100, labels[i]))

        return results

    def B_execute_inference(self, folder, classes, Nmax=5, target_classes = [0]):
        """
        Execute model inference.
        """
        if self.interpreter is None:
            raise ValueError("Not loaded model.")
            
        CLASSES = len(classes) > 0
        number_per_class = [0] * (len(classes) if CLASSES else 1001)
        detections = {} # data to be published
        
        for image_name in os.listdir(folder)[:Nmax]:
            image_path = os.path.join(folder, image_name)

            if not os.path.isfile(image_path):
                continue
            
            HWC_img = cv2.imread(image_path)
            img = cv2.resize(HWC_img.astype(np.float32), (self.input_shape[1], self.input_shape[2]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[np.newaxis, :, :, :]
            img[:, :, :, 0] -= self.mu[0]
            img[:, :, :, 1] -= self.mu[1]
            img[:, :, :, 2] -= self.mu[2]
            img /= self.std

            # Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], img.astype(self.input_details[0]['dtype']))
            self.interpreter.invoke()

            # Get results
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            results = np.squeeze(output_data)
            #self.process_results(results, self.output_details[0], labels_file=classes, N=Nmax)
            
            classId = np.argmax(results)
            pred_class = classes[classId]
            confidence = results[classId]
            number_per_class[classId] += 1
            print(f"Image: {image_name}, Prediction: {pred_class}, Probability: {100 * confidence:.2f}%")
            
            if classId in target_classes: # relevant data to be published
                try:     detections[classId].append((image_path, confidence))
                except:  detections[classId] = [(image_path, confidence)]
                
            '''# Move data to folders
            if CLASSES:
                dst_folder = os.path.join(OUTPUT_FOLDER, pred_class)
                os.makedirs(dst_folder, exist_ok=True)
                dst_imgfile = os.path.join(dst_folder, image_name)
                shutil.move(image_path, dst_imgfile)'''
                
            
        self.result_vector = number_per_class
        self.detections = detections

            

