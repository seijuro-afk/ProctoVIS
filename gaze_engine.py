import numpy as np
from l2cs import Pipeline
import torch
import os
import cv2

class GazeTracker:
    def __init__(self, weights_path="models/L2CSNet_gaze360.pkl"):
        self.weights_path = str(weights_path).replace('\\', '/')
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.pipeline = Pipeline(
            weights=self.weights_path, 
            arch='ResNet50',
            device=self.device
        )

    def estimate_from_crop(self, filepath):
        try:
            # 1. Read the image natively in Python (solves the scrambling issue)
            img = cv2.imread(str(filepath))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 2. Pass to L2CS-Net
            results = self.pipeline.step(img_rgb)
            
            if results.pitch.shape[0] > 0:
                pitch_rad = float(results.pitch[0])
                yaw_rad = float(results.yaw[0])
                return yaw_rad, pitch_rad
            else:
                return 0.0, 0.0
                
        except Exception as e:
            return 0.0, 0.0