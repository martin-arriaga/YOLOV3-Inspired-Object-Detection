import torch
import cv2
import numpy as np

"""
    Pre-processing image to maintain aspect ratio to prevent distortion 
    returns scale and new dimensions 
"""

def preprocess(image, img_size = 416):
    h,w,_ = image.shape
    scale = img_size / max(h,w)
    nh, nw = int(scale* h) , int(scale * w)
    image_resized = cv2.resize(image, (nw, nh))
    canvas = np.zeros((img_size, img_size, 3), dtype=np.float32)
    canvas[:nh, :nw, : ] = image_resized / 255.
    image_tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)
    return image_tensor,scale , (nw, nh)
