

import cv2
import numpy as np

def getdescriptors(image):
    sift = cv2.SIFT()
    return sift.detectAndCompute(image, None)