import utils
import cv2
import numpy as np

patches = utils.getquerypatches()

for patch in patches:
    print(patch[0])
    img = patch[1]
    cv2.imshow('image', img)
    key = cv2.waitKey(0)
    while key is not 27:
        key = cv2.waitKey(0)
