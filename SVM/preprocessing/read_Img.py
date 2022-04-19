import os
import cv2
import numpy as np

def ReadImg(dirPath):
    img_list = []
    for file in os.listdir(dirPath):
        if os.path.isfile(os.path.join(dirPath, file)) == True:
            c = os.path.basename(file)
            name = dirPath + '/' + c
            img = cv2.imread(name, 0)
            # resize the image shape into 20*20
            img.resize((20,20))
            # a row stands for an image
            img = img.flatten()
            img_list.append(img)
    return img_list

