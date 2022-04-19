import numpy as np
import cv2
import os

def ReFileName(dirPath,flag):
    img_list = []
    for file in os.listdir(dirPath):
        if os.path.isfile(os.path.join(dirPath, file)) == True:
            c = os.path.basename(file)
            name = dirPath + '/' + c
            # print('name',name)
            img = cv2.imread(name, 0)
            # print('img',img)
            img = img.flatten()
            if flag == 1:
                img = np.append(img, 1)
            else:
                img = np.append(img, -1)
            img_list.append(img)
    return img_list