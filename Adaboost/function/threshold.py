import cv2
import numpy as np
from function.harr_f import *
from function.read_Img import *
import operator
from functools import reduce
import matplotlib.pyplot as plt


def count_elements(seq) -> dict:
    hist = {}
    for i in seq:
        hist[i] = hist.get(i, 0) + 1
    return hist


def get_same(dic1,dic2):
    t = dic1.items() & dic2.items()
    t = list(t)
    th = []
    for i in range(len(t)):
        th.append(t[i][0])
    res = np.median(th)
    return res


def get_thr(image_list,initial_haarblock_height, initial_haarblock_width, current_scale_num):
    feature_for_all_img = []
    for i in range(len(image_list)):
        img = np.delete(image_list[i], -1)
        img = img.reshape(20, 20)
        f = harr(img, initial_haarblock_height, initial_haarblock_width, current_scale_num)
        feature_1 = reduce(operator.add, f)
        feature_list = reduce(operator.add, feature_1)
        feature_for_all_img.append(feature_list)
    feature_for_all_img = np.array(feature_for_all_img)
    feature_for_all_img = np.transpose(feature_for_all_img)

    thr = []
    h_number = int(feature_for_all_img.shape[0])
    img_number = int(feature_for_all_img.shape[1])

    for j in range(h_number):
        # pos_f = np.zeros((1, int(img_number / 2)))
        # neg_f = np.zeros((1, int(img_number / 2)))
        pos_f= feature_for_all_img[j][0:int(img_number/2)]
        # print(pos_f)
        neg_f= feature_for_all_img[j][int(img_number/2): img_number]
        pos_f_list = pos_f.tolist()
        neg_f_list = neg_f.tolist()
        # print(pos_f_list.type)
        pos_f_list_hist = count_elements(pos_f_list)
        neg_f_list_hist = count_elements(neg_f_list)
        # print('pos_f_list_hist',pos_f_list_hist)
        # print('neg_f_list_hist', neg_f_list_hist)

        t = get_same(pos_f_list_hist,neg_f_list_hist)
        thr.append(t)
    return thr

