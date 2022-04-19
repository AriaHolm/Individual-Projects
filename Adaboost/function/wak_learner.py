import cv2
import numpy as np
from function.read_Img import *
from function.threshold import *
from function.harr_f import *
# from threshold import *
# from input import *
import operator
from functools import reduce


# initial_haarblock_height = 6
# initial_haarblock_width = 6
# current_scale_num = 3
flag_face = 1
dirPath1 = r"./data/training/face"
face_training = ReFileName(dirPath1,flag_face)
face_training = np.array(face_training)
# print(face_training.shape)



flag_nonface = -1
dirPath2 = r"./data/training/nonface"
nonface_training = ReFileName(dirPath2,flag_nonface)
nonface_training = np.array(nonface_training)
# print(nonface_training.shape)


fun_list = np.concatenate((face_training, nonface_training), axis=0)
#
# tao = get_thr(fun_list,initial_haarblock_height, initial_haarblock_width, current_scale_num)


# 给一个img，可以输出所有ht(xi)的list
def weak_learner_h_list(img, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num):
    # print('this is weak_learner_h_list for harr')
    # print('img.shape',img.shape)
    f = harr(img,initial_haarblock_height,initial_haarblock_width,current_scale_num)
    # print('f.shape[0]', f.shape[0])
    # print('f.shape[1]', f.shape[1])
    # print('f.shape[2]', f.shape[2])
    # print('f.shape[3]', f.shape[3])
    # print('f.shape[4]', f.shape[4])

    feature_1 = reduce(operator.add, f)
    feature_list = reduce(operator.add, feature_1)

    h_list = [None] * len(feature_list)
    # print(h_list)
    for i in range(len(feature_list)):
        # print('i', i)
        # print(feature_list[i])
        if feature_list[i] - tao[i] >= 0:
            h_list[i] = 1
        else:
            h_list[i] = -1
    return h_list


# get the t_th of h_res for img
def h(t, img, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num):
    h_list = weak_learner_h_list(img, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)
    return h_list[t]


def h_for_all_img(img_list, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num):
    h_for_all_img_list = []
    for j in range(len(img_list)):
        # print('img_list[j]',img_list[j])
        # print(img_list[j].shape)
        img = np.delete(img_list[j], -1)
        # print(img.shape)
        img = img.reshape(20, 20)
        # print('this is h_for_all_img for weak_learner_h_list(img, tao)')
        h_l = weak_learner_h_list(img, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)
        h_for_all_img_list.append(h_l)
    return np.array(h_for_all_img_list)



def error_of_h_number(img_list, h_number,initial_haarblock_height, initial_haarblock_width, current_scale_num, tao):
    # print('this is error_of_h_number for h_for_all_list = h_for_all_img(img_list)')
    h_for_all_list = h_for_all_img(img_list,tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)
    N = len(img_list)
    error_for_all_img_list = []
    for m in range(h_for_all_list.shape[1]):
        error_ht = 0
        for n in range(h_for_all_list.shape[0]):
            label = img_list[n][-1]
            # img1 = np.append(img1, 1)
            if h_for_all_list[n][m] == label:
                # print('right')
                error_ht += 0
            else:
                # print('wrong')
                error_ht += 1
        error_for_all_img_list.append((1/N)*error_ht)
    return error_for_all_img_list[h_number]




def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def error_of_h_number_1(img_list, h_number, w_t,initial_haarblock_height, initial_haarblock_width, current_scale_num,tao):
    # print('this is error_of_h_number for h_for_all_list = h_for_all_img(img_list)')
    h_for_all_list = h_for_all_img(img_list, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)
    N = len(img_list)
    error_for_all_img_list = []
    for m in range(h_for_all_list.shape[1]):
        error_ht = 0
        for n in range(h_for_all_list.shape[0]):
            label = img_list[n][-1]
            # img1 = np.append(img1, 1)
            if h_for_all_list[n][m] == label:
                # print('right')
                error_ht += 0
            else:
                # print('wrong')
                error_ht += 1 * w_t[n]
        error_for_all_img_list.append(error_ht)
    return error_for_all_img_list[h_number]


# get the list number of min error
def pick_best_feature(img_list, tao, w_t,initial_haarblock_height, initial_haarblock_width, current_scale_num):
    # print('this is pick_best_feature for h_all_list = h_for_all_img(img_list, tao)')
    h_all_list = h_for_all_img(img_list, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)
    err = []
    for num in range(h_all_list.shape[1]):
        # print('this is pick_best_feature for error_of_h_number(img_list,num)')
        err_h_num = error_of_h_number_1(img_list, num, w_t,initial_haarblock_height, initial_haarblock_width, current_scale_num,tao)
        err.append(err_h_num)
    dic = {}
    for q in range(h_all_list.shape[1]):
        cur_err = err[q]
        dic[q] = cur_err
    min_error = min(dic.values())
    # print('dic', dic)
    # print('min_error', min_error)
    best_feature = min(dic, key=dic.get)
    return min_error, best_feature



# get the list number of n min errors
def pick_best_n_feature(img_list, tao,n,initial_haarblock_height, initial_haarblock_width, current_scale_num):
    h_all_list = h_for_all_img(img_list, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)
    err = []
    for num in range(h_all_list.shape[1]):
        err_h_num = error_of_h_number(img_list,num,initial_haarblock_height, initial_haarblock_width, current_scale_num, tao)
        err.append(err_h_num)
    dic = {}
    for t in range(h_all_list.shape[1]):
        cur_err = err[t]
        dic[t] = cur_err
    # print('dic',dic)
    err_list = sorted(dic.items(), key=lambda item:item[1])
    # print('err_list',err_list)
    min_n_err = []
    best_n_feature = []
    for i in range(n):
        best_h = err_list[i][0]
        min_err = err_list[i][1]
        best_n_feature.append(best_h)
        min_n_err.append(min_err)
    return best_n_feature,min_n_err





