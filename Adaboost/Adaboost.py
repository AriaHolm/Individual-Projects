from function.wak_learner import *
from function.threshold import *
import math
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from tqdm import tqdm as tq
import time


# Processing traning img_list
face_training = []
nonface_training = []


dirPath1 = r"./data/training/face"
face_training = ReFileName(dirPath1,flag_face)

dirPath2 = r"./data/training/nonface"
nonface_training = ReFileName(dirPath2, flag_nonface)
train2_list = np.concatenate((face_training, nonface_training), axis=0)
# print('img_list',np.array(img_list).shape)


initial_haarblock_height = 6
initial_haarblock_width = 6
current_scale_num = 3
tao = get_thr(train2_list,initial_haarblock_height, initial_haarblock_width, current_scale_num)

# Processing harr characters h_list
image_1 = np.delete(train2_list[0], -1)
image_1 = image_1.reshape(20, 20)
h_list = weak_learner_h_list(image_1, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)

T_num = len(h_list)
N_num = len(train2_list)



def Adaboost(image_list, N, T, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num):
    # def h(t,img,tao):
    #     h_list = weak_learner_h_list(img,tao)
    #     return h_list[t]
    w1 = len(image_list) * [1 / N]
    w = {}
    w[0] = w1
    h_best_number_list = []
    alpha_t_list = []
    for t in tq(range(1, T + 1)):
        time.sleep(0.5)
        # 找到T次迭代里每次的alpha_t和h_t
        # print('t',t)
        min_err = pick_best_feature(image_list, tao, w[t-1],initial_haarblock_height, initial_haarblock_width, current_scale_num)[0]
        # print('min_err',min_err)
        if min_err == 0.5:
            min_err = min_err+0.1
        # print(pick_best_feature(image_list, tao, w[t - 1],initial_haarblock_height, initial_haarblock_width, current_scale_num))
        h_best_number = pick_best_feature(image_list, tao,w[t-1],initial_haarblock_height, initial_haarblock_width, current_scale_num)[1]

        # print('min_err', min_err)
        # print('int-min_err', int(min_err))
        # print('h_best_number', h_best_number)
        # print('int-h_best_number', int(h_best_number))
        alpha_t = (1 / 2) * np.log((1 - min_err) / min_err)
        h_best_number_list.append(h_best_number)
        alpha_t_list.append(alpha_t)
        # update wi for each xi
        Z0 = 0
        w_update = [None] * len(image_list)
        # print('len(image_list)',len(image_list))
        for i in tq(range(len(image_list))):
            time.sleep(0.5)
            # print('cishu',i)
            y_i = image_list[i][-1]
            img = np.delete(image_list[i], -1)
            img = img.reshape(20, 20)
            # print('i-img',img)
            h_new_xi = h(h_best_number, img, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)
            z = w[t - 1][i] * math.exp(-y_i * alpha_t * h_new_xi)
            Z0 += z
        for j in tq(range(len(image_list))):
            time.sleep(0.5)
            w_update[j] = w[t - 1][j] / Z0
        w[t] = w_update
        # using training data to get the best_n_number and alpha of t_itertaion
    return h_best_number_list,alpha_t_list


# Processing test images
flag_face = 1
dirPath3 = r"./data/test/face"
face_test = ReFileName(dirPath3, flag_face)
face_test = np.array(face_test)
# print(face_training.shape)

flag_nonface = -1
dirPath4 = r"./data/test/nonface"
nonface_test = ReFileName(dirPath4, flag_nonface)
nonface_test = np.array(nonface_test)
# print(nonface_training.shape)

testimg_list = np.concatenate((face_test, nonface_test), axis=0)
# test_tao = get_thr(testimg_list)




# using Adaboos and training data（img_list）to get all h_best_list and alpha_list of all T iteration
h_best_list, alpha_list= Adaboost(train2_list, N_num, T_num, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)



def get_H_res_for_test_img(test_img,h_best_list,alpha_list,initial_haarblock_height, initial_haarblock_width, current_scale_num):
    H_upd = 0
    for i in tq(range(len(h_best_list))):
        time.sleep(0.5)
        # 用tao，因为tao是用training data学习到的每一个h的threshold
        H_upd += alpha_list[i] * h(h_best_list[i], test_img, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)
        # print('alpha_list[i]',alpha_list[i])
        a = h(h_best_list[i], test_img, tao,initial_haarblock_height, initial_haarblock_width, current_scale_num)
        # print('aaaaa',a)
        # print('H_upd',H_upd)
    res = 0
    if H_upd > 0:
        res = 1
    if H_upd < 0:
        res = -1
    return res


# plot ROC
def ROC(test_img,predict_score):
    # print(test_img[:,-1].shape)
    # print(predict_score.shape)
    # 给定true_label和predict_label
    # print('test_img',test_img)
    # print('predict_score',predict_score)
    fpr,tpr,threshold = roc_curve(test_img, predict_score)
    roc_auc = auc(fpr,tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False_Positive Rate')
    plt.ylabel('True_Positive Rate')
    plt.title('ROC_Curve')
    plt.legend(loc="upper left")
    plt.savefig("./ROC.png")


# get predict_label_List
def get_H_res_for_img_list(test_img_list,initial_haarblock_height, initial_haarblock_width, current_scale_num):
    H_res_list = []
    for image in test_img_list:
        image = np.delete(image, -1)
        image = image.reshape(20, 20)
        H_res = get_H_res_for_test_img(image, h_best_list, alpha_list,initial_haarblock_height, initial_haarblock_width, current_scale_num)
        H_res_list.append(H_res)
    return H_res_list


# get true_label_List
def get_Test_label_list(test_img_list):
    Test_label_list = []
    for image in test_img_list:
        Test_label = image[-1]
        Test_label_list.append(Test_label)
    return Test_label_list


H_label_list = get_H_res_for_img_list(testimg_list,initial_haarblock_height, initial_haarblock_width, current_scale_num)
true_label_list = get_Test_label_list(testimg_list)

# plot ROC
print('H_label_list',H_label_list)
print('true_label_list',true_label_list)
ROC(H_label_list, true_label_list)
