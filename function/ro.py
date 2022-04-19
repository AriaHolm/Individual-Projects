import math
import scipy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from Adaboost import *

from read_Img import *
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

def parameters(predict_label, test_label, n):
    True_Positive = 0
    False_Negative = 0
    False_Positive = 0
    True_Negative = 0
    for i in range (n):
        if predict_label[i] == 1 and test_label[i]==1:
            True_Positive = True_Positive + 1
        if predict_label[i] == 1 and test_label[i]==-1:
            False_Positive = False_Positive + 1
        if predict_label[i] == -1 and test_label[i]==1:
            False_Negative =False_Negative + 1
        if predict_label[i] == -1 and test_label[i]==-1:
            True_Negative = True_Negative + 1
    print("True_Positive=",True_Positive,"False_Negative=",False_Negative,"False_Positive=",False_Positive,"True_Negative=",True_Negative)
    print("False_Positive rate:", False_Positive/(False_Positive+True_Negative))
    print("False_Negative rate:", False_Negative/(True_Positive+False_Negative))
    print("misclassification rate:", (False_Positive+False_Negative)/n)

H_label_list = get_H_res_for_img_list(testimg_list,initial_haarblock_height, initial_haarblock_width, current_scale_num)
true_label_list = get_Test_label_list(testimg_list)

ROC(H_label_list, true_label_list)
parameters(H_label_list,true_label_list)