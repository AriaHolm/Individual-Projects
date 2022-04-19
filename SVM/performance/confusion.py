import numpy as np


# using confusion matrix as the metric
def confusion_matrix(number,class_name,true_label,predict_label):
    confusion_matrix = np.zeros((number,number))
    Recall_list = []
    Precision_list = []
    Overall_accuracy = 0

    # get the confusion matrix
    # row is for predict class labels
    # column is for true class labels
    for m in range(len(predict_label)):
        for i in range(number):
            for j in range(number):
                if true_label[m] == class_name[i] and predict_label[m] == class_name[j]:
                    confusion_matrix[i,j] += 1
                else:
                    confusion_matrix[i, j] += 0
    # get the Recall and Precision for each class
    for i in range(number):
        Recall = confusion_matrix[i,i]/np.sum(confusion_matrix, axis=1)[i]
        Precision = confusion_matrix[i,i]/np.sum(confusion_matrix, axis=0)[i]
        Recall_list.append(Recall)
        Precision_list.append(Precision)
    #  get the Overall_accuracy
    for j in range(number):
        Overall_accuracy += confusion_matrix[j,j]
    Overall_accuracy /= sum(sum(confusion_matrix))

    # visualize the test performance
    print(confusion_matrix)
    print(Recall_list)
    print(Precision_list)
    print(Overall_accuracy)

    return Recall_list,Precision_list,Overall_accuracy
