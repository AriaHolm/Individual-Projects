from preprocessing.read_Img import *
from algorithm.SVM.two_class_svm import *
from performance.confusion import *


class Multi_Class_SVM():
    def __init__(self, C, kernel, sigma = None):
        self.C = C
        self.kernel = kernel
        self.sigma = sigma
    # using one v.s. one strategy to construct
    def construct_one_to_one_svmPredictor(self,class_path_list,class_name):
        classifiers = []
        names_comb = []
        for i in range(len(class_path_list)):
            path_1 = class_path_list[i]
            data1 = ReadImg(path_1)
            label1 = [1]*len(data1)
            name1 = class_name[i]
            for j in range(len(class_path_list)):
                if j !=i:
                    path_2 = class_path_list[j]
                    data2 = ReadImg(path_2)
                    label2 = [-1] * len(data2)
                    name2 = class_name[j]

                    names = []
                    names.append(name1)
                    names.append(name2)

                    training_data = np.concatenate((data1, data2), axis=0)
                    training_label = np.concatenate((label1, label2), axis=0)
                    single_predictor = SVM(self.C, self.kernel)
                    names_comb.append(names)

                    classifiers.append(single_predictor.train(training_data, training_label))
                else:
                    pass

        return names_comb, classifiers

    def predict(self,class_path_list, class_name, test_data):
        names_comb,classifiers = self.construct_one_to_one_svmPredictor(class_path_list,class_name)
        result = []
        for i in range(len(test_data)):
            pre = []
            for j in range(len(classifiers)):
                classifier = classifiers[j]
                name_comb = names_comb[j]
                res = classifier.predict(test_data[i])
                predict_class_name = ' '
                if res == 1:
                    predict_class_name = name_comb[0]
                if res == -1:
                    predict_class_name = name_comb[1]
                pre.append(predict_class_name)
            predict_class = max(pre,key=pre.count)
            result.append(predict_class)
        return result


