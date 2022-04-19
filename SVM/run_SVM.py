from algorithm.SVM.kernel import *
from algorithm.SVM.multi_class_SVM import *
from algorithm.KNN.knn import *


face_Path = '././dataset/training/face'
cat_Path = '././dataset/training/cat'
dog_Path = '././dataset/training/dog'

face = ReadImg(face_Path)
cat = ReadImg(cat_Path)
dog = ReadImg(dog_Path)

# training data
training_data = np.concatenate((face, cat,dog), axis=0)

# training label
training_label = ['face']*len(face) + ['cat']*len(cat) + ['dog']*len(dog)


face_test_Path = '././dataset/test/face'
cat_test_Path = '././dataset/test/cat'
dog_test_Path = '././dataset/test/dog'

face_test = ReadImg(face_test_Path)
cat_test = ReadImg(cat_test_Path)
dog_test = ReadImg(dog_test_Path)

# test data
test_data = np.concatenate((face_test, cat_test,dog_test), axis=0)

# true label
true_label = ['face']*len(face_test) + ['cat']*len(cat_test) + ['dog']*len(cat_test)


# using SVM
class_path = [face_Path,cat_Path,dog_Path]
class_name = ['face','cat','dog']

poly_kernel = polynomial_kernel(2,1)
gaussian_kernel = rbf_kernel(1/600)

# choose C =2 and poly_kernel
svm = Multi_Class_SVM(2, poly_kernel)
# get test result
test_label = svm.predict(class_path, class_name, test_data)
# visualize the test performance
Recall_list,Precision_list,Overall_accuracy = confusion_matrix(3,class_name,true_label,test_label)


