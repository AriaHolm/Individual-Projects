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

# using KNN
class_name = ['face','cat','dog']
knn = KNN(k = 5)
# get the test result
test_label = knn.predict(test_data,training_data,training_label)
# visualize the test performance
Recall_list,Precision_list,Overall_accuracy = confusion_matrix(3,class_name,true_label,test_label)


