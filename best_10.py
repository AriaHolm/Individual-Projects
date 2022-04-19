from function.wak_learner import *
from function.threshold import *

initial_haarblock_height = 6
initial_haarblock_width = 6
current_scale_num = 3

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

trainingimg1_list = np.concatenate((face_training, nonface_training), axis=0)

tao = get_thr(trainingimg1_list,initial_haarblock_height, initial_haarblock_width, current_scale_num)


best_10_h = pick_best_n_feature(trainingimg1_list, tao,10,initial_haarblock_height, initial_haarblock_width, current_scale_num)[0]
min_10_err = pick_best_n_feature(trainingimg1_list,tao,10,initial_haarblock_height, initial_haarblock_width, current_scale_num)[1]

print('best_res', best_10_h)
print('min_10_err', min_10_err)



