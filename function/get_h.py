import cv2
from tqdm import tqdm as tq
import time
# 获取在一张training图像上的h_block显示
def get_best_h_feature_block_in_img(img, h_number, haarblock_height, haarblock_width):
    # for h_number in best_10_h:
        # if h is belong to scale1
        # haarimg_shape is 20-6+1 =15（15*15）
        # height = haarblock_height
        # width = haarblock_width
        if h_number+1 <=1125:
            height = haarblock_height
            width = haarblock_width
            #     type1 (black for left, white for right)
            img = img
            if h_number+1 in range(1,226):
                # img1 = img
                i_j = []
                dic1 = {}
                p = 0
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic1[p] = cur
                        p += 1
                m = dic1[h_number][0] + height
                n = dic1[h_number][1] + width
                print(m,n)
            #     type1是左黑右白
                print('img',img)
                img1 = img
                img1[m - height:m,n - width: n - int(width/2)] = 0
                img1[m - height:m,n - int(width / 2):n] = 255
                print(m - height,n - width,n - int(width/2))
                print('img1',img1)
                # cv2.imwrite('./best_h1.jpg', img1)
                # img1 = img
                return img1
            if h_number+1 in range(226,451):
                img2 = img
                i_j = []
                dic2 = {}
                p = 225
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic2[p] = cur
                        p += 1
                print(h_number)
                print('dic2[h_number]', dic2)
                m = dic2[h_number][0] + height
                n = dic2[h_number][1] + width
                print('m',m)
                print('n',n)
                #     type2 (black for upper, white for below)
                img2[m - height:m - int(height / 2) , n - width: n ] = 0
                img2[m - int(height / 2):m , n - width: n ] = 255
                # cv2.imwrite('best_h2.jpg', img2)
                return img2
            if h_number+1 in range(451,676):
                img3 = img
                i_j = []
                dic3 = {}
                p = 450
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic3[p] = cur
                        p += 1
                m = dic3[h_number][0] + height
                n = dic3[h_number][1] + width
                #     type3 (black for middle, white for right and left)
                img3[m - height:m, n - width: n - 2 * int(width / 3)] = 255
                img3[m - height:m, n - 2 * int(width / 3): n - int(width / 3)] = 0
                img3[m - height:m, n - int(width / 3): n] = 255
                # cv2.imwrite('./best_h3.jpg', img3)
                return img3
            if h_number+1 in range(676,901):
                img4 = img
                i_j = []
                dic4 = {}
                p = 675
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic4[p] = cur
                        p += 1
                m = dic4[h_number][0] + height
                n = dic4[h_number][1] + width
                #     type4 (black for middle, white for upper and below)
                img4[m - height:m - 2 * int(height / 3) , n - width: n ] = 255
                img4[m - 2 * int(height / 3): m - int(height / 3) , n - width: n] = 0
                img4[m - int(height / 3):m , n - width: n ] = 255
                # cv2.imwrite('./best_h4.jpg', img4)
                return img4

            if h_number+1 in range(901,1126):
                img5 = img
                i_j = []
                dic5 = {}
                p = 900
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic5[p] = cur
                        p += 1
                m = dic5[h_number][0] + height
                n = dic5[h_number][1] + width
                #     type5(black for left below, white for right upper)
                img5[m - int(height / 2):m, n - width: n - int(width / 2)] = 0
                img5[m - height: m - int(height / 2), n - int(width / 2): n] = 0
                img5[m - height:m - int(height / 2), n - width: n - int(width / 2)] = 255
                img5[m - int(height / 2):m, n - int(width / 2): n] = 255
                # cv2.imwrite('./best_h5.jpg', img5)
                return img5
        else:
            height = haarblock_height * 2
            width = haarblock_width * 2
            if h_number+1 in range(1126,1207):
                img6 = img
                i_j = []
                dic6 = {}
                p = 1125
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic6[p] = cur
                        p += 1
                m = dic6[h_number][0] + height
                n = dic6[h_number][1] + width

                img6[m - height:m , n - width: n - int(width / 2)] = 0
                img6[m - height:m , n - int(width / 2):n] = 255
                # cv2.imwrite('./best_h6.jpg', img6)
                return img6
            if h_number+1 in range(1207,1288):
                img7 = img
                i_j = []
                dic7 = {}
                p = 1206
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic7[p] = cur
                        p += 1
                m = dic7[h_number][0] + height
                n = dic7[h_number][1] + width

                img7[m - height:m - int(height / 2), n - width: n] = 0
                img7[m - int(height / 2):m, n - width: n] = 255
                # cv2.imwrite('./best_h7.jpg', img7)
                return img7
            if h_number+1 in range(1288,1369):
                img8 = img
                i_j = []
                dic8 = {}
                p = 1287
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic8[p] = cur
                        p += 1
                m = dic8[h_number][0] + height
                n = dic8[h_number][1] + width

                img8[m - height:m, n - width: n - 2 * int(width / 3)] = 255
                img8[m - height:m, n - 2 * int(width / 3): n - int(width / 3)] = 0
                img8[m - height:m, n - int(width / 3): n] = 255
                # cv2.imwrite('./best_h8.jpg', img8)
                return img8
            if h_number+1 in range(1369,1450):
                img9 = img
                i_j = []
                dic9 = {}
                p = 1368
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic9[p] = cur
                        p += 1
                m = dic9[h_number][0] + height
                n = dic9[h_number][1] + width
                print(m,n)

                img9[m - height:m - 2 * int(height / 3), n - width: n] = 255
                img9[m - 2 * int(height / 3): m - int(height / 3), n - width: n] = 0
                img9[m - int(height / 3):m, n - width: n] = 255
                print(img9)
                # cv2.imwrite('./best_h9.jpg', img9)
                return img9
            if h_number+1 in range(1450,1531):
                img10 =img
                i_j = []
                dic10 = {}
                p = 1449
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic10[p] = cur
                        p += 1
                m = dic10[h_number][0] + height
                n = dic10[h_number][1] + width

                img10[m - int(height / 2):m, n - width: n - int(width / 2)] = 0
                img10[m - height: m - int(height / 2), n - int(width / 2): n] = 0
                img10[m - height:m - int(height / 2), n - width: n - int(width / 2)] = 255
                img10[m - int(height / 2):m, n - int(width / 2): n] = 255
                # cv2.imwrite('./best_h10.jpg', img10)
                return img10
        if h_number+1 >1530:
            height = haarblock_height * 3
            width = haarblock_width * 3
            if h_number+1 in range(1531,1540):
                img11= img
                i_j = []
                dic11 = {}
                p = 1530
                for a in range(0,9):
                    i= a
                    for b in range(0,9):
                        j = b
                        cur = [i, j]
                        dic11[p] = cur
                        p += 1
                m = dic11[h_number][0] + height
                n = dic11[h_number][1] + width

                img11[m - height:m , n - width: n - int(width / 2)] = 0
                img11[m - height:m , n - int(width / 2):n] = 255
                # cv2.imwrite('./best_h11.jpg', img11)
                return img11
            if h_number+1 in range(1540,1549):
                img12 = img
                i_j = []
                dic12 = {}
                p = 1539
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic12[p] = cur
                        p += 1
                m = dic12[h_number][0] + height
                n = dic12[h_number][1] + width

                img12[m - height:m - int(height / 2), n - width: n] = 0
                img12[m - int(height / 2):m, n - width: n] = 255
                # cv2.imwrite('./best_h12.jpg', img12)
                return img12
            if h_number+1 in range(1549,1558):
                img13 = img
                i_j = []
                dic13 = {}
                p = 1548
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic13[p] = cur
                        p += 1
                m = dic13[h_number][0] + height
                n = dic13[h_number][1] + width

                img13[m - height:m, n - width: n - 2 * int(width / 3) ] = 255
                img13[m - height:m, n - 2 * int(width / 3): n - int(width / 3) ] = 0
                img13[m - height:m, n - int(width / 3): n ] = 255
                # cv2.imwrite('./best_h13.jpg', img13)
                return img13
            if h_number+1 in range(1558,1567):
                img14 = img
                i_j = []
                dic14 = {}
                p = 1557
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic14[p] = cur
                        p += 1
                m = dic14[h_number][0] + height
                n = dic14[h_number][1] + width

                img14[m - height:m - 2 * int(height / 3), n - width: n] = 255
                img14[m - 2 * int(height / 3): m - int(height / 3), n - width: n] = 0
                img14[m - int(height / 3):m, n - width: n] = 255
                # cv2.imwrite('./best_h14.jpg', img14)
                return img14
            if h_number+1 in range(1567,1576):
                img15 = img
                i_j = []
                dic15 = {}
                p = 1566
                for a in range(0,15):
                    i= a
                    for b in range(0,15):
                        j = b
                        cur = [i, j]
                        dic15[p] = cur
                        p += 1
                m = dic15[h_number][0] + height
                n = dic15[h_number][1] + width

                img15[m - int(height / 2):m , n - width: n - int(width / 2) ] = 0
                img15[m - height: m - int(height / 2) , n - int(width / 2): n ] = 0
                img15[m - height:m - int(height / 2) , n - width: n - int(width / 2) ] = 255
                img15[m - int(height / 2) :m , n - int(width / 2): n ] = 255
                # cv2.imwrite('./best_h15.jpg', img15)
                return img15

img_op = cv2.imread('../data/training/face/000000.jpg', 0)
cv2.imwrite('../task1_result/origin.jpg', img_op)
best_res = [794]
import os
def get_h(img):
    for i in tq(best_res):
        time.sleep(0.5)
        print('ppp')
        path = input()
        img_input = img
        k = get_best_h_feature_block_in_img(img_input, i, 6, 6)
        newname = path + os.sep + str(i) + '.jpg'
        cv2.imwrite(newname, k)


get_h(img_op)

best_res1 =  [756, 757, 246, 64, 755, 78, 247, 734, 261, 794]
min_10_err = [0.192, 0.20700000000000002, 0.2175, 0.221, 0.225, 0.2295, 0.23, 0.2315, 0.233, 0.233]
# get_best_h_feature_block_in_img()