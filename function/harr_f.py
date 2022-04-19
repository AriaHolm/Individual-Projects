import cv2
import numpy as np

def integral(img):

    # print('img',img)
    # print('img.shape', img.shape)
    integimg = np.zeros(shape=(img.shape[0] + 1, img.shape[1] + 1), dtype=np.int32)
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            integimg[i][j] = img[i][j] + integimg[i - 1][j] + integimg[i][j - 1] - integimg[i - 1][j - 1]
    # plt.imshow( integimg )
    # plt.show()
    # print('    Done!')
    # print('integimg', np.shape(integimg))
    return integimg



def haar_onescale(img, integimg, haarblock_height, haarblock_width):
    # integimg = integral(img)
    #  no padding
    haarimg = np.zeros(shape=(img.shape[0] - haarblock_height + 1, img.shape[1] - haarblock_width + 1), dtype=np.int32)
    #     haarimg = np.zeros(shape=(haarblock_height, haarblock_width), dtype=np.int32)
    # plt.imshow( haarimg )
    # plt.show()
    haar_feature_onescale = []
    haar_feature_onescale_kind_1 = []
    for i in range(haarimg.shape[0]):
        for j in range(haarimg.shape[1]):
            # m, n is the coordinate of i,j in origin image
            m = haarblock_height + i
            n = haarblock_width + j
            # print('m', m)
            # print('n', n)
            # print('m - haarblock_height', m - haarblock_height)
            # print('n - haarblock_width', n - haarblock_width)

            haar_all = integimg[m][n] - integimg[m - haarblock_height][n] - integimg[m][n - haarblock_width] + \
                       integimg[m - haarblock_height][n - haarblock_width]
            # print('integimg[m][n]', integimg[m][n])
            # print('integimg[m - haarblock_height][n]', integimg[m - haarblock_height][n])
            # print('integimg[m][n - haarblock_width]', integimg[m][n - haarblock_width])
            # print('integimg[m - haarblock_height][n - haarblock_width]',
            #       integimg[m - haarblock_height][n - haarblock_width])
            # print('haar_all', haar_all)
            haar_black = integimg[m][n - int(haarblock_width / 2)] - integimg[m - haarblock_height][
                n - int(haarblock_width / 2)] - integimg[m][n - haarblock_width] + integimg[m - haarblock_height][
                             n - haarblock_width]
            #             print('haar_black',haar_black)
            # 1*all - 2*black = white - black
            haarimg[i][j] = 1 * haar_all - 2 * haar_black
            #             print('haarimg[i][j]',haarimg[i][j])
            haar_feature_onescale_kind_1.append(haarimg[i][j])
    # print('haar_feature_onescale_kind_1', haar_feature_onescale_kind_1)
    haar_feature_onescale.append(haar_feature_onescale_kind_1)

    haar_feature_onescale_kind_2 = []
    for i in range(haarimg.shape[0]):
        for j in range(haarimg.shape[1]):

            m = haarblock_width + i
            n = haarblock_height + j
            haar_all = integimg[m][n] - integimg[m - haarblock_height][n] - integimg[m][n - haarblock_width] + \
                       integimg[m - haarblock_height][n - haarblock_width]
            haar_black = integimg[m - int(haarblock_height / 2)][n] - integimg[m - int(haarblock_height / 2)][
                n - haarblock_width] - integimg[m - haarblock_height][n] + integimg[m - haarblock_height][
                             n - haarblock_width]
            # 1*all - 2*black = white - black
            haarimg[i][j] = 1 * haar_all - 2 * haar_black
            # print('haarimg[i][j]-2', haarimg[i][j])
            haar_feature_onescale_kind_2.append(haarimg[i][j])
    haar_feature_onescale.append(haar_feature_onescale_kind_2)

    haar_feature_onescale_kind_3 = []
    for i in range(haarimg.shape[0]):
        for j in range(haarimg.shape[1]):

            m = haarblock_width + i
            n = haarblock_height + j
            haar_all = integimg[m][n] - integimg[m - haarblock_height][n] - integimg[m][n - haarblock_width] + \
                       integimg[m - haarblock_height][n - haarblock_width]
            haar_black = integimg[m][n - int(haarblock_width / 3)] - integimg[m - haarblock_height][
                n - int(haarblock_width / 3)] - integimg[m][n - 2 * int(haarblock_width / 3)] + \
                         integimg[m - haarblock_height][
                             n - 2 * int(haarblock_width / 3)]
            # 1*all - 3*black = white - 2*black
            haarimg[i][j] = 1 * haar_all - 3 * haar_black
            # print('haarimg[i][j]-3', haarimg[i][j])
            haar_feature_onescale_kind_3.append(haarimg[i][j])
    haar_feature_onescale.append(haar_feature_onescale_kind_3)

    haar_feature_onescale_kind_4 = []
    for i in range(haarimg.shape[0]):
        for j in range(haarimg.shape[1]):

            m = haarblock_width + i
            n = haarblock_height +j
            haar_all = integimg[m][n] - integimg[m - haarblock_height][n] - integimg[m][n - haarblock_width] + \
                       integimg[m - haarblock_height][n - haarblock_width]
            haar_black = integimg[m - int(haarblock_height / 3)][n] - integimg[m - int(haarblock_height / 3)][
                n - haarblock_width] - integimg[m - 2 * int(haarblock_height / 3)][n] + \
                         integimg[m - 2 * int(haarblock_height / 3)][
                             n - haarblock_width]
            # 1*all - 3*black = white - 2*black
            haarimg[i][j] = 1 * haar_all - 3 * haar_black
            # print('haarimg[i][j]-4', haarimg[i][j])
            haar_feature_onescale_kind_4.append(haarimg[i][j])
    haar_feature_onescale.append(haar_feature_onescale_kind_4)

    haar_feature_onescale_kind_5 = []
    for i in range(haarimg.shape[0]):
        for j in range(haarimg.shape[1]):

            m = haarblock_width + i
            n = haarblock_height + j
            haar_all = integimg[m][n] - integimg[m - haarblock_height][n] - integimg[m][n - haarblock_width] + \
                       integimg[m - haarblock_height][n - haarblock_width]
            haar_black_1 = integimg[m][n - int(haarblock_width / 2)] - integimg[m][
                n - haarblock_width] - integimg[m - int(haarblock_height / 2)][n - int(haarblock_width / 2)] + \
                           integimg[m - int(haarblock_height / 2)][n - haarblock_width]
            haar_black_2 = integimg[m - int(haarblock_height / 2)][n] - integimg[m - haarblock_height][
                n] - integimg[m - int(haarblock_height / 2)][n - int(haarblock_width / 2)] + \
                           integimg[m - haarblock_height][n - int(haarblock_width / 2)]
            # 1*all - 2*(black_1+black_2) = white_1+white_2 - (black_1+black_2)
            haarimg[i][j] = 1 * haar_all - haar_black_1 - haar_black_2
            # print('haarimg[i][j]-5', haarimg[i][j])
            haar_feature_onescale_kind_5.append(haarimg[i][j])
    haar_feature_onescale.append(haar_feature_onescale_kind_5)

    return haar_feature_onescale


def harr(img,haarblock_height,  haarblock_width, Scale_num ):
    feature = []
    haar_num = 0
    integimg = integral(img)
    for i in range( Scale_num):

        haarblock_width = i*haarblock_width+haarblock_width
        haarblock_height = i*haarblock_height+haarblock_height
        haar_feature_onescale = haar_onescale(img, integimg, haarblock_height, haarblock_width)
        # print(np.shape(haar_feature_onescale))
        haar_num += len( haar_feature_onescale )
        feature.append( haar_feature_onescale )
        # print(len(feature[0]))
#         initialize the haarblock_width and haarblock_height back to 6
        haarblock_width = 6
        haarblock_height = 6

    return feature