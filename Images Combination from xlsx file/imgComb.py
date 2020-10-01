import pandas as pd
import cv2
import numpy as np



col_list = ["pic_north", 'pic_east', 'pic_west', "pic_south", 'MK', 'BID']

path = 'C:/Users/a/PycharmProjects/pythonProject/test.xls' # choose the images path
print('Path:', path)


df = pd.read_excel(path, usecols=col_list)


newArray = np.array(df)


def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)

    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    print(w_min)
    im_list_resize = [cv2.resize(im, (1000, 300), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


j = 0
for i in range(newArray.shape[0]): # this will create the image name
    name = f'{newArray[i][4]}-{newArray[i][5]}.jpg'

    emptyArray = []
    while j < newArray.shape[1]:
        emptyArray.append(newArray[j])

        im1 = cv2.imread(newArray[j][0])
        im2 = cv2.imread(newArray[j][1])
        im3 = cv2.imread(newArray[j][2])
        im4 = cv2.imread(newArray[j][3])

        im_tile_resize = concat_tile_resize([[im1, im2],
                                             [im3, im4]])

        cv2.imwrite(name, im_tile_resize)
        emptyArray = []
        j += 1
        break


