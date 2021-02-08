#!/usr/bin/env python3
# Find different words in newspaper article
# We'll do this using morphology operators and connected components.
# %%

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image




kernel = np.zeros((5, 5), dtype=np.uint8)
kernel[2, :] = 1
kernel[:, 2] = 1

figsize = (10, 10)

#%%
im1 = cv2.imread("news.jpg")
im_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

im2 = cv2.imread("news.jpg")
im_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
########################################################################################################################
thresh11 = 125
maxval11 = 255
kernel_new1 = np.ones((5, 5), dtype=np.uint8)

th, new_img11 = cv2.threshold(im_gray, thresh11, maxval11, cv2.THRESH_BINARY)

new_binary = cv2.dilate(new_img11, kernel_new1)

binary_only_title_cc_img = cv2.erode(new_binary, kernel_new1, iterations=4)
#########################################################################################################################
plt.figure(figsize=figsize)
plt.imshow(binary_only_title_cc_img, cmap="gray", vmin=0, vmax=255)
plt.show()

#%%
# TODO: let's start with turning the image to a binary one
thresh = 120
maxval = 255

ret, im_th = cv2.threshold(im_gray, thresh, maxval, cv2.THRESH_BINARY)
plt.figure(figsize=(20, 20))
plt.imshow(im_th, cmap="gray", vmin=0, vmax=255)
plt.title('Threshold')
plt.show()

dilated_im = cv2.erode(im_th, kernel)
plt.figure(figsize=(20, 20))
plt.imshow(dilated_im, cmap="gray", vmin=0, vmax=255)
plt.title('dilated_img')
plt.show()



#%%


def find_words(dilate, im):
    mask = np.zeros(dilate.shape, np.uint8)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for idx in range(0, len(hierarchy[0])):
        rect = x, y, rect_width, rect_height = cv2.boundingRect(contours[idx])
        # fill the contour
        mask = cv2.drawContours(mask, contours, idx, (255, 255, 255), cv2.FILLED)
        # ratio of non-zero pixels in the filled region
        r = (cv2.countNonZero(mask)) / (rect_width * rect_height)
        if r > 0.45 and rect_height > 8 and rect_width > 6:
            im = cv2.rectangle(im, (x, y + rect_height), (x + rect_width, y), (0, 255, 0), 1)
    res = im.copy()


    # 1. find all connected components
    # 2. build a mask of only one connected component each time, and find it extremeties
    return res




def plot_rec(mask,res_im):
    # plot a rectengle around each word in res image using mask image of the word
    xy = np.nonzero(mask)
    y = xy[0]
    x = xy[1]
    left = x.min()
    right = x.max()
    up = y.min()
    down = y.max()

    res_im = cv2.rectangle(res_im, (left, up), (right, down), (0, 20, 200), 2)
    return res_im
#%%

plt.figure(figsize=(20,20))
plt.imshow(find_words(binary_only_title_cc_img, im1),cmap="gray", vmin=0, vmax=255)
plt.show()


plt.figure(figsize=(20,20))
plt.imshow(find_words(dilated_im, im2),cmap="gray", vmin=0, vmax=255)
plt.show()


#%%





