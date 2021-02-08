#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2

figsize = (10, 10)

#%%

img = np.zeros((50, 50))
img[20:30, 20:30] = 1

plt.figure(figsize=figsize)
plt.imshow(img, cmap="gray")
plt.title('Original Image')
plt.show()

#%%

result = np.zeros_like(img)
kernel = np.zeros((5, 5),dtype=np.uint8)
kernel[2, :] = 1
kernel[:, 2] = 1


# %%
#dilation
def dilate(img, kernel):
    new_img1 = cv2.filter2D(img, -1, kernel) # this will mask the original image with the kerenl
    thresh = 0
    maxValue = 1
    th, dst1 = cv2.threshold(new_img1, thresh, maxValue, cv2.THRESH_BINARY) # this will turn each pixle of a non black or white value into a white only pixle
    return dst1

#erosion

def erosion(img, kernel):
    new_img = cv2.filter2D(img, -1, kernel)  # this will mask the original image with the kernel
    maxval = np.amax(new_img)  # this will give us the max value of array

    for index, pixel in np.ndenumerate(new_img):  # this will allow us to iterate over a 2d list
        x,y, = index
        if new_img[x, y] < maxval:
            new_img[x, y] = 0  # this will turn every pixel that is between a 255 and 0 off the grey scale to black only
        elif new_img[x, y] == maxval:  # in case the max value is different from 1 we'll change it to 1
            new_img[x, y] = 1
    return new_img
