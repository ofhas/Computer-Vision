#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
#%%

forest_img = cv2.imread("forest.jpg")
pyramid_img = cv2.imread("pyramids.png")
pyramid_img = cv2.cvtColor(pyramid_img, cv2.COLOR_BGR2GRAY)
forest_img = cv2.cvtColor(forest_img, cv2.COLOR_BGR2GRAY)

forest_img_resize = cv2.resize(forest_img, (pyramid_img.shape[1], pyramid_img.shape[0]))

pyramid_img[pyramid_img == 0] = forest_img_resize[pyramid_img == 0]



size = forest_img.shape[1::-1] # this will allow you to get any image shape size

res_im = pyramid_img

new_res = cv2.resize(res_im, size)

#%%
plt.figure()
plt.imshow(new_res, cmap='gray', vmin=0, vmax=255)
plt.show()
#%%
