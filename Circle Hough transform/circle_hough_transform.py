
#Circle Hough transform
#we will implement step by step circle hough transform.

# %%
from matplotlib import pyplot as plt
import cv2
import numpy as np

figsize = (10, 10)

# %% [markdown]
# ## Import an image
# %%
im3 = cv2.imread("circles.bmp")
im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
im = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=figsize)
plt.imshow(im3, cmap='gray', vmin=0, vmax=255)
plt.title("original image")
plt.show()


blur_image = cv2.GaussianBlur(im, (3, 3), 0)


# %% [markdown]
# ## Find edges of an image using Canny
# %%
# Canny edge detection of image

canny = cv2.Canny(im, 127, 255)
plt.figure(figsize=figsize)
plt.imshow(canny)
plt.title("canny  image")
plt.show()


# %% [markdown]
# ## Initialize accumulation matrix
# %%
# state parameters for accumulation matrix

r_step = 1
rmax = np.sqrt(im.shape[0]**2+im.shape[1]**2)
r_vec = np.arange(0, rmax, r_step)
a_vec = np.arange(0, im.shape[1], 1)
b_vec = np.arange(0, im.shape[0], 1)

# init accumulation matrix (one line)
acc_mat = np.zeros((a_vec.shape[0], b_vec.shape[0], r_vec.shape[0]))# in this case we also need radius due to a circles we need to detect

# %% [markdown]
# ## Fill accumulation matrix
# %%
# get indices of edges
edge_inds = np.argwhere(canny > 0)
# run on all a,b and edge indices and find corresponding R
for yx in edge_inds:
    x = yx[1]
    y = yx[0]
    print('point:', (x, y))
    for a, a0 in enumerate(a_vec):
        for b, b0 in enumerate(b_vec):
            # find best corresponding r0 (1 line)

            r0 = np.sqrt((x-a0)**2+(y-b0)**2)
            r_ind = np.argmin(np.abs(r0-r_vec))
            # update accumulation matrix (one line)
            acc_mat[a, b, r_ind] += 1
# %%


# %% [markdown]
# ## Threshold accumulation matrix
# %%
TH = 50
acc_mat_th = acc_mat > TH


# %% [markdown]
# ## Min distance
# This is a new feature that deals with noise in the accumulation matrix.
# 1. Search in the neighborhood of each above TH bin for other above TH bins
# 2. compare the two and delete the less important one
# %%
edge_inds = np.argwhere(acc_mat_th > 0)

min_dist = 15

acc_mat_th_dist = acc_mat_th.copy()
# run on all above TH bins
for i in range(edge_inds.shape[0]):
    b0, a0, r0 = edge_inds[i]

    # search in all other above TH bins
    for j in range(i+1, edge_inds.shape[0]):
        b1, a1, r1 = edge_inds[j]

        # if the two above are neighbors (below min_dist) - delete the less important
        if ((r0-r1)*r_step)**2+((a0-a1))**2+((b0-b1))**2 < min_dist**2:
            if acc_mat[b0, a0, r0] >= acc_mat[b1, a1, r1]:
                # one line fill here
                acc_mat_th_dist[b1, a1, r1] = 0
            else:
                # one line fill here
                acc_mat_th_dist[b0, a0, r0] = 0
# %%
plt.figure(figsize=figsize)
plt.imshow(np.sum(acc_mat_th_dist, axis=2), extent=[
             b_vec.min(), b_vec.max(), a_vec.max(), a_vec.min()], aspect='auto')
plt.xlabel('a')
plt.ylabel('b')
plt.title('accumulation matrix TH and min_dist summed over r axis')
plt.colorbar()
plt.show()

# %% [markdown]
# ## Plot circles found by hough
# %%
# get indices of acc_mat_th_dist
edge_inds = np.argwhere(acc_mat_th_dist > 0)

res = im3.copy()
for b_ind, a_ind, r_ind in edge_inds:
    r0 = r_vec[r_ind]
    a0 = a_vec[a_ind]
    b0 = b_vec[b_ind]

    # draw the outer circle
    res = cv2.circle(res, (int(b0), int(a0)), int(r0), (0, 255, 0), 1)

plt.figure(figsize=figsize)
plt.imshow(res)
plt.title("final result")
plt.show()
# %% [markdown]
# ## Comparison to cv2.HoughCircles
# %%
res = im3.copy()

# explanation can ve found here:
# https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=8, minRadius=5, maxRadius=30)

for xyr in circles[0, :]:
    # draw the outer circle
    res = cv2.circle(res, (xyr[0], xyr[1]), xyr[2], (0, 255, 0), 1)

plt.figure(figsize=figsize)
plt.imshow(res)
plt.title("final result- cv2.HoughCircles")
plt.show()

# %% [markdown]
# Now let's try something a bit more complex...
# Let's identify coins!
# in the image given below we want to detect each coin currency,
# and we'll do it with cv2.HoughCircles!
# %%
im3 = cv2.imread("coins.png")
im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
im = cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)
res = im3.copy()

# 
# detect the right circle dimeter and place
acc_ratio = 1.5
min_dist = 105
canny_upper_th = 100
acc_th = 85

circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, acc_ratio, min_dist, param1=canny_upper_th, param2=acc_th, minRadius=40, maxRadius=65)
#circles = cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, dp=1.5, minDist=105, param1=100, param2=85, minRadius=40, maxRadius=65)
# === font vars
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 0.8
fontColor = (0, 0, 0)
lineType = 2

# ==== for each detected circle
for xyr in circles[0, :]:
    # draw the outer circle
    res = cv2.circle(res, (xyr[0], xyr[1]), xyr[2], (0, 255, 0), 3)
    print(xyr[2])

    coins_name = ['Quarter', 'Nickel', 'Dime']
    if xyr[2] > 60.85:
        cv2.putText(res, coins_name[0], (xyr[0], xyr[1]), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 100))
    elif xyr[2] < 60.85 and xyr[2] > 51.35:
        cv2.putText(res, coins_name[1], (xyr[0], xyr[1]), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
    else:
        cv2.putText(res, coins_name[2], (xyr[0], xyr[1]), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 20))
    
    
   
    pass

plt.figure(figsize=figsize)
plt.imshow(res)
plt.title("final result- coins detection")
plt.show()

# %%
