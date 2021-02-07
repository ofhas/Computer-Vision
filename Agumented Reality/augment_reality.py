import cv2
import numpy as np
import os
import pywavefront
import matplotlib.pyplot as plt


def image_proc(img,scale_factor):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    #Luminance channel of HSV image
    lum = img_hsv[:,:,2]

    #Adaptive thresholding
    lum_thresh = cv2.adaptiveThreshold(lum,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,15)

    #Remove all small connected components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(lum_thresh, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 90*scale_factor

    lum_clean = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            lum_clean[output == i + 1] = 255

    # use mask to remove all neat outline of original image
    lum_seg = np.copy(lum)
    lum_seg[lum_clean!=0] = 0
    lum_seg[lum_clean==0] = 255

    # Gaussian smoothing of the lines
    lum_seg = cv2.GaussianBlur(lum_seg,(3,3),1)

    return lum_seg

# Compute the homography
def computeHomography(src_pts, dst_pts):
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

    return M

# Draw lines for frame boundary
def draw_frame(img,dst):

    #img = cv2.polylines(img, [np.int32(dst)], True, cv2.LINE_AA)

    return img

# Feauture identification and matching using BRISK detector and FLANN feature matching
def brisk_flann(img1, img2):
    # Initiate BRISK detector
    brisk = cv2.BRISK_create()
    orb = cv2.ORB()
    #sift = cv2.SIFT()
    kp1, des1 = brisk.detectAndCompute(img1, None)
    kp2, des2 = brisk.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 1
    index_params = dict(algorithm=6,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=1)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 105
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = computeHomography(src_pts, dst_pts)

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

    else:
        print( "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        plt.imshow(des2)
        plt.show()
    return dst_pts, dst

# Plot cube in current frame of video
def plot_cube(img_marked, rvecs, tvecs, camera_matrix, dist_coefs):
    # Cube corner points in world coordinates
    axis8 = np.float32([[0, 0, 0], [32, 0, 0], [32, 32, 0], [0, 32, 0], [0, 0, -32], [32, 0, -32], [32, 32, -32],
                        [0, 32, -32]]).reshape(-1, 3)

    #vert = np.float32(objectToMatrix()).reshape(-1, 3)

    # Project corner points of the cube in image frame
    imgpts, jac = cv2.projectPoints(axis8, rvecs, tvecs, camera_matrix, dist_coefs)

    # Render cube in the video
    # Two faces (top and bottom are shown. They are connected by red lines.
    imgpts = np.int32(imgpts).reshape(-1, 2)
    face1 = imgpts[:4]
    face2 = np.array([imgpts[0], imgpts[1], imgpts[5], imgpts[4]])
    face3 = np.array([imgpts[2], imgpts[3], imgpts[7], imgpts[6]])
    face4 = imgpts[4:]

    # Bottom face
    img = cv2.drawContours(img_marked, [face1], -1, (255, 0, 0), -3)

    # Draw lines connected the two faces
    img = cv2.line(img_marked, tuple(imgpts[0]), tuple(imgpts[4]), (0, 0, 255), 2)
    img = cv2.line(img_marked, tuple(imgpts[1]), tuple(imgpts[5]), (0, 0, 255), 2)
    img = cv2.line(img_marked, tuple(imgpts[2]), tuple(imgpts[6]), (0, 0, 255), 2)
    img = cv2.line(img_marked, tuple(imgpts[3]), tuple(imgpts[7]), (0, 0, 255), 2)

    # Top face
    img = cv2.drawContours(img_marked, [face4], -1, (0, 255, 0), -3)
    #plt.imshow(img_marked)
    #plt.show()
    return img_marked




dataset_path = '/dataset1/'
param_path = '/param/'
template_path = '/templates/'

template_filename = template_path + 'temp.jpg'
video_filename = dataset_path + 'testvid.mp4'
camera_params_filename = param_path + 'cam_intrinsic_distort1.npz'
output_filename = 'Project 2 - Augmented_Reality.mp4'
print(camera_params_filename)
# 3D model points in world coordinates
# Corners of the colouring page
pg_points = np.array([
    (93.0, 135.0, 0.0),  # 1
    (93.0, -135.0, 0.0),  # 2
    (-93.0, -135.0, 0.0),  # 3
    (-93.0, 135.0, 0.0)  # 4
])

# Load camera matrix and distortion coefficient from camera calibration
# The phone camera was calibrated usign checkerboard pattern
cam_params = np.load('cam_intrinsic_distort1.npz')
camera_matrix = cam_params['arr_1']
dist_coefs = cam_params['arr_0']
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Define codec to write output video
# Define the codec and create VideoWriter object
#for fourcc, the ASCII is directly provided as the cv2.VideoWriter_fourcc() doesn't work
out = cv2.VideoWriter(output_filename, fourcc, 29.0,(960, 540))

# The colouring page- original drawing
# In the case of this implementation, an already coloured image was used. The image_proc function is used to remove all
# the colours and extract just the line drawing.
img_org = cv2.imread('temp.jpg')

# Resize image for image size reduction
scale_factor = 0.25
# img1 = image_proc(cv2.resize(img_org,None,fx=scale_factor,fy=scale_factor),scale_factor)
img1 = image_proc(cv2.resize(img_org, (540,960)), scale_factor)

# Load the video and read out the first frame and process it to extract the line drawing
cap = cv2.VideoCapture('testvid.mp4')
_, img_fframe = cap.read()
img_fframe_resize = cv2.resize(img_fframe, None, fx=0.5, fy=0.5)
img2_fframe = image_proc(img_fframe_resize, 0.5)

# STAGE 1, where the features from the first video frame and the template image is identified and matched
# The features identified in this stage is used to track the colouring page in the subsequent video frames

# Feature identification and matching
dst_pts, dst = brisk_flann(img1, img2_fframe)

# draw frame boundary and display in video
img_marked = draw_frame(img_fframe_resize, dst)
cv2.imshow('Video',img_marked)

# STAGE-2 where the features identified will be used to track (Lucas-Kanade optical flow) the colouring in video
# In this stage, we estimate homography and camera pose and use it to render a cube in the video frame in real-time

# Copy feature points and image frame. The feature points from brisk-flann will be used in optical flow tracking
src_pts = np.copy(dst_pts)
img2_old = np.copy(img2_fframe)

# Setup parameters for optical tracking in video
# Parameters for Shi-Tomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
#
# src_pts = cv2.goodFeaturesToTrack(img2_fframe, mask = None, **feature_params)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Read and process frames from video
while True:
    # Write the frame into the file 'output.mp4'
    out.write(img_marked)

    # Read frame
    ret, img_scn = cap.read()

    if ret:
        # Resize frame to smaller size
        img_scn_resize = cv2.resize(img_scn, None, fx=0.5, fy=0.5)

        # Remove all colours and make frame close to original template as possible
        img2 = image_proc(img_scn_resize, 0.5)

        # Calculate optical flow
        dst_pts, st, err = cv2.calcOpticalFlowPyrLK(img2_old, img2, src_pts, None, **lk_params)

        # Select good points
        good_new = dst_pts[st == 1]
        good_old = src_pts[st == 1]

        # Compute Homography
        M = computeHomography(good_old, good_new)

        #Transform frame edge based on new homography
        dst = cv2.perspectiveTransform(dst, M)

        # draw frame boundary and display in video
        img_marked = draw_frame(img_scn_resize, dst)

        # Copy feature points and frame for processing of next frame
        src_pts = np.copy(good_new).reshape(-1,1,2)
        img2_old = np.copy(img2)

        # Estimate the camera pose from frame corner points in world coordinates and image frame
        # THe rotation vectors and translation vectors are obtained
        ret, rvecs, tvecs, inlier_pt = cv2.solvePnPRansac(pg_points, dst, camera_matrix, dist_coefs)

        # Render cube in the video
        # Project cube corners in world coordinates to image frame
        # Two faces (top and bottom are shown. They are connected by red lines.
        img_marked = plot_cube(img_marked, rvecs, tvecs, camera_matrix, dist_coefs)

        # Display in video
        cv2.imshow('Video', img_marked)

        # Press 'q' on keyboard to exit program
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        print ('End of video')
        break

# Close all windows and release video capture object
cv2.destroyAllWindows()
cap.release()
