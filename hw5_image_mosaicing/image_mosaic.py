import numpy as np
import cv2
from automatic_homography_calculator import *
from projectimage import *
from calHomography import *
import math
from copy import copy, deepcopy

#read image
img = []
for i in range(5):
	img_temp = cv2.imread('./Pics/'+str(i)+'.jpg')
	img.append(img_temp)
	img[i] = cv2.resize(img[i], (1008, 756), interpolation = cv2.INTER_CUBIC)
	

# compute correspondences points using SIFT
img_corr = []
set_of_corr_poins = []
for i in range(4):
	img_temp, corr_points = find_corresponding_points(img[i], img[i+1])
	img_corr.append(img_temp)
	set_of_corr_poins.append(corr_points)
	cv2.imwrite('corr'+str(i)+str(i+1)+'.jpg', img_corr[i])

# set parameters
n = 6 # number of corrs to compute Homography
N = 7 # number of trials
delta = 40 # decision threshold for considering inliers
p = 0.99 # probability of at least one of N trial that will reject all the outliers. 
epsilon = 0.1 # the rough estimation of the ratio of number of false corrs to total number of corrs.

Hs = []
for i in range(4):
	M = int(math.ceil(len(set_of_corr_poins[i]) * (1-epsilon)))
	H_init, inliers = RANSAC(set_of_corr_poins[i], n, N, M, delta)
	# next use Levenberg-Maquardt Method to refine the H.
	H = Nonlinear_Least_Squares_Min(H_init, inliers)
	Hs.append(H)


# to change H to accommodate the interchange of x and y of cv2.surf and imread
for i in range(4):
	H_temp = deepcopy(Hs[i])
	Hs[i] = deepcopy(H_temp)
	Hs[i][0][0] = H_temp[1][1]
	Hs[i][0][1] = H_temp[1][0]
	Hs[i][0][2] = H_temp[1][2]

	Hs[i][1][0] = H_temp[0][1]
	Hs[i][1][1] = H_temp[0][0]
	Hs[i][1][2] = H_temp[0][2]

	Hs[i][2][0] = H_temp[2][1]
	Hs[i][2][1] = H_temp[2][0]

# finally project img0, img1, img3, img4 into img2:
H_center = []
H_center.append( np.matmul(Hs[0], Hs[1]) )
H_center.append( Hs[1] )
H_center.append( np.identity(3) )
H_center.append( np.linalg.inv(Hs[2]) )
H_center.append( np.linalg.inv(np.matmul(Hs[2], Hs[3])) )



# do mosaic:
imgc = mosaic_image_out2in(img, H_center)

cv2.imwrite('mosaiced_image.jpg', imgc)
