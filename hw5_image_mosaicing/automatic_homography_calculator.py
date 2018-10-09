import numpy as np
import cv2
import random
from scipy.optimize import least_squares
from calHomography import *

def use_sift(img):
	 gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	 sift = cv2.xfeatures2d.SIFT_create()
	 kp, des = sift.detectAndCompute(gray, None)
	 return kp, des

def use_surf(img):
	 gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	 surf = cv2.xfeatures2d.SURF_create(400)
	 kp, des = surf.detectAndCompute(gray, None)
	 return kp, des

def find_corresponding_points(img0, img1):

	kp0, des0 = use_surf(img0)
	kp1, des1 = use_surf(img1)

	print(len(kp0))
	print(len(kp1))

	# create BFMatcher object
	bf = cv2.BFMatcher()

	# Match descriptors.
	matches = bf.match(des0,des1)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	# here we take only 10 percent of total matches. Only retain best of these matches
	n_reserved_matches = int(0.1 * len(matches))

	# extract matches pointes
	corr_points = []
	for i in range(n_reserved_matches):
		idx0 = matches[i].queryIdx
		idx1 = matches[i].trainIdx
		corr_points.append((kp0[idx0].pt, kp1[idx1].pt))

	# Draw first 10 matches.
	img2 = cv2.drawMatches(img0,kp0,img1,kp1,matches[:n_reserved_matches], None, flags=2)
	
	return img2, corr_points


def RANSAC_1_trial(corr_points, n, delta):
	# Random choosing n correspondences from corr_points and compute Homography
	chosen_corrs = random.sample(corr_points, n)
	H = cal_Homography_Linear_Least_Squares(chosen_corrs)

	# project domain-image into range-image and count the inliers
	inliers = []
	pd = np.zeros((3,1)) # domain-image interesting point
	pr = np.zeros((3,1)) # projected on range-image
	pl = np.zeros((1,2)) # interesting point on second image
	counter = 0
	for i in range(len(corr_points)):
		pd[0][0] = corr_points[i][0][0]
		pd[1][0] = corr_points[i][0][1]
		pd[2][0] = 1.0

		pr = np.matmul(H, pd)
		pr = pr / pr[2][0]
		pr = np.array([pr[0][0], pr[1][0]])

		pl = np.array([corr_points[i][1][0], corr_points[i][1][1]])

		if (np.linalg.norm(pr-pl) < delta):
			inliers.append(corr_points[i])
			counter += 1

	return counter, H, inliers


def RANSAC(corr_points, n, N, M, delta):
	maxcounter = 0
	H = None
	inliers = None
	for i in range(N):
		counter, Hcand, inliers_cand = RANSAC_1_trial(corr_points, n, delta)
		print('counter:', counter, 'M:', M)
		if (counter > maxcounter and counter > M):
			maxcounter = counter
			H = Hcand
			inliers = inliers_cand

	if H is None:
		print('failed to find Homography that will give', M, ' per inliers')
	else:
		print('successfully find a Homography that will give', M, 'per inliers')

	return H, inliers

def residual_func_RANSAC_LM(h, inliers_dom, inliers_ran):
	res = []
	for i in range(len(inliers_dom)):
		x = inliers_dom[i][0]
		y = inliers_dom[i][1]
		f1 = (h[0] * x + h[1] * y + h[2]) / (h[6] * x + h[7] * y + h[8])
		f2 = (h[3] * x + h[4] * y + h[5]) / (h[6] * x + h[7] * y + h[8])

		xp = inliers_ran[i][0]
		yp = inliers_ran[i][1]
		cost = (f1 - xp)**2 + (f2 - yp)**2
		res.append(cost)
	res = np.asarray(res)
	return res

def Nonlinear_Least_Squares_Min(H, inliers):
	inliers = np.asarray(inliers)
	inliers_dom = inliers[:, 0, :] # domain image
	inliers_ran = inliers[:, 1, :] # range image
	h0 = H.flatten()
	res_lsq = least_squares(residual_func_RANSAC_LM, h0, method = 'lm', args = (inliers_dom, inliers_ran))
	h = res_lsq.x
	H = h.reshape((3,3))
	print('Nonlinear_Least_Squares successful ?: ',res_lsq.success)

	return H
