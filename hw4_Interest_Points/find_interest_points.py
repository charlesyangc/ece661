import numpy as np
import scipy.ndimage
import math
import cv2

def get_haar_filter(sigma):
	N = math.ceil(4 * sigma)
	if (N % 2 == 1):
		N += 1;

	hx = np.zeros((N, N))
	for i in range(N):
		for j in range(N//2):
			hx[i][j] = -1
		for j in range(N//2, N):
			hx[i][j] = 1

	hy = np.zeros((N, N))
	for j in range(N):
		for i in range(N//2):
			hy[i][j] = 1
		for i in range(N//2, N):
			hy[i][j] = -1
	return hx, hy, N

def get_square_window(m, n, coo):
	#return a window of m centered at (i, j) with size (2*n+1)*(2*n+1)
	return m[coo[0]-n:coo[0]+n+1, coo[1]-n:coo[1]+n+1]

def harris_corner_detector(img, sigma, threshold):
	#compute haar filter
	hx, hy, N_haar_filter = get_haar_filter(sigma)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#convolve the img with haar_filter
	Gx = scipy.ndimage.convolve(gray, hx, mode='nearest')
	Gy = scipy.ndimage.convolve(gray, hy, mode='nearest')

	minGx = np.amin(Gx)
	maxGx = np.amax(Gx)
	Gx = (Gx - minGx) / (maxGx - minGx)
	minGy = np.amin(Gy)
	maxGy = np.amax(Gy)
	Gy = (Gy - minGy) / (maxGy - minGy)

	#compute the squares and products of the gradients at each pixel:
	Gx2 = Gx ** 2
	Gy2 = Gy ** 2
	Gxy = Gx * Gy

	#define window:
	len_conv_window = int(round(5*sigma))
	if (len_conv_window % 2 == 0):
		len_conv_window += 1

	#compute Ratio of DET to Trace:
	halflen_conv_window = (len_conv_window - 1) // 2
	R = np.zeros(np.shape(gray))
	lenx, leny = np.shape(gray)
	C = np.zeros((2,2))
	for i in range(halflen_conv_window, lenx - halflen_conv_window + 1):
		for j in range(halflen_conv_window, leny - halflen_conv_window + 1):
			C[0][0] = np.sum(get_square_window(Gx2, halflen_conv_window, [i, j]))
			C[0][1] = np.sum(get_square_window(Gxy, halflen_conv_window, [i, j]))
			C[1][0] = C[0][1]
			C[1][1] = np.sum(get_square_window(Gy2, halflen_conv_window, [i, j]))
			if (np.trace(C) != 0):
				R[i][j] = np.linalg.det(C) / (np.trace(C) * np.trace(C) )
			else:
				R[i][j] = 0.0


	R_mean = np.mean(R)
	# next find local maxima
	C = np.zeros(gray.shape, dtype = 'int8')
	list_Corner = []
	len_local_maxima = 29
	halflen_local_maxima = (len_local_maxima - 1) // 2
	for i in range(halflen_local_maxima, lenx - halflen_local_maxima + 1):
		for j in range(halflen_local_maxima, leny - halflen_local_maxima + 1):
			local = get_square_window(R, halflen_local_maxima, [i, j])
			local_maxima = np.amax(local)
			if ( (R[i][j] == local_maxima) and (R[i][j] > 0 ) and ( abs(R[i][j]) > abs(R_mean) ) and (abs(R[i][j]) > threshold) ):
				C[i][j] = 1 
				list_Corner.append([i, j])

	return list_Corner

def get_last_element(a):
	return a[-1]

def find_correspondent_points(gray0, gray1, list_Corner0, list_Corner1, ncc_threshold):
	list_ssd_cand = []
	list_ssd = []
	list_ncc = []
	len_corr_window = 21
	halflen_corr_window = (len_corr_window - 1) // 2

	for i in range(len(list_Corner0)):
		for j in range(len(list_Corner1)):
			coo0 = list_Corner0[i]
			coo1 = list_Corner1[j]
			f0 = get_square_window(gray0, halflen_corr_window, coo0)
			f1 = get_square_window(gray1, halflen_corr_window, coo1)
			ssd = np.sum( (f0 - f1)**2 )
			list_ssd_cand.append([coo0, coo1, ssd])

			mean0 = np.sum(f0) / np.size(f0)
			mean1 = np.sum(f1) / np.size(f1)
			ncc = np.sum( (f0 - mean0) * (f1 - mean1) ) / math.sqrt( np.sum((f0 - mean0)**2) * np.sum((f1 - mean1)**2) )
			if (ncc > ncc_threshold):
				list_ncc.append([coo0, coo1])

	ssdmin = min(list_ssd_cand, key = get_last_element)

	for i in range(len(list_ssd_cand)):
		if (list_ssd_cand[i][2] < 2 * ssdmin[2]):
			list_ssd.append([list_ssd_cand[i][0], list_ssd_cand[i][1]])

	return list_ssd, list_ncc

def output_two_imgs_with_corr_points(img0, img1, corr_points):
	size = np.shape(img0)
	lenx0 = size[0]
	leny0 = size[1]
	img_o = np.concatenate((img0, img1), axis = 1)
	for i in range(len(corr_points)):
		leftpoint = (corr_points[i][0][1], corr_points[i][0][0])
		rightpoint = (corr_points[i][1][1] + leny0, corr_points[i][1][0])
		cv2.line(img_o, leftpoint, rightpoint, (255, 0, 0), 1)

	return img_o

def use_sift(img):
	 gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	 sift = cv2.xfeatures2d.SIFT_create(10000)
	 kp, des = sift.detectAndCompute(gray, None)
	 return kp, des

def use_surf(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	surf = cv2.xfeatures2d.SURF_create(1000)
	kp, des = surf.detectAndCompute(img, None)
	return kp, des
