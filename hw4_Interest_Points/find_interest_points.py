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
	Gx = scipy.ndimage.correlate(gray, hx, mode='nearest')
	Gy = scipy.ndimage.correlate(gray, hy, mode='nearest')

	#compute the squares and products of the gradients at each pixel:
	Gx2 = Gx ** 2
	Gy2 = Gy ** 2
	Gxy = Gx * Gy

	#define window:
	len_window = int(round(5*sigma))
	if (len_window % 2 == 0):
		len_window += 1

	#compute Ratio of DET to Trace:
	half_len_window = (len_window - 1) // 2
	R = np.zeros(np.shape(gray))
	lenx, leny = np.shape(gray)
	C = np.zeros((2,2))
	for i in range(half_len_window, lenx - half_len_window + 1):
		for j in range(half_len_window, leny - half_len_window + 1):
			C[0][0] = np.sum(get_square_window(Gx2, half_len_window, [i, j]))
			C[0][1] = np.sum(get_square_window(Gxy, half_len_window, [i, j]))
			C[1][0] = C[0][1]
			C[1][1] = np.sum(get_square_window(Gy2, half_len_window, [i, j]))
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
			local_maxima = np.amax(R[i - halflen_local_maxima, i + halflen_local_maxima + 1])
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

	print('corner0', len(list_Corner0))
	print('corner1', len(list_Corner1))

	for i in range(len(list_Corner0)):
		for j in range(len(list_Corner1)):
			coo0 = list_Corner0[i]
			coo1 = list_Corner1[j]
			f0 = get_square_window(gray0, halflen_corr_window, coo0)
			f1 = get_square_window(gray1, halflen_corr_window, coo1)
			ssd = np.sum( (f0 - f1)**2 )
			list_ssd_cand.append([coo0, coo1, ssd])

			mean0 = f0 / np.sum(f0)
			mean1 = f1 / np.sum(f1)
			ncc = np.sum( (f0 - mean0) * (f1 - mean1) ) / math.sqrt( (f0 - mean0)**2 * (f1 - mean1)**2 )
			if (ncc > ncc_threshold):
				list_ncc.append([coo0, coo1])

	ssdmin = min(list_ssd_cand, key = get_last_element)

	for i in range(len(list_ssd_cand)):
		if (list_ssd_cand[i][2] < 10 * ssdmin):
			list_ssd.append(list_ssd_cand[:2])

	return list_ssd, list_ncc

def output_two_imgs_with_corr_points(img0, img1, corr_points):
	lenx0, leny0 = np.shape(img0)
	img_o = np.concatenate((img0, img1), axis = 1)
	for i in range(len(corr_points)):
		rightpoint = [corr_points[i][1][0], corr_points[i][1][1] + leny0]
		cv2.line(img_o, corr_points[i][0], rightpoint, (255, 0, 0), 3)
	return img_o

