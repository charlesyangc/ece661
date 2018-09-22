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


def harris_corner_detector(img, sigma):
	hx, hy, N_haar_filter = get_haar_filter(sigma)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#convolve the img with haar_filter
	Gx = scipy.ndimage.correlate(gray, hx, mode='nearest')
	Gy = scipy.ndimage.correlate(gray, hy, mode='nearest')

	#compute the squares and products of the gradients at each pixel:
	Gx2 = Gx ** 2
	Gy2 = Gy ** 2
	Gxy = Gx * Gy
	return gray


img = cv2.imread('./HW4Pics/pair1/1.jpg')
gray = harris_corner_detector(img, math.sqrt(2))
#cv2.imshow('gray', gray)
cv2.destroyAllWindows()