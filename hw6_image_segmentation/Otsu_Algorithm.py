import numpy as np
import cv2

def OtsuMethod_OnePass(gray, mask):
	L = np.amax(gray)

	print('L', L)

	[rows, cols] = gray.shape

	# first get ni
	n = np.zeros(L+1)
	for i in range(rows):
		for j in range(cols):
			n[gray[i][j]] += 1

	# first compute pi
	N = gray.size
	p = np.zeros(L+1)
	for i in range(L+1):
		p[i] = n[i] / N

	# first compute first order moment:
	omega = np.zeros(L+1)
	omega[0] = p[0]
	for k in range(L+1):
		omega[k] = omega[k-1] + p[k]

	# compute second order moment:
	mu = np.zeros(L+1)
	mu[0] = 0
	for k in range(L+1):
		mu[k] = mu[k-1] + k * p[k]

	muT = mu[L]

	# l and L is the range of the gray level.
	simga2 = 0
	simga2_temp = 0
	kp = 0
	for k in range(L+1):
		if ( omega[k] > 0 and omega[k] < 1):
			simga2_temp = (muT*omega[k] - mu[k])**2 / (omega[k] * (1 - omega[k]))	
			if (simga2_temp > simga2):
				simga2 = simga2_temp
				kp = k

	print('kp', kp)

	# get mask and apply mask:
	for i in range(rows):
		for j in range(cols):
			if gray[i][j] > kp:
				mask[i][j] = 1
			else:
				gray[i][j] = 0;

	return kp, mask, gray

def OtsuMethod(gray, N_iter):
	for i in range(N_iter):
		mask = np.zeros(gray.shape)
		kp, mask, gray = OtsuMethod_OnePass(gray, mask)
	return mask

def RGB_segment(img, Flag_Neg):
	RGB_iter = [1, 1, 1]
	RGB_mask = []
	for i in range(3): # for R, G, B channels respectively.
		gray = img[:,:,i]
		RGB_mask.append( OtsuMethod(gray, RGB_iter[i]) )
		cv2.imwrite('mask'+str(i)+'.jpg', 255*RGB_mask[i])

	# next 'AND' three masks according to Flag_Neg:
	for i in range(3):
		if (Flag_Neg[i] == 1):
			RGB_mask[i] = 1 - RGB_mask[i]

	mask = RGB_mask[0] * RGB_mask[1] * RGB_mask[2] 
	cv2.imwrite('RGB_mask.jpg', 255*mask)
	return mask

def get_texture(gray, n): # n is the length of the window
	texture = np.zeros(gray.shape)
	(rows, cols) = gray.shape
	img_window = np.zeros((n, n))
	s = n // 2
	print('s', s)
	for i in range(s, rows - s):
		for j in range(s, cols - s):
			img_window = gray[i-s:i+s+1, j-s:j+s+1]
			texture[i][j] = np.var(img_window, dtype=np.float64)

	# scale to [0, 255]
	texture_max = np.amax(texture)
	print('texture_max', texture_max)
	texture = texture / texture_max * 255
	texture = texture.astype(int)
	return texture


def texture_segment(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	texture_iter = [1, 1, 1]
	texture_mask = []
	for i in range(3):
		texture = get_texture(gray, 2*i+3)
		texture_mask.append( OtsuMethod(texture, texture_iter[i]) )
		cv2.imwrite('texture'+str(2*i+3)+'.jpg', texture)
		cv2.imwrite('texture_mask'+str(2*i+3)+'.jpg', 255*texture_mask[i])

	mask = texture_mask[0] * texture_mask[1] * texture_mask[2]
	cv2.imwrite('texture_segment.jpg', 255*mask)
	return mask

def contour_extraction(mask):
	(rows, cols) = mask.shape
	tst_mask = np.ones((3,3))
	tst = np.ones((3,3))
	contour = np.zeros((rows, cols))
	for i in range(1,rows-1):
		for j in range(1, cols-1):
			tst = tst_mask * mask[i-1:i+2, j-1:j+2]
			if ( tst.all() ):
				contour[i][j] = 0
			else:
				contour[i][j] = mask[i][j]

	return contour
