import numpy as np
import cv2

def construct_int_from_bin(bin):
	r = 0
	for i in range(len(bin)):
		r = r + bin[i] * ( 2 ** (len(bin) - i - 1) )
	return r

def Bilinear_interpolation(delta_k, delta_l, A, B, C, D):
	r = (1 - delta_k) * (1 - delta_l) * A + (1 - delta_k) * delta_l * B + delta_k * (1 - delta_l) * C + delta_k * delta_l * D
	return r

def get_binary_pattern(M):
	delta = np.sqrt(2) / 2
	pattern = np.zeros(8)
	pattern[0] = M[2][1]
	pattern[2] = M[1][2]
	pattern[4] = M[0][1]
	pattern[6] = M[1][0]
	pattern[1] = Bilinear_interpolation(delta, delta, M[1][1], M[1][2], M[2][1], M[2][2])
	pattern[3] = Bilinear_interpolation(1-delta, delta, M[0][1], M[0][2], M[1][1], M[1][2])
	pattern[5] = Bilinear_interpolation(1-delta, 1-delta, M[0][0], M[0][1], M[1][0], M[1][1])
	pattern[7] = Bilinear_interpolation(delta, 1-delta, M[1][0], M[1][1], M[2][0], M[2][1])

	for i in range(8):
		if (pattern[i] >= M[1][1]):
			pattern[i] = 1
		else:
			pattern[i] = 0

	return pattern

def get_minbv(pattern):
	P = len(pattern)
	minbv = construct_int_from_bin(pattern)
	for i in range(P):
		container = pattern[0]
		for j in range(P - 1):
			pattern[j] = pattern[j + 1]
		pattern[P - 1] = container

		minbv_cand = construct_int_from_bin(pattern)
		if (minbv_cand < minbv):
			minbv = minbv_cand

	return minbv

def Encode_minbv(minbv, P):
	minintval = [int(x) for x in bin(minbv)[2:]]
	l = len(minintval)
	if (l != P):
		# meaning there are zeros before the list
		if (l == 1):
			if (minintval[0] == 0):
				return 0
			else:
				return 1
		else:
			# 1 < l < P, first element in minintval is 1
			for i in range(l):
				if (minintval[i] == 0):
					# more than two runs
					return (P + 1)
			return l
	else:
		# meaning there are all ones
		return l

def LBP(img):

	P = 8 #P is the number of surrounding points for a pixel.
	lbp_hist = {t:0 for t in range(P+2)}

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	(rows, cols) = gray.shape
	for i in range(1, rows-1):
		for j in range(1, cols-1):
			binary_pattern = get_binary_pattern(gray[i-1:i+2, j-1:j+2])
			minbv = get_minbv(binary_pattern)
			code = Encode_minbv(int(minbv), P)
			lbp_hist[code] += 1

	# next normalized the lbp_hist
	s = sum(lbp_hist.values())
	for key, value in lbp_hist.items():
		lbp_hist[key] = lbp_hist[key] / s
	return lbp_hist

def Euclidean_dist_dict(a, b):
	r = 0
	for i in range(len(a)):
		r = r + (a[i] - b[i]) ** 2
	return r

def kNN_testing(data, labels, instance, k):
	l = len(data)
	arr_dist = np.zeros(l)
	for i in range(l):
		dist = Euclidean_dist_dict(data[i], instance)
		arr_dist[i] = dist

	sorted_indice = np.argsort(arr_dist)

	classCount = {}
	for i in range(k):
		ind = sorted_indice[i]
		label = labels[ind]
		if label in classCount:
			classCount[label] += 1
		else:
			classCount[label] = 1
	
	sortedcount = sorted(classCount.items(), key=lambda kv: kv[1], reverse=True)
	return sortedcount[0][0]