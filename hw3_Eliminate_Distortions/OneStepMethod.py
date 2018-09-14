import numpy as np
from calHomography import *
import cv2
from projectimage import *
from numpy import linalg as LA

def Cal_Homography_One_Step(perpen_lines_set):
	M = np.zeros((5,5), dtype = 'float')
	b = np.zeros((5,1), dtype = 'float')

	for i in range(5):
		l = perpen_lines_set[i].l
		m = perpen_lines_set[i].m

		M[i][0] = l[0] * m[0]
		M[i][1] = (l[0] * m[1] + l[1] * m[0]) / 2 
		M[i][2] = l[1] * m[1]
		M[i][3] = (l[0] * m[2] + l[2] * m[0]) / 2
		M[i][4] = (l[1] * m[2] + l[2] * m[1]) / 2
		b[i][0] = - l[2] * m[2]

	c = np.matmul(np.linalg.inv(M), b)
	C = np.array([[c[0][0],   c[1][0]/2, c[3][0]/2], 
				  [c[1][0]/2, c[2][0],   c[4][0]/2], 
				  [c[3][0]/2, c[4][0]/2, 1.0]], dtype = 'float')

	# U, d, V = np.linalg.svd(C) #H here is from undistorted to distorted
	# Hinv = U
	# #U = U / np.sqrt(d[0])
	# H = LA.inv(U) # H here is from distorted to undistorted 

	S = np.array([[C[0][0], C[0][1]], [C[1][0], C[1][1]]], dtype = 'float')
	U, d, V = LA.svd(S)
	D = np.diag(np.sqrt(d))
	A = np.matmul(U, np.matmul(D, V))
	t = np.array([ C[2][0], C[2][1] ], dtype = 'float')
	Ainv = LA.inv(A)
	v = np.matmul(t, Ainv.transpose())
	H = np.array( [[A[0][0], A[0][1], 0], 
				   [A[1][0], A[1][1], 0], 
				   [v[0],    v[1],    1]], dtype = 'float' )
	H = H
	Hinv = LA.inv(H)
	return H, Hinv

#first find five sets of perpandicular lines:
A = Point(1245, 246)
B = Point(1356, 231)
C = Point(1356, 231)
D = Point(1230, 294)
lines0 = PerpendicularLines(A, B, C, D)

A1 = Point(1212, 342)
B1 = Point(1182, 393)
C1 = Point(1182, 393)
D1 = Point(1299, 369)
lines1 = PerpendicularLines(A1, B1, C1, D1)

A2 = Point(1524, 234)
B2 = Point(1629, 213)
C2 = Point(1629, 213)
D2 = Point(1638, 171)
lines2 = PerpendicularLines(A2, B2, C2, D2)

A3 = Point(1503, 285)
B3 = Point(1623, 258)
C3 = Point(1623, 258)
D3 = Point(1614, 312)
lines3 = PerpendicularLines(A3, B3, C3, D3)

A4 = Point(1113, 828)
B4 = Point(1083, 918)
C4 = Point(1083, 918)
D4 = Point(1215, 903)
lines4 = PerpendicularLines(A4, B4, C4, D4)

#find five sets of perpanducular lines for pic 2
# A = Point(75, 247)
# B = Point(84, 326)
# C = Point(75, 247)
# D = Point(272, 245)
# lines0 = PerpendicularLines(A, B, C, D)

# A1 = Point(268, 325)
# B1 = Point(84, 326)
# C1 = Point(268, 325)
# D1 = Point(272, 244)
# lines1 = PerpendicularLines(A1, B1, C1, D1)

# A2 = Point(137, 58)
# B2 = Point(141, 128)
# C2 = Point(137, 58)
# D2 = Point(234, 57)
# lines2 = PerpendicularLines(A2, B2, C2, D2)

# A3 = Point(232, 126)
# B3 = Point(141, 128)
# C3 = Point(232, 126)
# D3 = Point(234, 57)
# lines3 = PerpendicularLines(A3, B3, C3, D3)

# A4 = Point(162, 405)
# B4 = Point(167, 440)
# C4 = Point(162, 405)
# D4 = Point(224, 405)
# lines4 = PerpendicularLines(A4, B4, C4, D4)

linesset = [lines0, lines1, lines2, lines3, lines4]

H, Hinv = Cal_Homography_One_Step(linesset)
img = cv2.imread('./HW3Pics/1.jpg')
imgc = correct_distorted_image_out2in(img, H, Hinv)
cv2.imwrite('11.jpg',imgc)
cv2.destroyAllWindows()
