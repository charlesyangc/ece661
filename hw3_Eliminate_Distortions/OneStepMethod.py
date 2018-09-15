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

	C = C / np.amax(C)
	# U, d, V = np.linalg.svd(C) #H here is from undistorted to distorted
	# Hinv = U
	# H = LA.inv(U) # H here is from distorted to undistorted 

	S = np.array([[C[0][0], C[0][1]], [C[1][0], C[1][1]]], dtype = 'float')
	U, d, V = LA.svd(S)
	print('U', U)
	print('V,', V)
	D = np.diag(np.sqrt(d))
	Vt = V.transpose()
	A = np.matmul(Vt, np.matmul(D, V))
	t = np.array([ C[2][0], C[2][1] ], dtype = 'float')
	Ainv = LA.inv(A)
	v = np.matmul(t, Ainv.transpose())
	H = np.array( [[A[0][0], A[0][1], 0], 
				   [A[1][0], A[1][1], 0], 
				   [v[0],    v[1],    1]], dtype = 'float' )
	Hinv = H
	H = LA.inv(H)
	return H, Hinv

#five sets of perpandicular lines for Pic. 1:
A = Point(1681,594)
B = Point(1670,670)
C = Point(1775,656)
D = Point(1784,579)

lines0 = PerpendicularLines(A,B,B,C)
lines1 = PerpendicularLines(B,C,C,D)
lines2 = PerpendicularLines(A,C,B,D)
lines3 = PerpendicularLines(C,D,D,A)
lines4 = PerpendicularLines(D,A,A,B)

#five sets of perpandicular lines for Pic. 2:
# A = Point(71,245)
# B = Point(82,326)
# C = Point(265,323)
# D = Point(269,244)

# lines0 = PerpendicularLines(A, B, B, C)
# lines1 = PerpendicularLines(B,C,C,D)
# lines2 = PerpendicularLines(C,D,D,A)
# #left
# A1 = Point(161,404)
# B1 = Point(163,436)
# C1 = Point(223,437)
# D1 = Point(223,404)
# lines3 = PerpendicularLines(A1, C1, B1, D1)

# A1 = Point(135,58)
# B1 = Point(139,126)
# C1 = Point(230,126)
# D1 = Point(231,59)
# lines4 = PerpendicularLines(A1, C1, B1, D1)

#five sets of perpandicular lines for pic. 8
A = Point(1894, 1408)
B = Point(1767, 1844)
C = Point(2366, 1607)
D = Point(2192, 2058)

lines0 = PerpendicularLines(A,B,B,D)
lines1 = PerpendicularLines(B,D,D,C)
lines2 = PerpendicularLines(A,D,B,C)
lines3 = PerpendicularLines(D,C,C,A)
lines4 = PerpendicularLines(C,A,A,B)

# for PIC 15
A = Point(249, 363)
B = Point(245, 384)
C = Point(288, 363)
D = Point(281, 384)

lines0 = PerpendicularLines(A,B,B,D)
lines1 = PerpendicularLines(B,D,D,C)
lines2 = PerpendicularLines(A,D,B,C)
lines3 = PerpendicularLines(D,C,C,A)
lines4 = PerpendicularLines(C,A,A,B)


linesset = [lines0, lines1, lines2, lines3, lines4]

H, Hinv = Cal_Homography_One_Step(linesset)
print('H', H)
img = cv2.imread('./HW3Pics/15.jpg')
imgc = correct_distorted_image_out2in(img, H)
cv2.imwrite('151s.jpg',imgc)
cv2.destroyAllWindows()
