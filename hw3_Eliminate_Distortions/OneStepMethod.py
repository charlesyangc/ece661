import numpy as np
from calHomography import *
import cv2
from projectimage import *
from numpy import linalg as LA

#first find five sets of perpandicular lines:
pt00 = Point(501, 1851)
pt01 = Point(405, 2031)
pt02 = Point(699, 1848)
ptsets0 = [pt00, pt01, pt02]

pt10 = Point(774, 1674)
pt11 = Point(702, 1848)
pt12 = Point(1071, 1665)
ptsets1 = [pt10, pt11, pt12]

pt20 = Point(1128, 1506)
pt21 = Point(1074, 1665)
pt22 = Point(1350, 1494)
ptsets2 = [pt20, pt21, pt22]

pt30 = Point(1383, 1347)
pt31 = Point(1350, 1497)
pt32 = Point(1695, 1323)
ptsets3 = [pt30, pt31, pt32]

pt40 = Point(1713, 1188)
pt41 = Point(1704, 1332)
pt42 = Point(1899, 1176)
ptsets4 = [pt40, pt41, pt42]

#find five sets of perpanducular lines for pic 2
pt00 = Point(75, 247)
pt01 = Point(84, 326)
pt02 = Point(272, 245)
ptsets0 = [pt00, pt01, pt02]

pt10 = Point(268, 325)
pt11 = Point(84, 326)
pt12 = Point(272, 244)
ptsets1 = [pt10, pt11, pt12]

pt20 = Point(137, 58)
pt21 = Point(141, 128)
pt22 = Point(234, 57)
ptsets2 = [pt20, pt21, pt22]

pt30 = Point(232, 126)
pt31 = Point(141, 128)
pt32 = Point(234, 57)
ptsets3 = [pt30, pt31, pt32]

pt40 = Point(162, 405)
pt41 = Point(167, 440)
pt42 = Point(224, 405)
ptsets4 = [pt40, pt41, pt42]

setofsets = [ptsets0, ptsets1, ptsets2, ptsets3, ptsets4]

M = np.zeros((5,5), dtype = 'float')
b = np.zeros((5,1), dtype = 'float')

for i in range(5):
	l = np.cross( setofsets[i][0].hp, setofsets[i][1].hp)
	l = l / np.linalg.norm(l)
	m = np.cross( setofsets[i][0].hp, setofsets[i][2].hp) 
	m = m / np.linalg.norm(m)
	M[i][0] = l[0] * m[0]
	M[i][1] = (l[0] * m[1] + l[1] * m[0]) / 2 
	M[i][2] = l[1] * m[1]
	M[i][3] = (l[0] * m[2] + l[2] * m[0]) / 2
	M[i][4] = (l[1] * m[2] + l[2] * m[1]) / 2
	b[i][0] = - l[2] * m[2]

c = np.matmul(np.linalg.inv(M), b)
#c = c / np.linalg.norm(c)
C = np.array([[c[0][0], c[1][0]/2, c[3][0]/2], [c[1][0]/2, c[2][0], c[4][0]/2], [c[3][0]/2, c[4][0]/2, 1.0]], dtype = 'float')


U, d, V = np.linalg.svd(C) #H here is from correct to distorted
H = LA.inv(U) # H here is from distorted to correct 
#H = U

print('d', d)

conic = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]


lp = np.cross(pt00.hp, pt01.hp)
mp = np.cross(pt00.hp, pt02.hp)
HnT = np.linalg.inv(np.transpose(H))
print(lp, mp)
l = np.matmul(HnT, lp.reshape((3,1)))
m = np.matmul(HnT, mp.reshape((3,1)))

print(l.reshape(1,3), m.reshape(1,3))
Cinf = np.array([ [1, 0, 0], [0, 1, 0], [0, 0, 0] ])
tst = np.matmul(l.transpose(), np.matmul(Cinf, m)  )
print('tst', tst)



img = cv2.imread('./HW3Pics/2.jpg')
imgc = correct_distorted_image(img, H)
cv2.imwrite('21.jpg',imgc)
cv2.destroyAllWindows()
