import numpy as np
import cv2
from calHomography import *
from projectimage import *

#Task 1(a)
#NOTICE if using GIMP to get coordinates of points, the (x, y) is different from python cv2.imread function
P = Point(182, 1513)
Q = Point(726, 2948)
R = Point(2244, 1496)
S = Point(2046, 2998)
quad = Quad(P, Q, R, S)

P2 = Point(347, 1326)
Q2 = Point(627, 3014)
R2 = Point(2019, 1298)
S2 = Point(1898, 3025)
quad2 = Quad(P2, Q2, R2, S2)

P3 = Point(743, 913)
Q3 = Point(374, 2811)
R3 = Point(2079, 902)
S3 = Point(2228, 2838)
quad3 = Quad(P3, Q3, R3, S3)

Pp = Point(180, 354)
Qp = Point(180, 929)
Rp = Point(705, 354)
Sp = Point(705, 929)
quadp = Quad(Pp, Qp, Rp, Sp)

#read image
img1 = cv2.imread('./PicsHw2/1.jpg')
img2 = cv2.imread('./PicsHw2/2.jpg')
img3 = cv2.imread('./PicsHw2/3.jpg')
imgj = cv2.imread('./PicsHw2/Jackie.jpg')
DoProject(img1, imgj, quad, quadp)
DoProject(img2, imgj, quad2, quadp)
DoProject(img3, imgj, quad3, quadp)
cv2.imwrite('1j.jpg',img1)
cv2.imwrite('2j.jpg',img2)
cv2.imwrite('3j.jpg',img3)
cv2.destroyAllWindows()


#Task 1(b)
Hab = cal_Homography(quad2, quad)
Hbc = cal_Homography(quad3, quad2)
H = np.matmul(Hab, Hbc)
img1 = cv2.imread('./PicsHw2/1.jpg')
img3p = np.zeros(img1.shape, np.uint8)
#to project the image
x0, x1, y0, y1 = quad.get_box()

for x in range(x0,x1+1):
	for y in range(y0,y1+1):
		pt = Point(x,y)
		if is_inside(quad, pt):
			ptp = np.matmul(H, pt.hp.reshape((3,1)))
			ptp = ptp / ptp[2][0]
			#ptp = Point(ptp[0][0], ptp[1][0])
			ptpx = int(round(ptp[0][0]))
			ptpy = int(round(ptp[1][0]))
			img3p[ptpx, ptpy] = getRGB(pt, img1)

cv2.imwrite('3p.jpg',img3p)
cv2.destroyAllWindows()

