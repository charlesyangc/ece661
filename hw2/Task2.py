import numpy as np
import cv2
from calHomography import *
from projectimage import *

#Task 2(a)
#NOTICE if using GIMP to get coordinates of points, the (x, y) is different from python cv2.imread function
P = Point(102, 699)
Q = Point(309, 1422)
R = Point(1392, 663)
S = Point(1299, 1494)
quad = Quad(P, Q, R, S)

P2 = Point(339, 546)
Q2 = Point(126, 1275)
R2 = Point(1347, 492)
S2 = Point(1443, 1308)
quad2 = Quad(P2, Q2, R2, S2)

P3 = Point(195, 600)
Q3 = Point(159, 1446)
R3 = Point(1275, 531)
S3 = Point(1293, 1512)
quad3 = Quad(P3, Q3, R3, S3)

Pp = Point(12, 692)
Qp = Point(12, 1262)
Rp = Point(790, 692)
Sp = Point(790, 1262)
quadp = Quad(Pp, Qp, Rp, Sp)

#read image
img1 = cv2.imread('./PicsHw2/4.jpg')
img2 = cv2.imread('./PicsHw2/5.jpg')
img3 = cv2.imread('./PicsHw2/6.jpg')
imgj = cv2.imread('./PicsHw2/NicolasCage.jpg')
DoProject(img1, imgj, quad, quadp)
DoProject(img2, imgj, quad2, quadp)
DoProject(img3, imgj, quad3, quadp)
cv2.imwrite('4j.jpg',img1)
cv2.imwrite('5j.jpg',img2)
cv2.imwrite('6j.jpg',img3)
cv2.destroyAllWindows()


#Task 1(b)
Hab = cal_Homography(quad2, quad)
Hbc = cal_Homography(quad3, quad2)
H = np.matmul(Hab, Hbc)
img1 = cv2.imread('./PicsHw2/4.jpg')
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

cv2.imwrite('6p.jpg',img3p)
cv2.destroyAllWindows()
