import numpy as np
import cv2
from calHomography import *

#Calculate the Homography
#P = Point(1513, 182)
#Q = Point(2948, 726)
#R = Point(1496, 2244)
#S = Point(2998, 2046)
P = Point(182, 1513)
Q = Point(726, 2948)
R = Point(2244, 1496)
S = Point(2046, 2998)
quad = Quad(P, Q, R, S)

Pp = Point(354, 180)
Qp = Point(929, 180)
Rp = Point(354, 705)
Sp = Point(929, 705)

Pp = Point(180, 354)
Qp = Point(180, 929)
Rp = Point(705, 354)
Sp = Point(705, 929)
quadp = Quad(Pp, Qp, Rp, Sp)

H = cal_Homography(quadp, quad)

#read image
img1 = cv2.imread('./PicsHw2/1.jpg')
imgj = cv2.imread('./PicsHw2/Jackie.jpg')

#decide is p is in PQRS of image 1
def is_inside(quad, p):
	y0 = (- quad.edges[0][0] * p.x - quad.edges[0][2]) / quad.edges[0][1]
	y1 = (- quad.edges[2][0] * p.x - quad.edges[2][2]) / quad.edges[2][1]
	x1 = (- quad.edges[1][1] * p.y - quad.edges[1][2]) / quad.edges[1][0]
	x0 = (- quad.edges[3][1] * p.y - quad.edges[3][2]) / quad.edges[3][0]

	print(x0, x1, y0, y1)
	print(p.x,p.y)

	if p.x >= x0 and p.x <= x1 and p.y >= y0 and p.y <= y1 :
		return True
	return False 

#get RGB when p is not a integer pixel point
def getRGB(pt, img):
	x = int(round(pt.x))
	y = int(round(pt.y))
	return img[x, y]

#to project the image
x0, x1, y0, y1 = quad.get_box()

for x in range(x0,x1+1):
	for y in range(y0,y1+1):
		pt = Point(x,y)
		if is_inside(quad, pt):
			ptp = np.matmul(H, pt.hp.reshape((3,1)))
			ptp = ptp / ptp[2][0]
			ptp = Point(ptp[0][0], ptp[1][0])
			img1[x,y] = getRGB(ptp, imgj)

cv2.imwrite('1j.jpg',img1)
cv2.destroyAllWindows()
