import numpy as np
import cv2
from calHomography import *
from projectimage import *

P = Point(822, 1140)
Q = Point(762, 1254)
R = Point(990, 1125)
S = Point(942, 1245)
quad = Quad(P, Q, R, S)

P2 = Point(72, 245)
Q2 = Point(83, 327)
R2 = Point(272, 245)
S2 = Point(269, 323)
quad2 = Quad(P2, Q2, R2, S2)

Pc = Point(100, 100)
Qc = Point(100, 140)
Rc = Point(180, 100)
Sc = Point(180, 140)
quadc = Quad(Pc, Qc, Rc, Sc)

def P2PCorr(quad, quadc, img):
	# Calculate the Homography
	Hc = cal_Homography(quadc, quad) # Calculate the Homography of quad to quadc
	
	size = img.shape


	# find the boundary of corrected picture
	Pb = Point(0, 0)
	Qb = Point(0, size[1])
	Rb = Point(size[0], 0)
	Sb = Point(size[0], size[1])

	#Pb2 = Point(0, size[1]/2)
	#Qb2 = Point(size[0]/2, size[1])
	#Rb2 = Point(size[0], size[1]/2)
	#Sb2 = Point(size[0]/2, 0)

	quadb = Quad(Pb, Qb, Rb, Sb)
	box = find_projected_boundary(quadb, Hc)

	#quadb2 = Quad(Pb2, Qb2, Rb2, Sb2)
	#box2 = find_projected_boundary(quadb2, Hc)

	#box.bd[0] = min(box.bd[0], box2.bd[0])
	#box.bd[1] = max(box.bd[1], box2.bd[1])
	#box.bd[2] = min(box.bd[2], box2.bd[2])
	#box.bd[3] = max(box.bd[3], box2.bd[3])

	#correct distortion:
	sizec = [box.bd[1]-box.bd[0]+2, box.bd[3]-box.bd[2]+2, 3]
	imgc = np.zeros(sizec, np.uint8)

	for x in range(0, size[0]):
		for y in range(0, size[1]):
			pt = Point(x,y)
			ptc = np.matmul(Hc, pt.hp.reshape((3,1)))
			ptc = ptc / ptc[2][0] 
			ptcx = int(round(ptc[0][0]))
			ptcy = int(round(ptc[1][0]))
			#print(x, y, ptcx, ptcy)
			if (ptcx < box.bd[1]) and (ptcx > box.bd[0]) and (ptcy > box.bd[2]) and (ptcy < box.bd[3]):
				imgc[ptcx-box.bd[0], ptcy-box.bd[2]] = getRGB(pt, img)

	return imgc

img = cv2.imread('./HW3Pics/2.jpg')
imgc = P2PCorr(quad2, quadc, img)
cv2.imwrite('o.jpg',imgc)
cv2.destroyAllWindows()
