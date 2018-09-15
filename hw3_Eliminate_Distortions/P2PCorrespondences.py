import numpy as np
import cv2
from calHomography import *
from projectimage import *

# For PIC1
P = Point(822, 1140)
Q = Point(762, 1254)
R = Point(990, 1125)
S = Point(942, 1245)
quad = Quad(P, Q, R, S)

# For PIC2
P = Point(72, 245)
Q = Point(83, 327)
R = Point(272, 245)
S = Point(269, 323)
quad = Quad(P, Q, R, S)

#for pic 8
P = Point(1894, 1408)
Q = Point(1768, 1844)
R = Point(2366, 1604)
S = Point(2188, 2056)
quad = Quad(P, Q, R, S)

#for PIC 15
P = Point(249, 363)
Q = Point(245, 384)
R = Point(288, 363)
S = Point(281, 384)
quad = Quad(P, Q, R, S)

Pc = Point(100, 100)
Qc = Point(100, 120)
Rc = Point(120, 100)
Sc = Point(120, 120)
quadc = Quad(Pc, Qc, Rc, Sc)

def P2PCorr(quad, quadc, img):
	# Calculate the Homography
	Hc = cal_Homography(quadc, quad) # Calculate the Homography of quad to quadc
	#imgc = correct_distorted_image_4ptsDetBd(img, Hc)
	imgc = correct_distorted_image_out2in(img, Hc)
	return imgc



img = cv2.imread('./HW3Pics/15.jpg')
imgc = P2PCorr(quad, quadc, img)
cv2.imwrite('o.jpg',imgc)
cv2.destroyAllWindows()
