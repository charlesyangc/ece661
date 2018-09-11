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
	#imgc = correct_distorted_image_4ptsDetBd(img, Hc)
	imgc = correct_distorted_image_out2in(img, Hc)
	return imgc



img = cv2.imread('./HW3Pics/2.jpg')
imgc = P2PCorr(quad2, quadc, img)
cv2.imwrite('o.jpg',imgc)
cv2.destroyAllWindows()
