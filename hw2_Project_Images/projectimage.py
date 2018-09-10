import numpy as np
import cv2
from calHomography import *

#decide is p is in PQRS of image 1
def is_inside(quad, p):
	y0 = (- quad.edges[0][0] * p.x - quad.edges[0][2]) / quad.edges[0][1]
	y1 = (- quad.edges[2][0] * p.x - quad.edges[2][2]) / quad.edges[2][1]
	x1 = (- quad.edges[1][1] * p.y - quad.edges[1][2]) / quad.edges[1][0]
	x0 = (- quad.edges[3][1] * p.y - quad.edges[3][2]) / quad.edges[3][0]

	if p.x >= x0 and p.x <= x1 and p.y >= y0 and p.y <= y1 :
		return True
	return False 

#get RGB when p is not a integer pixel point
def getRGB(pt, img):
	x = int(round(pt.x))
	y = int(round(pt.y))
	return img[x, y]

def DoProject(img, imgp, quad, quadp):
	#First Calculate the Homography from quad to quadq
	H = cal_Homography(quadp, quad)

	#to project the image
	x0, x1, y0, y1 = quad.get_box()

	for x in range(x0,x1+1):
		for y in range(y0,y1+1):
			pt = Point(x,y)
			if is_inside(quad, pt):
				ptp = np.matmul(H, pt.hp.reshape((3,1)))
				ptp = ptp / ptp[2][0]
				ptp = Point(ptp[0][0], ptp[1][0])
				img[x,y] = getRGB(ptp, imgp)

