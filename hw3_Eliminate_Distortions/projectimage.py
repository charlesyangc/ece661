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


def find_projected_boundary(quad, Hc):
	quadp = quad
	for i in range(4):
		pc = np.matmul(Hc, quad.A[i].hp.reshape((3,1))) 
		pc = pc / pc[2][0]
		quadp.A[i] = Point(pc[0][0], pc[1][0])
	x0, x1, y0, y1 = quadp.get_box()
	box = box_2d(x0, x1, y0, y1)
	box.get_int_coo_box()
	print('box:', x0, x1, y0, y1, box.bd)
	return box

def correct_distorted_image(img, H):
	size = img.shape
	# find the boundary of corrected picture
	Pb = Point(0, 0)
	Qb = Point(0, size[1])
	Rb = Point(size[0], 0)
	Sb = Point(size[0], size[1])

	#Pb = Point(0, size[1]/2)
	#Qb = Point(size[0]/2, size[1])
	#Rb = Point(size[0], size[1]/2)
	#Sb = Point(size[0]/2, 0)

	quadb = Quad(Pb, Qb, Rb, Sb)
	box = find_projected_boundary(quadb, H)

	print(box.bd)
	print(size[0], size[1])

	#correct distortion:
	sizec = [box.bd[1]-box.bd[0]+1, box.bd[3]-box.bd[2]+1, 3]
	imgc = np.zeros(sizec, np.uint8)

	print(sizec)

	init = np.matmul(H, [[size[0]/2],[size[1]/2],[1.0]])
	x0 = init[0][0]
	x1 = init[0][0]
	y0 = init[1][0]
	y1 = init[1][0]
	#first find box:
	for x in range(0, size[0]):
		for y in range(0, size[1]):
			pt = Point(x,y)
			ptc = np.matmul(H, pt.hp.reshape((3,1)))
			ptc = ptc / ptc[2][0] 
			ptcx = int(round(ptc[0][0]))
			ptcy = int(round(ptc[1][0]))
			if ptcx < x0:
				x0 = ptcx-1
			if ptcx > x1:
				x1 = ptcx+1
			if ptcy < y0:
				y0 = ptcy-1
			if ptcy > y1:
				y1 = ptcy+1
			#print(x, y, ptcx, ptcy)

	print('boundary:', x0, x1, y0, y1)
	ratio = 1e-5

	bx0 = int(round( x0 * ratio ))
	bx1 = int(round( x1 * ratio ))
	by0 = int(round( y0 * ratio ))
	by1 = int(round( y1 * ratio ))

	imgc = np.zeros([bx1-bx0,by1-by0,3], np.uint8)

	for x in range(0, size[0]):
		for y in range(33, size[1]):
			pt = Point(x,y)
			ptc = np.matmul(H, pt.hp.reshape((3,1)))
			ptc = ptc / ptc[2][0] 
			ptcx = int(round(ptc[0][0]))
			ptcy = int(round(ptc[1][0]))

			if (ptcx < bx1) and (ptcx > bx0) and (ptcy > by0) and (ptcy < by1):
				imgc[ptcx-bx0, ptcy-by0] = getRGB(pt, img)

			#if (ptcx < box.bd[1]) and (ptcx > box.bd[0]) and (ptcy > box.bd[2]) and (ptcy < box.bd[3]):
			#	imgc[ptcx-box.bd[0], ptcy-box.bd[2]] = getRGB(pt, img)
	return imgc

