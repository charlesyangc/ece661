import numpy as np
import math
from numpy.linalg import inv
from numpy import linalg as LA

class Point():
	"""docstring for point"""
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.hp = np.array([x,y,1])

class Quad():
	"""docstring for rectangle"""
	def __init__(self, P, Q, R, S):
		self.A = [P, Q, R, S]
		self.edges = [np.cross(P.hp, R.hp), np.cross(R.hp, S.hp), np.cross(S.hp, Q.hp), np.cross(Q.hp, P.hp)]

	def get_box(self):
		x0 = self.A[0].x
		x1 = self.A[0].x
		y0 = self.A[0].y
		y1 = self.A[0].y
		for i in range(4):
			if self.A[i].x < x0:
				x0 = self.A[i].x
			if self.A[i].x > x1:
				x1 = self.A[i].x
			if self.A[i].y < y0:
				y0 = self.A[i].y 
			if self.A[i].y > y1:
				y1 = self.A[i].y
		return x0, x1, y0, y1

class box_2d():
	"""docstring for box"""
	def __init__(self, x0, x1, y0, y1):
		self.bd = [x0, x1, y0, y1]

	def get_int_coo_box(self):
		for i in range(4):
			if self.bd[i] < 0:
				self.bd[i] = math.floor(self.bd[i])
			else:
				self.bd[i] = math.ceil(self.bd[i])

		

def cal_Homo_2l(p0, p1):
	A = np.zeros((2,8))

	A[0][0] = p1.x
	A[0][1] = p1.y
	A[0][2] = 1
	A[0][6] = - p1.x * p0.x 
	A[0][7] = - p1.y * p0.x

	A[1][3] = p1.x
	A[1][4] = p1.y
	A[1][5] = 1
	A[1][6] = - p1.x * p0.y
	A[1][7] = - p1.y * p0.y

	b = np.zeros((2,1))
	b[0] = p0.x
	b[1] = p0.y

	return A, b
	

def cal_Homography(quad0, quad1):
	#this function computes the projection H from (Pc, Qc, Rc, Sc) to (P, Q, R, S)
	#create left hand side
	A, b = cal_Homo_2l(quad0.A[0], quad1.A[0])
	for i in range(3):
		C, d = cal_Homo_2l(quad0.A[i+1], quad1.A[i+1])
		A = np.concatenate((A,C), axis = 0)
		b = np.concatenate((b,d), axis = 0)

	Ainv = inv(A)
	h = np.matmul(Ainv, b)
	h = np.append(h,[[1]])
	h = np.reshape(h,(3,3))
	return h

def cal_Homography_Projective_distortion(quad):
	l0 = np.cross(quad.A[0].hp, quad.A[1].hp)
	l1 = np.cross(quad.A[2].hp, quad.A[3].hp)
	VP0 = np.cross(l0, l1)

	l2 = np.cross(quad.A[0].hp, quad.A[2].hp)
	l3 = np.cross(quad.A[1].hp, quad.A[3].hp)
	VP1 = np.cross(l2, l3)

	VL = np.cross(VP0, VP1)

	VL = VL / np.linalg.norm(VL)

	H = np.zeros((3,3), dtype = 'float')
	H[0] = [1, 0, 0]
	H[1] = [0, 1, 0]
	H[2] = VL

	return H

def cal_Homography_Affine_distortion(setofptsets):

	M = np.zeros((2,2), dtype = 'float')
	b = np.zeros((2,1), dtype = 'float')
	for i in range(2):
		l = np.cross(setofptsets[i][0].hp, setofptsets[i][1].hp)
		m = np.cross(setofptsets[i][0].hp, setofptsets[i][2].hp)
		M[i][0] = l[0] * m[0]
		M[i][1] = l[0] * m[1] + l[1] * m[0]
		b[i][0] = -l[2] * m[2]
		print(M)

	s = np.matmul(inv(M), b)
	s = s / LA.norm(s)
	S = np.array([ [s[0][0], s[1][0]], [s[1][0], 1.0] ], dtype = 'float')	
	U, d, V = LA.svd(S)
	d = np.sqrt(d)
	D = np.diag(d)
	A = np.matmul(U, np.matmul(D, V))
	A = A / LA.norm(A)

	Ha = np.array([[A[0][0], A[0][1], 0], [A[1][0], A[1][1], 0], [0, 0, 1]], dtype = 'float')

	Ha = inv(Ha)

	return(Ha)

