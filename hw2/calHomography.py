import numpy as np
from numpy.linalg import inv

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

