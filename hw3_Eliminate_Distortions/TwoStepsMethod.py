import numpy as np
from calHomography import *
from projectimage import *
import cv2
from numpy import linalg as LA

#first eliminate projective distortion:
#First set of two parallel lines for PIC. 1
# P = Point(765, 1254)
# Q = Point(708, 1380)
# R = Point(942, 1242)
# S = Point(891, 1368)

#for PIC. 2
P = Point(72, 246)
Q = Point(83, 326)
R = Point(145, 244)
S = Point(154, 325)

#for PIC 8
# P = Point(1894, 1408)
# Q = Point(1768, 1844)
# R = Point(2366, 1604)
# S = Point(2188, 2056)

# #for PIC 15
# P = Point(249, 363)
# Q = Point(245, 384)
# R = Point(288, 363)
# S = Point(281, 384)

quad = Quad(P, Q, R, S)


Hp = cal_Homography_Projective_distortion(quad)
print('Hp', Hp)
img = cv2.imread('./HW3Pics/2.jpg')
img1 = correct_distorted_image_out2in(img, Hp)
cv2.imwrite('220.jpg',img1)


#next eliminate Affine distortion:

#change examining points in Pic. 1 to Projective distortion removed world.
quad_a = quad
for i in range(4):
	pt = np.matmul(Hp, quad.A[i].hp.reshape((3,1)))
	quad_a.A[i] = Point(pt[0][0]/pt[2][0], pt[1][0]/pt[2][0])

lines0 = PerpendicularLines(quad_a.A[0], quad_a.A[1], quad_a.A[1], quad_a.A[3])
lines1 = PerpendicularLines(quad_a.A[0], quad_a.A[3], quad_a.A[1], quad_a.A[2])

# lines0 = PerpendicularLines(quad.A[0], quad.A[1], quad.A[0], quad.A[2])
# lines1 = PerpendicularLines(quad.A[1], quad.A[2], quad.A[0], quad.A[3])

linesset = [lines0, lines1]

Ha = cal_Homography_Affine_distortion(linesset) #this Ha is from distorted to correct 

print('Ha', Ha)
H = np.matmul(Ha, Hp)
#H = H / np.amax(H)
print('H', H)
imgf = correct_distorted_image_out2in(img, H)

cv2.imwrite('221.jpg',imgf)
cv2.destroyAllWindows()


