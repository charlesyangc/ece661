import numpy as np
from calHomography import *
from projectimage import *
import cv2
from numpy import linalg as LA

#first eliminate projective distortion:
#First set of two parallel lines
P = Point(1248, 246)
Q = Point(408, 2028)
R = Point(1827, 132)
S = Point(1611, 2094)
quad = Quad(P, Q, R, S)

#for figure 2
# P = Point(55, 236)
# Q = Point(73, 334)
# R = Point(282, 236)
# S = Point(275, 333)
# quad = Quad(P, Q, R, S)

Hp = cal_Homography_Projective_distortion(quad)
print(Hp)
img = cv2.imread('./HW3Pics/1.jpg')
img1 = correct_distorted_image_out2in(img, Hp)
cv2.imwrite('120.jpg',img1)




#next eliminate Affine distortion:


#for figure 2
pt01 = Point(40, 156)
pt02 = Point(184, 157)
pt03 = Point(55, 244)

#for figure 2
pt04 = Point(185, 157)
pt05 = Point(202, 245)
pt06 = Point(44, 156)


# for figure 1
A = Point(756, 582)
B = Point(1071, 492)
C = Point(198, 963)
D = Point(1263, 252)
lines1 = PerpendicularLines(A, B, C, D)
A2 = Point(1149, 324)
B2 = Point(1464, 237)
C2 = Point(513, 870)
D2 = Point(1575, 165)
lines2 = PerpendicularLines(A2, B2, C2, D2)

# A = Point(1365, 228)
# B = Point(621, 2046)
# C = Point(591, 1155)
# D = Point(1737, 1056)
# lines1 = PerpendicularLines(A, B, C, D)
# A2 = Point(1827, 135)
# B2 = Point(1611, 2094)
# C2 = Point(1131, 498)
# D2 = Point(1800, 384)
# lines2 = PerpendicularLines(A2, B2, C2, D2)

linesset = [lines1, lines2]

Ha = cal_Homography_Affine_distortion(linesset) #this Ha is from distorted to correct 

img = cv2.imread('./HW3Pics/1.jpg')
print(Ha)
H = np.matmul(Ha, Hp)
imgf = correct_distorted_image_out2in(img1, H)

cv2.imwrite('121.jpg',imgf)
cv2.destroyAllWindows()


