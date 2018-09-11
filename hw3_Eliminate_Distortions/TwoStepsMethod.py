import numpy as np
from calHomography import *
from projectimage import *
import cv2
from numpy import linalg as LA

#first eliminate projective distortion:
#First set of two parallel lines
P = Point(870, 1041)
Q = Point(828, 1140)
R = Point(1032, 1014)
S = Point(990, 1125)
quad = Quad(P, Q, R, S)

#for figure 2
P = Point(55, 236)
Q = Point(73, 334)
R = Point(282, 236)
S = Point(275, 333)
quad = Quad(P, Q, R, S)

Hp = cal_Homography_Projective_distortion(quad)
img = cv2.imread('./HW3Pics/2.jpg')
img1 = correct_distorted_image_out2in(img, Hp)
cv2.imwrite('220.jpg',img1)




#next eliminate Affine distortion:
# first set of perpendiculer lines:
pt01 = Point(525, 271)
pt02 = Point(489, 437)
pt03 = Point(654, 147)

#for figure 2
pt01 = Point(40, 156)
pt02 = Point(184, 157)
pt03 = Point(55, 244)


ptsets0 = [pt01, pt02, pt03]

# second set of perpendicular lines:
pt04 = Point(347, 538)
pt05 = Point(259, 766)
pt06 = Point(544, 348)

#for figure 2
pt04 = Point(201, 244)
pt05 = Point(55, 243)
pt06 = Point(185, 156)


ptsets1 = [pt04, pt05, pt06]

setofptsets = [ptsets0, ptsets1]

Ha = cal_Homography_Affine_distortion(setofptsets) #this Ha is from distorted to correct 

img = cv2.imread('./HW3Pics/2.jpg')
H = np.matmul(Ha, Hp)
imgf = correct_distorted_image_out2in(img, H)

cv2.imwrite('221.jpg',imgf)
cv2.destroyAllWindows()


