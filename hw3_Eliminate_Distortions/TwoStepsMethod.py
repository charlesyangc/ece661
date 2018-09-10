import numpy as np
from calHomography import *
from projectimage import *
import cv2

#first eliminate projective distortion:
#First set of two parallel lines
P = Point(870, 1041)
Q = Point(828, 1140)
R = Point(1032, 1014)
S = Point(990, 1125)
quad = Quad(P, Q, R, S)

Hp = cal_Homography_Projective_distortion(quad)

img = cv2.imread('./HW3Pics/1.jpg')
img1 = correct_distorted_image(img, Hp)
cv2.imwrite('120.jpg',img1)


#next eliminate Affine distortion:
# first set of perpendiculer lines:
pt01 = Point(525, 271)
pt02 = Point(489, 437)
pt03 = Point(654, 147)

ptsets0 = [pt01, pt02, pt03]

# second set of perpendicular lines:
pt04 = Point(347, 538)
pt05 = Point(259, 766)
pt06 = Point(544, 348)

ptsets1 = [pt04, pt05, pt06]

Ha = cal_Homography_Affine_distortion(ptsets0, ptsets1) #this Ha is from distorted to correct 

img = cv2.imread('./HW3Pics/1.jpg')
H = np.matmul(Ha, Hp)
imgf = correct_distorted_image(img, H)

cv2.imwrite('121.jpg',imgf)
cv2.destroyAllWindows()


