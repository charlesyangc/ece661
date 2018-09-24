import numpy as np
import cv2
from find_interest_points import *

img0 = cv2.imread('./HW4Pics/pair4/1.jpg')
img1 = cv2.imread('./HW4Pics/pair4/2.jpg')

img0 = cv2.resize(img0, (756, 1008), interpolation = cv2.INTER_CUBIC)
img1 = cv2.resize(img1, (756, 1008), interpolation = cv2.INTER_CUBIC)

kp0, des0 = use_sift(img0)
kp1, des1 = use_sift(img1)

print(len(kp0))
print(len(kp1))

# create BFMatcher object
bf = cv2.BFMatcher()

# Match descriptors.
matches = bf.match(des0,des1)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img2 = cv2.drawMatches(img0,kp0,img1,kp1,matches, None, flags=2)

cv2.imwrite('4sift.jpg', img2)
