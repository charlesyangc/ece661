import numpy as np
import cv2
from find_interest_points import *

img0 = cv2.imread('./HW4Pics/pair1/1.jpg')
kp0, des0 = use_surf(img0)

img1 = cv2.imread('./HW4Pics/pair1/2.jpg')
kp1, des1 = use_surf(img1)

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

cv2.imwrite('1surf.jpg', img2)