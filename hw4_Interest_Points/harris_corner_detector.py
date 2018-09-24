import numpy as np
import cv2
from find_interest_points import *

sigma = math.sqrt(2)
corner_threshold = 0.04

img0 = cv2.imread('./HW4Pics/pair4/1.jpg')
img1 = cv2.imread('./HW4Pics/pair4/2.jpg')
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img0 = cv2.resize(img0, (600, 800), interpolation = cv2.INTER_CUBIC)
img1 = cv2.resize(img1, (600, 800), interpolation = cv2.INTER_CUBIC)


print('shape of img0: ', np.shape(img0))
print('shape of img1: ', np.shape(img1))

list_Corner0 = harris_corner_detector(img0, sigma, corner_threshold)
list_Corner1 = harris_corner_detector(img1, sigma, corner_threshold)

print('length of list_Corner0: ', len(list_Corner0))
print('length of list_Corner1: ', len(list_Corner1))

ncc_threshold= 0.4
list_ssd, list_ncc = find_correspondent_points(gray0, gray1, list_Corner0, list_Corner1, ncc_threshold)	

print('shape of list_ssd: ', np.shape(list_ssd))
print('shape of list_ncc: ', np.shape(list_ncc))

img_ssd = output_two_imgs_with_corr_points(img0, img1, list_ssd)
cv2.imwrite('4ssd_sigma_'+str(sigma)+'.jpg', img_ssd)

img_ncc = output_two_imgs_with_corr_points(img0, img1, list_ncc)
cv2.imwrite('4ncc_sigma_'+str(sigma)+'.jpg', img_ncc)

#cv2.imshow('gray', gray)
cv2.destroyAllWindows()