import numpy as np
from find_interest_points import *



img0 = cv2.imread('./HW4Pics/pair1/1.jpg')
img1 = cv2.imread('./HW4Pics/pair1/2.jpg')
gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

sigma = math.sqrt(2)
corner_threshold = 4e-2
list_Corner0 = harris_corner_detector(img0, sigma, corner_threshold)
list_Corner1 = harris_corner_detector(img1, sigma, corner_threshold)

print('in harris_corner_detector')
print(list_Corner0)
print(len(list_Corner0))
print(len(list_Corner1))

ncc_threshold= 0.95
list_ssd, list_ncc = find_correspondent_points(gray0, gray1, list_Corner0, list_Corner1, ncc_threshold)	

img_ssd = output_two_imgs_with_corr_points(img0, img1, list_ssd)
cv2.imwrite('1ssd_sigma_'+str(sigma), img_ssd)

img_ncc = output_two_imgs_with_corr_points(img0, img1, list_ncc)
cv2.imwrite('1ncc_sigma_'+str(sigma), img_ncc)

#cv2.imshow('gray', gray)
cv2.destroyAllWindows()