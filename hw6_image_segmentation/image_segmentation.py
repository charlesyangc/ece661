import numpy as np
import cv2
from Otsu_Algorithm import *

#read image:
# img = cv2.imread('./HW6pics/baby.jpg')
# RGB_Flag_Neg = [0, 0, 0]

# img = cv2.imread('./HW6pics/lighthouse.jpg')
# RGB_Flag_Neg = [1, 1, 0] # 1 means take negative before AND for this channel

img = cv2.imread('./HW6pics/ski.jpg')
RGB_Flag_Neg = [1, 1, 0]

# RGB based segmentation:
mask = RGB_segment(img, RGB_Flag_Neg)
contour = contour_extraction(mask)
cv2.imwrite('contour_RGB.jpg', 255*contour)

mask = texture_segment(img)
contour = contour_extraction(mask)
cv2.imwrite('contour_texture.jpg', 255*contour)