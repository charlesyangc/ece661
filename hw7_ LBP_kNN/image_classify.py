import numpy as np
import cv2
from LBP_kNN import *

k = 5

# training process:
data = []
labels = []
for i in range(20):
	img = cv2.imread('./imagesDatabaseHW7/training/beach/' + str(i+1) +'.jpg')
	hist = LBP(img)
	data.append(hist)
	labels.append('beach')

for i in range(20):
	img = cv2.imread('./imagesDatabaseHW7/training/building/' + str(i+1).zfill(2) +'.jpg')
	hist = LBP(img)
	data.append(hist)
	labels.append('building')

for i in range(20):
	img = cv2.imread('./imagesDatabaseHW7/training/car/' + str(i+1).zfill(2) +'.jpg')
	hist = LBP(img)
	data.append(hist)
	labels.append('car')

for i in range(20):
	img = cv2.imread('./imagesDatabaseHW7/training/mountain/' + str(i+1).zfill(2) +'.jpg')
	hist = LBP(img)
	data.append(hist)
	labels.append('mountain')

for i in range(20):
	img = cv2.imread('./imagesDatabaseHW7/training/tree/' + str(i+1).zfill(2) +'.jpg')
	hist = LBP(img)
	data.append(hist)
	labels.append('tree')

# for i in range(100):
# 	print(data[i], labels[i])

# testing process:
print('for beach:')
for i in range(5):
	img = cv2.imread('./imagesDatabaseHW7/testing/beach_' + str(i+1) +'.jpg')
	hist = LBP(img)
	label = kNN_testing(data, labels, hist, k)
	print('predicted label is ', label)

print('for building:')
for i in range(5):
	img = cv2.imread('./imagesDatabaseHW7/testing/building_' + str(i+1) +'.jpg')
	hist = LBP(img)
	label = kNN_testing(data, labels, hist, k)
	print('predicted label is ', label)

print('for car:')
for i in range(5):
	img = cv2.imread('./imagesDatabaseHW7/testing/car_' + str(i+1) +'.jpg')
	hist = LBP(img)
	label = kNN_testing(data, labels, hist, k)
	print('predicted label is ', label)

print('for mountain:')
for i in range(5):
	img = cv2.imread('./imagesDatabaseHW7/testing/mountain_' + str(i+1) +'.jpg')
	hist = LBP(img)
	label = kNN_testing(data, labels, hist, k)
	print('predicted label is ', label)

print('for tree:')
for i in range(5):
	img = cv2.imread('./imagesDatabaseHW7/testing/tree_' + str(i+1) +'.jpg')
	hist = LBP(img)
	label = kNN_testing(data, labels, hist, k)
	print('predicted label is ', label)


