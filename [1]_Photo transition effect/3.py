import cv2
import numpy as np

#Import image
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('123.png')


scale_percent = 30 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img1 = cv2.resize(img1, dim)
img2 = cv2.resize(img2, dim)


#Loop D
view = img1.copy()
view1 = img1.copy()
view2 = img2.copy()
H = img1.shape[1]
for D in range(H):
    #Cut right of view
    view[:,0:D] = img1[:,H-D:]
    #Cut left of view
    view[:,D:]= img2[:,0:H-D]

    print(view[:,:])
    #show
    cv2.imshow('View', view)
    cv2.waitKey(1)
#pixel wised tranform
#double exposure