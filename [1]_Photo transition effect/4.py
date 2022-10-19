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

# img1 = cv2.resize(img1, (500, 500))
# img2 = cv2.resize(img2, (500, 500))

#Loop D
view = img1.copy()
view1 = img1.copy()
view2 = img2.copy()
H0 = img1.shape[0]
H1= img1.shape[1]

for D1, D2 in zip(range(H0), range(H1)):
    
    view[0:D1,0:D2] = img1[H0-D1:,H1-D2:]
    view[D1:,D2:]= img2[0:H0-D1,0:H1-D2]
    # view[:,D:]= 0
    # view[:,D:]= 0

    print(view[:,:])
    #show
    cv2.imshow('View', view)
    cv2.waitKey(5)
#pixel wised tranform
#double exposure