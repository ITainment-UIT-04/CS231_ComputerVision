import cv2
import numpy as np

#Buoc 1: Doc anh tu file
img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')


scale_percent = 20 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img1 = cv2.resize(img1, dim)
img2 = cv2.resize(img2, dim)


# Ảnh đi từ trên xuống
#Buoc 2: Vong lap voi D
view = img1.copy()
view1 = img1.copy()
view2 = img2.copy()
H = img1.shape[0]
for D in range(H):
    #Buoc 2.1 Cat phan dau cua view hien thi
    view[0:D,:] = img1[H-D:,:]
    #Buoc 2.2. Cat phan cuoi cua view hien thi
    view[D:,:]= img2[0:H-D,:]

    print(view[:,:])
    #Buoc 2.3. Hien thi anh
    cv2.imshow('View', view)
    cv2.waitKey(10)
#pixel wised tranform
#double exposure