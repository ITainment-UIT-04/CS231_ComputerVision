import cv2
import numpy as np
import cv2

#Import image
img1 = cv2.imread('./1.jpg')
img2 = cv2.imread('./123.png')


scale_percent = 30 # percent of original size
width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img1 = cv2.resize(img1, dim)
img2 = cv2.resize(img2, dim)

# cover 
view = img1.copy()
W = img1.shape[1]

for D in range(W - 1, 0, -1):
    view[:, D:W] = img1[:, 0:W - D]
    view[:, 0:D] = img2[:, 0:D]

    cv2.imshow('View', view)
    cv2.waitKey(1)
cv2.destroyAllWindows()


# uncover
view = img1.copy()
W = img1.shape[1]
for D in range(W - 1, 0, -1):
    view[:,0:D] = img1[:, W - D:]
    view[:, D:] = img2[:, D:]

    cv2.imshow('View', view)
    cv2.waitKey(1)
cv2.destroyAllWindows()

# push up
view = img1.copy()
H = img1.shape[0]
for D in range(H,0,-1):
    view[0:D,:] = img1[H-D:,:]
    view[D:,:]= img2[0:H-D,:]
    cv2.imshow('View', view)
    cv2.waitKey(1)
cv2.destroyAllWindows()

# push down
view = img1.copy()
H = img1.shape[0]
for D in range(H):
    view[0:D,:] = img1[H-D:,:]
    view[D:,:]= img2[0:H-D,:]
    cv2.imshow('View', view)
    cv2.waitKey(1)
cv2.destroyAllWindows()

# push left
view = img1.copy()
W = img1.shape[1]
for D in range(W):
    view[:,0:D] = img1[:,W-D:]
    view[:,D:]= img2[:,0:W-D]
    cv2.imshow('View', view)
    cv2.waitKey(1)
cv2.destroyAllWindows()


# Go left and down
view = img2.copy()
H, W = img1.shape[:2]
for p in range(101):
    y, x = int(p * H / 100), int(p * W / 100)
    view[0:y, W - x:] = img1[H - y:, 0:x]
    # view[y:H, 0: W - x] = img2[0:H - y, x:W]
    cv2.imshow('View', view)
    key = cv2.waitKey(10)
cv2.destroyAllWindows()
