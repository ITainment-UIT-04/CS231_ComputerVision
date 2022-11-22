import cv2
import numpy as np
import matplotlib.pyplot as plt

im_gray = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("gray", im_gray)
# cv2.waitKey(0)

hist = cv2.calcHist([im_gray],[0],None,[256],[0,256])
# plt.plot(hist)
# plt.show()

def imhist(im):
  # calculates normalized histogram of an image
	m, n = im.shape
	h = [0.0] * 256
	for i in range(m):
		for j in range(n):
			h[im[i, j]]+=1
	return np.array(h)/(m*n)

def cumsum(h):
	# finds cumulative sum of a numpy array, list
	return [sum(h[:i+1]) for i in range(len(h))]

def histeq(im):
	#calculate Histogram
    h = imhist(im)
    cdf = np.array(cumsum(h)) #cumulative distribution function
    sk = np.uint8(255 * cdf) #finding transfer function values
    s1, s2 = im.shape
    Y = np.zeros_like(im)
	# applying transfered values for each pixels
	
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = sk[im[i, j]]
    H = imhist(Y)
	#return transformed image, original and new istogram, 
	# and transform function
	
    return Y , h, H, sk


new_img, _,_,_ = histeq(im_gray)

cv2.imshow('Gray scale image',im_gray)
# cv2.waitKey(0)

fig = plt.figure("Plotting Histogram")
plt.title('Plotting histogram')
plt.plot(hist)
plt.show()
cv2.destroyAllWindows()
cv2.imshow('New Image after mapping color',new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
