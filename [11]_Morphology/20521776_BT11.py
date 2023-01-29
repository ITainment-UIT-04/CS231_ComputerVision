# import the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r"image.png", cv2.IMREAD_GRAYSCALE)
 
# binarize the image
binr = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)[1]
 
# define the kernel
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(binr, kernel,iterations = 7)
closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=5)
# print(closing)
plt.imshow(closing, cmap='gray')
plt.show()

n = cv2.connectedComponents(closing)
print("Number of blood cells: ", n[0])
