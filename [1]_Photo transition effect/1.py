# importing cv2 
import cv2
  
# path
path = r'rog.jpg'
  
# Using cv2.imread() method
img = cv2.imread(path)


scale_percent = 40 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim)

# Displaying the image
cv2.imshow('image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()



def change_brightness(img, value):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	v = cv2.add(v,value)
	v[v > 255] = 255
	v[v < 0] = 0
	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img

upbright = resized.copy()
upbright = change_brightness(upbright, value=20) 
cv2.imshow('Brightness', upbright)
cv2.waitKey(0)
cv2.destroyAllWindows()



originalImage = resized.copy()
flipVertical = cv2.flip(originalImage, 0)
flipHorizontal = cv2.flip(originalImage, 1)
#flipBoth = cv2.flip(originalImage, -1)
 
#cv2.imshow('Original image', originalImage)
cv2.imshow('Flipped vertical image', flipVertical)
cv2.imshow('Flipped horizontal image', flipHorizontal)
#cv2.imshow('Flipped both image', flipBoth)
 
cv2.waitKey(0)
cv2.destroyAllWindows()


image = resized.copy()
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
negative = 255- image
#cv2.imshow('Original image',image)
#cv2.imshow('Gray image', gray)
cv2.imshow('Negative image', negative)
cv2.waitKey(0)
cv2.destroyAllWindows()