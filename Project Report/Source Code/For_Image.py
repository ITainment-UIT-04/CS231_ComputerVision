import numpy as np
import cv2

kernel = np.ones((3,3),np.uint8)
def process(mask, kernel):
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    return mask

img = cv2.imread('1.png')
# image = cv2.imread(r"1.png", cv2.IMREAD_GRAYSCALE)
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

new_background = cv2.imread("background.jpg")

frame_width = img.shape[1]
frame_height = img.shape[0]
# print(frame_width, frame_height)
bkg = cv2.resize(new_background, (frame_width, frame_height))

cv2.imshow("image", img)
cv2.imshow("gray", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# low = np.array([0, 0, 0])
# high = np.array([200, 51, 51])

if (cv2.inRange(image, 0, 128) == 0).sum() > (cv2.inRange(image, 128, 255) == 0).sum():
    print(1)
    mask = cv2.inRange(image, 0, 128)
else:
    mask = cv2.inRange(image, 128, 255)


mask = process(mask, kernel)

cv2.imshow("mask", mask)
# print(mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

result = cv2.bitwise_and(img, img, mask=mask)
# result = cv2.bitwise_and(img, result, mask= None)
cv2.imshow("result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

rest = cv2.bitwise_or(bkg, bkg, mask=mask)
rest = cv2.bitwise_xor(rest, bkg)
cv2.imshow("Bitwise OR: Mask <OR> Background", rest)

# Change background after remove 
changebkg = cv2.bitwise_or(result, rest)
cv2.imshow("Change Background", changebkg)
cv2.imshow("New Background", bkg)
cv2.waitKey(0)
cv2.destroyAllWindows()