from PIL import Image
import cv2

img1 = Image.open(r'1.jpg')
img2 = Image.open(r'2.jpg')

# img1 = Image.resize(img1, (500, 500))
# img2 = Image.resize(img2, (500, 500))

img = Image.blend(img1, img2, alpha = 0.9)

img.show()
