import cv2
import numpy as np

img = cv2.imread('9-ro.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img', img_gray)

template = cv2.imread('template.png', 0)
w, h = template.shape[1], template.shape[0]
# cv2.imshow('template', template)

print(img_gray.shape)
print(template.shape)


def cross_correlation(a, b):
    a = np.array(a, dtype='double')
    b = np.array(b, dtype='double')
    numerator = np.dot(a, b)
    denominator = np.sqrt(sum(a ** 2) * sum(b ** 2))
    return numerator / denominator


def templat_matching(image, template):
    img_h, img_w = image.shape
    ker_h, ker_w = template.shape
    res_h, res_w = img_h - ker_h + 1, img_w - ker_w + 1
    res = np.zeros((res_h, res_w))
    template_flatten = template.flatten()
    for y in range(res_h):
        for x in range(res_w):
            img_flatten = image[y:y + ker_h, x:x + ker_w].flatten()
            res[y][x] = cross_correlation(template_flatten, img_flatten)

    return res


res = templat_matching(image=img_gray, template=template)
# cv2.imshow('res', res)
#
THRESHOLD = 0.95
loc = np.where(res >= THRESHOLD)

# Draw boudning box
for y, x in zip(loc[0], loc[1]):
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
cv2.imshow('img', img)
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF)
# print(res.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()
