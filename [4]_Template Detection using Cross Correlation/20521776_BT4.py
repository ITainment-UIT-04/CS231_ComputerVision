import cv2
import numpy as np

def normalize_image(input):
        input[:,:,0] = 255 * ((input[:,:,0] - input[:,:,0].min()) / (input[:,:,0].max() - input[:,:,0].min()))
        input[:,:,1] = 255 * ((input[:,:,1] - input[:,:,1].min()) / (input[:,:,1].max() - input[:,:,1].min()))
        input[:,:,2] = 255 * ((input[:,:,2] - input[:,:,2].min()) / (input[:,:,2].max() - input[:,:,2].min()))
        return input


class detection:
    def __init__(self):
        pass

    

    def resize(self, img, scale_percent = 50):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
          
        # resize image
        img = cv2.resize(img, dim)
        return img


    def cross_correlation(self, x, y):
        X, Y = x, y
        x_flten, y_flten = X.flatten(), Y.flatten()
        m, n = np.sum(x_flten**2), np.sum(y_flten**2)
        return np.dot(x_flten, y_flten) / np.sqrt(m * n)


    def template_matching(self, image, x, y, thresh_hold=0.5):
        a, b = x, y
        h_a, w_a = a.shape
        h_b, w_b = b.shape
        height = (h_a - h_b) + 1
        width = (w_a - w_b) + 1
        result = np.ones((height, width), dtype=np.float64)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i][j] = self.cross_correlation(a[i:i+h_b, j:j+w_b], b)
        (y_points, x_points) = np.where(result >= thresh_hold)
        boxes = []
        for (x, y) in zip(x_points, y_points):
            boxes.append((x, y, x + w_b, y + h_b))
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    def displayImage(self, image):
        cv2.imshow("Template matching", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pass



threshold = 0.9
detec = detection()

image = cv2.imread('9-ro.jpeg')
template = cv2.imread('template.png')


scale_percent = 200
width = int(template.shape[1] * scale_percent / 100)
height = int(template.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image


x = normalize_image(image)
y = normalize_image(template)
x = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
y = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
y1 = cv2.resize(y, dim)

x = x.astype('float64')
y = y.astype('float64')
y1 = y1.astype('float64')

detec.template_matching(image, x, y, threshold)

template_big = cv2.resize(template, dim)
detec.template_matching(image, x, y, threshold)
detec.template_matching(image, x, y1, threshold)
detec.displayImage(image)



