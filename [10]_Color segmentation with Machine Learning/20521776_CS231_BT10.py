import cv2
import numpy as np  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

## Make default scale 
default_scale = (1280, 720)

def _resize(img, scale_percent = 100, scale = None):
    if scale is None:
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img = cv2.resize(img, dim)
    else:
        img = cv2.resize(img, scale)
    return img

def _toRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


## Input
background = _resize(cv2.imread("background.png"), scale = default_scale)
foreground = _resize(cv2.imread("1.png"), scale = default_scale)

## Convert: RGB to RGB && Reshape to (-1, 3)
backgrd = _toRGB(background).reshape(-1, 3)
foregrd = _toRGB(foreground).reshape(-1, 3)
 
## Plot Image
cv2.imshow("Background", background)
cv2.imshow("Foreground", foreground)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Create X_train
X_train = np.append(backgrd, foregrd, axis = 0)

## Create y_train
true_label = np.ones(backgrd.shape[0])
false_label = np.zeros(foregrd.shape[0])
y_train = np.append(true_label, false_label, axis = 0)
# y_train = y_train.reshape(-1, 1)
# print(np.append(backgrd, foregrd, axis = 0).shape)

## LogisticRegression model
classification = LogisticRegression()
classification.fit(X_train, y_train)


## Predict from X_Test 
print("\n\tAuto Remove Background is processing...")
X_test = cv2.imread("image.jpg")
X_test = _toRGB(X_test)
y_pred = X_test.copy()

H, W = X_test.shape[0], X_test.shape[1]
for i in range(H):
    for j in range(W):
        if classification.predict(X_test[i][j].reshape(1,-1)) == 1:
            y_pred[i][j] = (0,0,0)

print("\n\tSuccessfully!")
cv2.imshow("Remove background", _toRGB(y_pred))
cv2.waitKey(0)
cv2.destroyAllWindows()



## Compare value
print("\nUse the Result of KMeans Clustering in CV as y_test\n\tProcessing...")
img = X_test.copy()
r, g, b = cv2.split(img)
len_r, len_g, len_b = len(np.unique(r)), len(np.unique(g)), len(np.unique(b))
K = int((len_r + len_g + len_b)*0.1/3)
max_iter = int((len(r) - len_r + len(g) - len_g + len(b) - len_b)/3)
vectorized = np.float32(img.reshape((-1,3)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
attempts = 10
ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))
convert = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(convert, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
thresh = 156
convert = cv2.threshold(convert, thresh, 255, cv2.THRESH_BINARY)[1]
oriImage = X_test.copy()
H, W, D = oriImage.shape[0], oriImage.shape[1], oriImage.shape[2] 
for i in range(H):
    for j in range(W):
        for k in range(D):
            if convert[i][j] == 255:
                oriImage[i][j][k] = 0

y_test = cv2.cvtColor(oriImage, cv2.COLOR_BGR2RGB)
# cv2.imshow("Remove Background", oriImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(classification_report(y_test.reshape(-1, 1), y_pred.reshape(-1, 1), labels=[0, 1]))