import cv2
import numpy as np

############################### Convolution ###############################

def conv_transform(image):
    image_copy = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]): 
            image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]
    return image_copy

def conv(image, kernel):
    kernel = conv_transform(kernel)
    image_h = image.shape[0]
    image_w = image.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    h = kernel_h//2
    w = kernel_w//2

    # image_conv = np.zeros(image.shape)
    image_conv = image.copy()

    for i in range(h, image_h-h):
        for j in range(w, image_w-w):
            sum = 0

            for m in range(kernel_h):
                for n in range(kernel_w):
                    sum = sum + kernel[m][n]*image[i-h+m][j-w+n]
                    # print(sum)
            
            image_conv[i][j] = sum
    # print(image_conv)
    return image_conv

img = cv2.imread("image.jpg")

scale_percent = 20 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim)

# cv2.imshow("Original image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

## Identity
kernel_1 = np.array([[0,0,0], 
                    [0,1,0], 
                    [0,0,0]])

## Gaussian blur 3 × 3
kernel_2 = 1/16 * np.array([[1,2,1], 
                            [2,4,2], 
                            [1,2,1]])

## Box blur
kernel_3 = 1/9 * np.array([[1,1,1], 
                            [1,1,1], 
                            [1,1,1]])

## Sharpen
kernel_4 = np.array([[0,-1,0], 
                    [-1,5,-1], 
                    [0,-1,0]])


print("Processing ...")

# conv_image_1 = conv(img, kernel_1)
# cv2.imshow("Identity", conv_image_1)

# conv_image_2 = conv(img, kernel_2)
# cv2.imshow("Gaussian blur", conv_image_2)

# conv_image_3 = conv(img, kernel_3)
# cv2.imshow("Box blur", conv_image_3)

conv_image_4 = conv(img, kernel_4)
cv2.imshow("Sharpen", conv_image_4)

cv2.imshow("Original image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()


############################### Cross-Correlation ###############################

def roi_image(image):
    image = cv2.imread(image, 0)
    roi = image[700:900, 1900:2100]
    return roi

def FouTransf(image):
    img_f32 = np.float32(image)
    d_ft = cv2.dft(img_f32, flags = cv2.DFT_COMPLEX_OUTPUT)
    d_ft_shift = np.fft.fftshift(d_ft)

    rows, cols = image.shape
    opt_rows = cv2.getOptimalDFTSize(rows)
    opt_cols = cv2.getOptimalDFTSize(cols)
    opt_img = np.zeros((opt_rows, opt_cols))
    opt_img[:rows, :cols] = image 
    crow, ccol = opt_rows / 2 , opt_cols / 2
    mask = np.zeros((opt_rows, opt_cols, 2), np.uint8)
    mask[int(crow-50):int(crow+50), int(ccol-50):int(ccol+50)] = 1

    f_mask = d_ft_shift*mask
    return f_mask


def inv_FouTransf(image):

    f_ishift = np.fft.ifftshift(image)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    return img_back

def rms(sigma):
    rms = np.std(sigma)
    return rms

# Step 1: Import images
a = roi_image('1.jpg')
b = roi_image('2.jpg')

# Step 2: Convert the image to frequency domain
G_t0 = FouTransf(a)
G_t0_conj = G_t0.conj()
G_t1 = FouTransf(b)

# Step 3: Compute C(m, v)
C = G_t0_conj * G_t1

# Step 4: Convert the image to space domain to obtain Cov (p, q)
c_w = inv_FouTransf(C)

# Step 5: Compute Cross correlation
R_pq = c_w / (rms(a) * rms(b)) 
print("\nCross correlation: \n", R_pq)