import cv2
## Crop image: https://stackoverflow.com/questions/70007837/how-to-select-circle-automatically-on-image-with-mouse-and-crop-it-using-python
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0

image = cv2.imread('image.jpg')
oriImage = image.copy()

def trans_outValue(n):
    # n = int(0.9*n)
    return n if n > 0 else 0

def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            # cv2.imshow("Cropped", roi)
            r, g, b = cv2.split(roi)
            min_r, max_r = r.min() - 100, r.max() + 20
            min_g, max_g = g.min() - 100, g.max() + 20
            min_b, max_b = b.min() - 100, b.max() + 20

            print(f"################# Cropped Area #################")
            print(f"\nRED: Min: {trans_outValue(min_r)}, Max: {max_r} \
                    \nGREEN: Min: {trans_outValue(min_g)}, Max: {max_g}\
                    \nBLUE: Min: {trans_outValue(min_b)}, Max: {max_b}")
            print("\n\tEnter 'Q' to exit\n")
            print("Please wait after cropping!")
            H, W = oriImage.shape[0], oriImage.shape[1]
            for i in range(H):
                for j in range(W):                    
                    if min_r < oriImage[i][j][0] < max_r \
                    and min_g < oriImage[i][j][1] < max_g \
                    and min_b < oriImage[i][j][2] < max_b:
                        oriImage[i][j][0] = 0
                        oriImage[i][j][1] = 0
                        oriImage[i][j][2] = 0
            cv2.imshow("Removed Background", oriImage)
            

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
while True:
    temp = image.copy()
    if not cropping:
        cv2.imshow("image", image)

    elif cropping:
        cv2.rectangle(temp, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", temp)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break

