import numpy as np
import cv2
import time

input = "TEST.mp4"
cap = cv2.VideoCapture(input)
new_background = cv2.imread("background.jpg")

scale = 100 if input == 0 else 50
frame_width = int(cap.get(3)*scale/100) 
frame_height = int(cap.get(4)*scale/100)
# frame_width = 720
# frame_height = 480

# low = 0 if input == 0 else 80
# high = 180 if input == 0 else 255
kernel = np.ones((3,3),np.uint8)

bkg = cv2.resize(new_background, (frame_width, frame_height))

# output = cv2.VideoWriter('grey.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height))


def SetRange(image):
    if (cv2.inRange(image, 0, 128) == 0).sum() > (cv2.inRange(image, 128, 255) == 0).sum():
        mask = cv2.inRange(image, 0, 128 + 70)
    else:
        mask = cv2.inRange(image, 128-30, 255)
    return mask


def process(mask, kernel):
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    return mask


# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

while True:
    # Get Frame
    ret, frame = cap.read()
    

    # new_frame_time = time.time()
    # fps = 1/(new_frame_time-prev_frame_time)
    # prev_frame_time = new_frame_time
    # print(int(fps))


    # Show Base Frame
    frame = cv2.resize(frame,(frame_width, frame_height))
    cv2.imshow("Origin", frame)

    # GRAY Scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GRAY Scale", gray)

    # Mask
    mask = SetRange(gray)
    mask = process(mask, kernel)
    cv2.imshow("Mask", mask)

    # Remove background
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Remove Background", result)

    # Change background of mask
    # rest = cv2.bitwise_or(bkg, bkg, mask=mask)
    # rest = cv2.bitwise_xor(rest, bkg)
    # cv2.imshow("Bitwise OR: Mask <OR> Background", rest)

    # Change background after remove 
    # changebkg = cv2.bitwise_or(result, rest)
    # cv2.imshow("Change Background", changebkg)
   


    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
