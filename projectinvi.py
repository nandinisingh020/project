import cv2
import numpy as np
import time

#open the webcam
cap = cv2.VideoCapture(0)

#allow the camera to warm up
time.sleep(3)

#capture the background
background = 0
for i in range(30):
    ret, background = cap.read()
    if not ret:
        continue

#flip the background (to match mirror view)
background = np.flip(background, axis=1)

print("Background captured! Let's begin the magic!!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #flip the frame
    frame = np.flip(frame, axis=1)

    #convert frame to HSV color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #define HSV range for pink cloak
    lower_pink = np.array([140, 80, 80])   # lower HSV bound
    upper_pink = np.array([170, 255, 255]) # upper HSV bound

    #create mask for pink color
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    #refining mask (morphological operations)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    #inverse mask for non-cloak area
    inverse_mask = cv2.bitwise_not(mask)

    #segment cloak and background
    res1 = cv2.bitwise_and(frame, frame, mask=inverse_mask)  # everything except cloak
    res2 = cv2.bitwise_and(background, background, mask=mask) # cloak area replaced by background

    #final output
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    #show result
    cv2.imshow("Invisible Cloak - Pink", final_output)

    #quit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()