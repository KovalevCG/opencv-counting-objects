import cv2
import numpy as np

cap = cv2.VideoCapture("img/objects_01.mp4")

while True:

    _, resize = cap.read()
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)

    # Threshold image
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Denoising Image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)

    # Finding contours
    contours, hierarchy = cv2.findContours(sure_fg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Drawing circles inside objects
    for con in contours:
        M = cv2.moments(con)
        cX = int(M["m10"] / (M["m00"]+0.1))
        cY = int(M["m01"] / (M["m00"]+0.1))
        cv2.circle(resize, (cX, cY), 7, (50, 255, 55), -1)

    # Put text, Number of objects counted
    resize[10:100, 10:140] = (0, 0, 0)
    cv2.putText(resize, str(len(contours)), (20, 80), 1, 5, (230, 200, 180), 5)

    final = resize
    cv2.imshow("Objects Counting", final)

    key = cv2.waitKey(15)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
