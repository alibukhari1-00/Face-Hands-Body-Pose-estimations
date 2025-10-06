import cv2
import mediapipe as mp
import numpy as np


feed=cv2.VideoCapture(0)

while True:
    ret,frame=feed.read()
    frame=cv2.resize(frame,(1350,640))
    frame=cv2.flip(frame,1)
    if not ret:
        print("Frame not Found !!!!!")
        break
    cv2.imshow("Webcame Feed ",frame)

    if cv2.waitKey(1) & 0XFF==27:
       break
feed.release()
cv2.destroyAllWindows()