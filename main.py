import cv2
import mediapipe as mp
import numpy as np
mpose=mp.solutions.pose
mpdraw=mp.solutions.drawing_utils
pose=mpose.Pose()
mpdraw1=mpdraw.DrawingSpec(thickness=2,circle_radius=3,color=(0,0,225))
mpdraw2=mpdraw.DrawingSpec(thickness=2,circle_radius=3,color=(0,225,0))
cap=cv2.VideoCapture('boy_-_21827 (360p).mp4')
while True:
    success, img = cap.read()
    img = cv2.resize(img,(800,700))
    result=pose.process(img)

    h,w,c=img.shape
    blank=np.zeros([h,w,c])
    blank.fill(255)
    mpdraw.draw_landmarks(blank, result.pose_landmarks, mpose.POSE_CONNECTIONS, mpdraw1, mpdraw2)
    cv2.imshow('extract',blank)
    cv2.imshow('image',img)
    cv2.waitKey(1)
