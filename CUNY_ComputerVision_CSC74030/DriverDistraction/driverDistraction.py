# load the packages
import cv2
import imutils
import drowse
import time
import dlib
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread

# constants
THRESHHOLD = 0.2
CONT_FRAME = 50
COUNT = 0
ALARM = False
path = 'G://CUNY/CV/Final Project/Eyes/'


# initialize face detector and facial landmark
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path+'shape_predictor_68_face_landmarks.dat')



# capture a video
vid = cv2.VideoCapture(path + 'test.mp4')
time.sleep(1.0)

while (True):
    # capture video frame
    ret, frame = vid.read()

    # preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # drowse or not
    drowsy, avg_ratio = drowse(THRESHHOLD, face_det, landmark, path, frame, gray)

    if drowsy:
        COUNT +=1
        if COUNT >= CONT_FRAME:
            if not ALARM:
                ALARM = True
            cv2.putText(frame, "DROWSINESS ALARM", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    else:
        COUNT = 0
        ALARM = False
        cv2.putText(frame, "Ratio: {:.2f}".format(avg_ratio), (200, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255),
                    2)

    # show the frame
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()
vid.release()