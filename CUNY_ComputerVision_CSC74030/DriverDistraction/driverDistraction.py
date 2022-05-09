# load the packages
import cv2
import imutils
import drowse1
import time
import dlib
from queue import Queue
from threading import Thread

# constants
DROWSE_THRESHHOLD = 0.2
SPEAKING_THRESHOLD = 0.1
SIDE_THRESHOLD = 0.25
CHIN_THRESHOLD = 0.05
CONT_FRAME_DROWSE = 30             # if the number of frame positive exceed this number, DROWSE_ALARM
CONT_FRAME_SPEAKING = 3
CONT_FRAME_VISION = 30
DROWSE_COUNT = 0
SPEAKING_COUNT = 0
VOOR_COUNT = 0
DROWSE_ALARM = False
SPEAKING_ALARM = False
VOOR_ALARM = False
path = 'G://CUNY/CV/Final Project/Eyes/'
speaking_q = Queue(maxsize=3)
voor_sides = []
voor_chin = []
nFrame = 0

# initialize face detector and facial landmark
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path + 'shape_predictor_68_face_landmarks.dat')



# capture a video
cap = cv2.VideoCapture('G://CUNY/CV/Final Project/Data/IMG_3260.mp4')
time.sleep(1.0)

# get the base line data
left_ratio, chin = drowse1.getFaceBase(face_det, landmark, cap)

while (cap.isOpened()):
    # capture video frame
    _, frame = cap.read()

    # preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # print(nFrame, _)

    # drowse or not
    # [drowsy, speaking] = drowse1.drowseNspeak(DROWSE_THRESHHOLD, SPEAKING_THRESHOLD, speaking_q, face_det, landmark, gray)
    [drowsy, speaking, voor] = drowse1.detect(DROWSE_THRESHHOLD, SPEAKING_THRESHOLD, SIDE_THRESHOLD, CHIN_THRESHOLD, speaking_q,
                                              left_ratio, chin, face_det, landmark, gray, nFrame)

    cv2.putText(frame, str(nFrame), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

    if drowsy:
        DROWSE_COUNT +=1
        if DROWSE_COUNT >= CONT_FRAME_DROWSE:
            if not DROWSE_ALARM:
                DROWSE_ALARM = True
            cv2.putText(frame, "DROWSINESS DROWSE_ALARM", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    else:
        DROWSE_COUNT = 0
        DROWSE_ALARM = False

    if speaking:
        SPEAKING_COUNT += 1
        if SPEAKING_COUNT >= CONT_FRAME_SPEAKING:
            if not SPEAKING_ALARM:
                SPEAKING_ALARM = True
            cv2.putText(frame, "SPEAKING SPEAK_ALARM", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    else:
        SPEAKING_COUNT = 0
        SPEAKING_ALARM = False

    if voor == True:
        VOOR_COUNT += 1
        if VOOR_COUNT >= CONT_FRAME_VISION:
            if not VOOR_ALARM:
                VOOR_ALARM = True
            cv2.putText(frame, "VOOR_ALARM", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    else:
        VOOR_COUNT = 0
        VOOR_ALARM = False

    nFrame += 1

    # show the frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
"""
    if voor == 'side':
        VOOR_COUNT += 1
        if VOOR_COUNT >= CONT_FRAME_VISION:
            if not VOOR_ALARM:
                VOOR_ALARM = True
            cv2.putText(frame, "side_ALARM", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    elif voor == 'chin':
        VOOR_COUNT += 1
        if VOOR_COUNT >= CONT_FRAME_VISION:
            if not VOOR_ALARM:
                VOOR_ALARM = True
            cv2.putText(frame, "chin_ALARM", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    else:
        VOOR_COUNT = 0
        VOOR_ALARM = False
"""


cv2.destroyAllWindows()
cap.release()

