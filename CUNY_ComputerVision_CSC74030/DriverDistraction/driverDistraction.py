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
SPEAKING_THRESHOLD = 0.05
CONT_FRAME = 10             # if the number of frame positive exceed this number, DROWSE_ALARM
DROWSE_COUNT = 0
SPEAKING_COUNT = 0
DROWSE_ALARM = False
SPEAKING_ALARM = False
path = 'G://CUNY/CV/Final Project/Eyes/'
speaking_q = Queue(maxsize=5)

# initialize face detector and facial landmark
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path + 'shape_predictor_68_face_landmarks.dat')



# capture a video
cap = cv2.VideoCapture('G://CUNY/CV/Final Project/Data/test1.mp4')
time.sleep(1.0)

while (cap.isOpened()):
    # capture video frame
    _, frame = cap.read()

    # preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # drowse or not
    [drowsy, speaking] = drowse1.drowseNspeak(DROWSE_THRESHHOLD, SPEAKING_THRESHOLD, speaking_q, face_det, landmark, gray)

    if drowsy:
        DROWSE_COUNT +=1
        if DROWSE_COUNT >= CONT_FRAME:
            if not DROWSE_ALARM:
                DROWSE_ALARM = True
            cv2.putText(frame, "DROWSINESS DROWSE_ALARM", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    else:
        DROWSE_COUNT = 0
        DROWSE_ALARM = False

    if speaking:
        SPEAKING_COUNT += 1
        if SPEAKING_COUNT >= CONT_FRAME:
            if not SPEAKING_ALARM:
                SPEAKING_ALARM = True
            cv2.putText(frame, "SPEAKING SPEAK_ALARM", (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
    else:
        SPEAKING_COUNT = 0
        SPEAKING_ALARM = False

        
    # show the frame
    cv2.imshow("Video", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

