# load the packages
import cv2
import imutils
import pandas as pd
import drowse1
import time
import dlib
from threading import Thread
from queue import Queue
import os
import tensorflow as tf
import numpy as np
import keras

# constants
DROWSE_THRESHHOLD = 0.2
SPEAKING_THRESHOLD = 0.05
SIDE_THRESHOLD = 0.3
CHIN_THRESHOLD = 0.05
CONT_FRAME_DROWSE = 30             # if the number of frame positive exceed this number, DROWSE_ALARM
CONT_FRAME_SPEAKING = 4
CONT_FRAME_VISION = 30
DROWSE_ALARM = False
SPEAKING_ALARM = False
projpath = 'G://CUNY/CV/Final Project/'
datapath = 'G://CUNY/CV/Final Project/Data/'
cvpath = 'G://CUNY/CV/Final Project/Eyes/'
dlpath = 'G://CUNY/CV/Final Project/DL/vgg16'
speaking_q = Queue(maxsize=5)
dl_cls_dict ={
    0: 0,
    1: 1,
    2: 9,
    3: 1,
    4: 9,
    5: 1,
    6: 9,
    7: 1,
    8: 9,
    9: 2
}

# initialize face detector and facial landmark
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(cvpath + 'shape_predictor_68_face_landmarks.dat')
vgg16 = keras.models.load_model(dlpath)

def cv(frame, left_ratio, chin, nFrame):
    # preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect distracted driving
    [drowsy, speaking, voor] = drowse1.detect(DROWSE_THRESHHOLD, SPEAKING_THRESHOLD, SIDE_THRESHOLD, CHIN_THRESHOLD,
                                              speaking_q, left_ratio, chin, face_det, landmark, gray, nFrame)

    if drowsy:
        return 3
    if speaking:
        return 2
    if voor:
        return 1
    else:
        return 0


def dl(model, img):
    X = cv2.resize(img, (224, 224))
    X = np.expand_dims(X, axis=0)
    pred = model.predict(X)
    pred = np.argmax(pred[0])
    return dl_cls_dict[pred]


def detect_noTemporal(cap):
    count_frame = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rslts = [[]]
    while count_frame < length-1:
        # capture video frame
        ret, frame = cap.read()
        if ret == False:
            rslts.append([count_frame, 0, 0])
        else:
            if count_frame == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                left_ratio, chin = drowse1.getFaceBase(face_det, landmark, cap)
            else:
                cv_c = cv(frame, left_ratio, chin, count_frame)
                dl_c = dl(vgg16, frame)
                rslts.append([count_frame, cv_c, dl_c])
        count_frame += 1
    return rslts


def detect_withTemporal(cap):
    count_frame = 0
    DROWSE_COUNT = 0
    SPEAKING_COUNT = 0
    VOOR_COUNT = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))     # the number of frames in a video
    rslts = [[]]
    while count_frame < length-1:
        # capture video frame
        ret, frame = cap.read()
        if ret == False:
            rslts.append([count_frame, 0])
        else:
            if count_frame == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                left_ratio, chin = drowse1.getFaceBase(face_det, landmark, cap)
            else:
                cv_c = cv(frame, left_ratio, chin, count_frame)
            if cv_c ==3:
                DROWSE_COUNT += 1
                if DROWSE_COUNT < CONT_FRAME_DROWSE:
                    cv_c = 0
            else:
                DROWSE_COUNT = 0
            if cv_c == 2:
                SPEAKING_COUNT += 1
                if SPEAKING_COUNT < CONT_FRAME_SPEAKING:
                    cv_c = 0
            else:
                SPEAKING_COUNT = 0
            if cv_c == 1:
                VOOR_COUNT += 1
                if VOOR_COUNT < CONT_FRAME_VISION:
                    cv_c = 0
            else:
                VOOR_COUNT = 0
            rslts.append([count_frame, cv_c])
        count_frame += 1
    return rslts


def compensate(rslts):
    # compensate the frames before count for dangerous behaviors and short lost value
    for i in range(len(rslts)):
        if rslts.loc[i, "CV_T"] == 3:
            if i - CONT_FRAME_DROWSE > 0:
                rslts.loc[i-CONT_FRAME_DROWSE:i, "CV_T"] = 3
                if 3 in rslts.loc[i-CONT_FRAME_DROWSE*2: i-CONT_FRAME_DROWSE, "CV_T"].values:
                    rslts.loc[i - CONT_FRAME_DROWSE * 2: i - CONT_FRAME_DROWSE, "CV_T"] = 3
            else:
                rslts.loc[:i, "CV_T"] = 3
        if rslts.loc[i, "CV_T"] == 2:
            if i - CONT_FRAME_SPEAKING > 0:
                rslts.loc[i-CONT_FRAME_SPEAKING:i, "CV_T"] = 2
                if 2 in rslts.loc[i-CONT_FRAME_SPEAKING*2: i-CONT_FRAME_SPEAKING, "CV_T"].values:
                    rslts.loc[i - CONT_FRAME_SPEAKING * 2: i - CONT_FRAME_SPEAKING, "CV_T"] = 2
            else:
                rslts.loc[:i, "CV_T"] = 2
    return rslts



if __name__ == '__main__':
    vidList = [file for file in os.listdir(datapath) if file.endswith(".mp4")]
    rslts1 = pd.DataFrame(columns=['fName', 'nFrame', 'CV', 'DL'])
    rslts2 = pd.DataFrame(columns=['fName', 'nFrame', 'CV_T'])
    for vid in vidList:
        print('Processing ' + vid + '...')
        # capture a video
        cap = cv2.VideoCapture(datapath + vid)
        rslt1 = detect_noTemporal(cap)
        rslt1 = pd.DataFrame(rslt1, columns=['nFrame', 'CV', 'DL'])
        rslt1['fName'] = vid
        rslts1 = rslts1.append(rslt1.iloc[1:-1, :])
        rslt2 = detect_withTemporal(cap)
        rslt2 = pd.DataFrame(rslt2, columns=['nFrame', 'CV_T'])
        rslt2['fName'] = vid
        rslt2 = compensate(rslt2)
        rslts2 = rslts2.append(rslt2.iloc[1:-1, :])
    rslts = pd.merge(rslts1, rslts2, on=['fName', 'nFrame'])
    rslts.to_csv(projpath + 'cv&dl.csv')


