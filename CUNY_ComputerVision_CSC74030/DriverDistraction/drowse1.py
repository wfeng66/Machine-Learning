from scipy.spatial import distance as dist
import cv2

def vh_ratio(v1, v2, h):
    # this function use to calculate the ratio of vertical distance and horizontal distance
    # input:    v1 - tuple which include two points' coordinates aligned in vertical that to be calculated the euclidean distance
    #           v2 - tuple which include two points' coordinates aligned in vertical that to be calculated the euclidean distance
    #           h  - tuple which include two points' coordinates aligned in horizontal that to be calculated the euclidean distance
    # output:   the average ratio of vertical distance and horizontal distance
    # euclidean distances in vertical
    A = dist.euclidean(v1[0], v1[1])
    B = dist.euclidean(v2[0], v2[1])
    # euclidean distances in horizontal
    C = dist.euclidean(h[0], h[1])
    return  (A+B)/(2.0*C)


def shape_to_np(shape, dtype="int"):
    # initialize
    import numpy as np
    coords = np.zeros((68, 2), dtype=dtype)
    # landmark to numpy
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def drowse(tH, left_eye, right_eye):
    left_ratio = vh_ratio((left_eye[1], left_eye[5]), (left_eye[2], left_eye[4]), (left_eye[0], left_eye[3]))
    right_ratio = vh_ratio((right_eye[1], right_eye[5]), (right_eye[2], right_eye[4]), (right_eye[0], right_eye[3]))

    # average ratio between two eyes
    avg_ratio = (left_ratio + right_ratio) / 2.0

    # condition, drowse or not
    if avg_ratio < tH:
        return True
    else:
        return False



def speak(tH, hist_q, shape):
    A = dist.euclidean(shape[61], shape[67])
    B = dist.euclidean(shape[63], shape[65])
    # mouth_ratio = vh_ratio((shape[51], shape[57]), (shape[62], shape[66]), (shape[48], shape[54]))
    # print(A, B)
    if B > 4 or A > 4:
        return True
    else:
        return False


def voor(tH_side, tH_ch, left_ratio_ori, chin_ori, shape, nFrame):
    right = dist.euclidean(shape[1], shape[33])
    left = dist.euclidean(shape[15], shape[33])
    chin = dist.euclidean(shape[8], shape[33])
    left_ratio = left / (left + right)
    # print(nFrame, left_ratio, left_ratio_ori, left_ratio - left_ratio_ori, tH_side)
    if left_ratio - left_ratio_ori > tH_side:
        # print("True")
        return True
    else:
        # print("False")
        return False

    if chin < (1-tH_ch)*chin_ori:
        return 'chin'
    if left_ratio - left_ratio_ori > tH_side:
        print('Side', left_ratio - left_ratio_ori)
        return 'side'
    return False

    if chin < (1-tH_ch)*chin_ori or (left_ratio - left_ratio_ori) > tH_side:
        # print("True")
        return True
    else:
        # print("False")
        return False



# def pose():



def getFaceBase(face_det, landmark, cap):
    # detect faces
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_det(gray, 0)
    # loop faces
    # in this case, there should be only one face inside it
    while len(rects) == 0:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_det(gray, 0)
    for rect in rects:
        # detect facial landmarks
        shape = landmark(gray, rect)
        shape = shape_to_np(shape)
        right = dist.euclidean(shape[1], shape[33])
        left = dist.euclidean(shape[15], shape[33])
        chin = dist.euclidean(shape[8], shape[33])
        left_ratio = left / (left+right)
    return left_ratio, chin




def detect(tH_d, tH_s, tH_side, tH_ch, s_queue, left_ratio, chin, face_det, landmark, gray, nFrame):
    # detect faces
    rects = face_det(gray, 0)
    drowsy, speaking, VOOR = False, False, False         # VOOR stands for vision out of road
    # loop faces
    # in this case, there should be only one face inside it
    for rect in rects:
        # detect facial landmarks
        shape = landmark(gray, rect)
        if shape is None:
            print('No shape')
            continue
        else:
            shape = shape_to_np(shape)

        # extract eyes coordinates
        left_eye = shape[42:48]
        right_eye = shape[36: 42]
        drowsy = drowse(tH_d, left_eye, right_eye)
        speaking = speak(tH_s, s_queue, shape)
        VOOR = voor(tH_side, tH_ch, left_ratio, chin, shape, nFrame)

    return [drowsy, speaking, VOOR]


