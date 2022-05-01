from scipy.spatial import distance as dist
import numpy as np

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
    mouth_ratio = vh_ratio((shape[51], shape[57]), (shape[62], shape[66]), (shape[48], shape[54]))
    if hist_q.full():
        pre_ratio = hist_q.get()
        hist_q.put(mouth_ratio)
        change_ratio = np.abs(mouth_ratio - pre_ratio)/pre_ratio
        print(mouth_ratio, change_ratio)
        if change_ratio > tH:
            return True
        else:
            return False
    else:
        hist_q.put(mouth_ratio)
        return False


def drowseNspeak(tH_d, tH_s, s_queue, face_det, landmark, gray):
    # detect faces
    rects = face_det(gray, 0)
    drowsy, speaking = False, False
    # loop faces
    # in this case, there should be only one face inside it
    for rect in rects:
        # detect facial landmarks
        shape = landmark(gray, rect)
        shape = shape_to_np(shape)

        # extract eyes coordinates
        left_eye = shape[42:48]
        right_eye = shape[36: 42]
        drowsy = drowse(tH_d, left_eye, right_eye)
        speaking = speak(tH_s, s_queue, shape)

    return [drowsy, speaking]



