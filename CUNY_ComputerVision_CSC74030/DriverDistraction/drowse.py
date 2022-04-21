
def eye_ratio(eye):
  # euclidean distances in vertical
  from scipy.spatial import distance as dist
  A = dist.euclidean(eye[1], eye[5])
  B = dist.euclidean(eye[2], eye[4])
  # euclidean distances in horizontal
  C = dist.euclidean(eye[0], eye[3])
  return  (A+B)/(2.0*C)


def shape_to_np(shape, dtype="int"):
    # initialize
    import numpy as np
    coords = np.zeros((68, 2), dtype=dtype)
    # landmark to numpy
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def drowse(tH, face_det, landmark, path, frame, gray):
    import cv2
    # detect faces
    rects = face_det(gray, 0)

    # loop faces
    # in this case, there should be only one face inside it
    for rect in rects:
        # detect facial landmarks
        shape = landmark(gray, rect)
        shape = shape_to_np(shape)

        # extract eyes coordinates and ratios
        left = shape[42:48]
        right = shape[36: 42]
        left_ratio = eye_ratio(left)
        right_ratio = eye_ratio(right)

        # average ratio between two eyes
        avg_ratio = (left_ratio + right_ratio) / 2.0

        # visualize eye
        leftEyeHull = cv2.convexHull(left)
        rightEyeHull = cv2.convexHull(right)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # condition, drowse or not
        if avg_ratio < tH:
            return True, avg_ratio
        else:
            return False, avg_ratio
