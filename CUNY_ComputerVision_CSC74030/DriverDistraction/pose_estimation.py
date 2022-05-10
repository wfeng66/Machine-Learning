from video import Video
from detection import Detection, Landmarks
# from calibration import Calibration
import cv2
import numpy as np
import utils
import dlib
import math


class Pose():
    def __init__(self):
        self.axis = np.float32([[200, 0, 0], [0, 200, 0], [0, 0, 200], [0, 0, 0]]).reshape(-1, 3)

    def draw_axes(self, imgpt, projpt, frame):
        frame = cv2.line(frame, imgpt, tuple(
            projpt[0].ravel().astype(int)), (255, 0, 0), 3)
        frame = cv2.line(frame, imgpt, tuple(
            projpt[1].ravel().astype(int)), (0, 255, 0), 3)
        frame = cv2.line(frame, imgpt, tuple(
            projpt[2].ravel().astype(int)), (0, 0, 255), 3)

    def estimate(self, frame, detection, intrinsic, allpts=False):
        font = cv2.FONT_HERSHEY_SIMPLEX

        distortion = np.zeros((4, 1))

        model_points = np.array([(0.0, 0.0, 0.0),  # Nose tip
                                 (0.0, -330.0, -65.0),  # Chin
                                 (-225.0, 170.0, -135.0),  # Left eye left corner
                                 (225.0, 170.0, -135.0),  # Right eye right corne
                                 (-150.0, -150.0, -125.0),  # Left Mouth corner
                                 (150.0, -150.0, -125.0)  # Right mouth corner
                                 ])

        image_points = np.array([detection.nose,
                                 detection.chin,
                                 detection.Leye_corner,
                                 detection.Reye_corner,
                                 detection.Lmouth_corner,
                                 detection.Rmouth_corner], dtype='double')

        if allpts:
            model_points, image_points = utils.get_all_68_pts(detection=detection)

        # perspective n points of model and image points
        _, rotation, translation = cv2.solvePnP(model_points, image_points, intrinsic,
                                                distortion, flags=cv2.SOLVEPNP_UPNP)

        # refines both rotation and translation
        if not allpts:
            rotation, translation = cv2.solvePnPRefineLM(model_points, image_points, intrinsic,
                                                         distortion, rotation, translation)

            # nose point on image plane
            nose = int(detection.nose[0]), int(detection.nose[1])
            # computes 3 3 projection axis form the nose using the parameters we calculated
            nose_end_2D, _ = cv2.projectPoints(self.axis, rotation, translation,
                                               intrinsic, distortion)
            self.draw_axes(nose, nose_end_2D, frame)

        # get 3x3 rotation matrix from rotation vector
        rotation = rotation.reshape((3,))
        rotation, _ = cv2.Rodrigues(rotation)

        proj_matrix = np.hstack((rotation, translation))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        self.pitch = math.degrees(math.asin(math.sin(pitch)))
        self.roll = -math.degrees(math.asin(math.sin(roll)))
        self.yaw = math.degrees(math.asin(math.sin(yaw)))

        cv2.putText(frame, "pitch: " + str(np.round(self.pitch, 2)), (50, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "yaw: " + str(np.round(self.yaw, 2)), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    2)
        cv2.putText(frame, "roll: " + str(np.round(self.roll, 2)), (50, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

        return self.pitch, self.yaw, self.roll