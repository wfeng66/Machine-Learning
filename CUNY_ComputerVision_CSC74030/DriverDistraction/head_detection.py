from video import Video
from pose_estimation import Pose
from detection import Detection
import cv2


# https://github.com/e-candeloro/Driver-State-Detection/blob/master/driver_state_detection/Driver_State_Detection.py

class DistractionScore():
    def __init__(self, ROLL_THRESH, PITCH_THRESH, YAW_THRESH, POSE_FRAME_THRESH=50.0):

        self.ROLL_THRESH = ROLL_THRESH
        self.PITCH_THRESH = PITCH_THRESH
        self.YAW_THRESH = YAW_THRESH

        self.FRAME_THRESH = POSE_FRAME_THRESH

        self.pose_counter = 0

    def evaluate(self, pitch, roll, yaw):
        distracted = False
        if ((abs(roll) > self.ROLL_THRESH) or (abs(pitch) > self.PITCH_THRESH) or (abs(yaw) > self.YAW_THRESH)):
            return True
        else:
            return False
"""
        if self.pose_counter >= self.FRAME_THRESH:
            distracted = True

        if ((abs(roll) > self.ROLL_THRESH) or (abs(pitch) > self.PITCH_THRESH) or (abs(yaw) > self.YAW_THRESH)):
            if not distracted:
                self.pose_counter += 1

        elif self.pose_counter > 0:
            self.pose_counter -= 1

        return distracted
        
"""