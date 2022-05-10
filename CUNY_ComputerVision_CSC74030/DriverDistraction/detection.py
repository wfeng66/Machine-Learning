import cv2
from cv2 import imwrite
import dlib
# import utils
from video import Video
import numpy as np
from copy import deepcopy
import drowse1

path = 'shape_predictor_68_face_landmarks.dat'
face_det = dlib.get_frontal_face_detector()
landmark = dlib.shape_predictor(path)


class Landmarks():
    def __init__(self, shape, face):
        self.shape = shape
        self.face = face

        self.left_eye = self.shape[36:42]
        self.right_eye = self.shape[42:48]
        self.mouth = [self.shape[48], self.shape[54], self.shape[51], self.shape[62], self.shape[66], self.shape[57]]
        self.nose = self.shape[30]
        self.chin = self.shape[8]

        self.Leye_corner = self.left_eye[0]
        self.Reye_corner = self.right_eye[3]
        self.Lmouth_corner = self.mouth[0]
        self.Rmouth_corner = self.mouth[1]


class Detection():
    def __init__(self, video):
        super().__init__()
        self.video = video
        self.video.window_title = 'Landmark Detection'

    def detect_face(self):
        if self.video.img is not None:
            faces = face_det(self.video.img, 0)
            return faces

    def detect_features(self, face):
        gray = cv2.cvtColor(self.video.img, cv2.COLOR_BGR2GRAY)
        self.shape = landmark(gray, face)
        self.shape = drowse1.shape_to_np(self.shape)
        landmarks = deepcopy(self.shape)

        self.face = ((face.left(), face.top()), (face.right(), face.bottom()))

        self.left_eye = self.shape[36:42]
        self.right_eye = self.shape[42:48]
        self.mouth = [self.shape[48], self.shape[54], self.shape[51], self.shape[62], self.shape[66], self.shape[57]]
        self.nose = self.shape[30]
        self.chin = self.shape[8]

        self.Leye_corner = self.left_eye[0]
        self.Reye_corner = self.right_eye[3]
        self.Lmouth_corner = self.mouth[0]
        self.Rmouth_corner = self.mouth[1]

        return landmarks, self.face

    def detect_landmarks(self, show='HPE'):
        faces = self.detect_face()
        if faces is None:
            return None
        else:
            for face in faces:
                landmarks, face = self.detect_features(face)
                self.draw_landmarks(self.shape, show)
                return landmarks, face

    def draw_landmarks(self, shape, show='HPE'):
        cv2.rectangle(self.video.img, self.face[0],
                      self.face[1], (0, 255, 0), 2)
        if show == 'HPE':
            pts = [self.nose, self.chin, self.Leye_corner, self.Reye_corner, self.Lmouth_corner, self.Rmouth_corner]
            for pt in pts:
                cv2.circle(self.video.img, pt, 2, (255, 0, 0), -1)

        elif show == 'GAZE':
            feats = [self.left_eye, self.right_eye, self.mouth, [self.nose], [self.chin]]
            for pts in feats:
                for pt in pts:
                    cv2.circle(self.video.img, pt, 2, (255, 0, 0), -1)

        else:
            for i in range(len(shape)):
                cv2.circle(self.video.img, shape[i], 1, (255, 0, 0), -1)

    def get_head(self):
        while True:
            self.get_frame()
            self.detect_landmarks()
            self.show_frame()
            if cv2.waitKey(1) & 0xFF == 27:
                # cv2.imwrite('data\calibration_data\head.jpg',self.img)
                break
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video = Video(0)
    detector = Detection(video)
    while True:
        video.get_frame()
        l, f = detector.detect_landmarks(show='GAZE')
        print(type(l))
        video.show_frame()
        if cv2.waitKey(1) & 0xFF == 27:
            break
    video.vid.release()
    cv2.destroyAllWindows()