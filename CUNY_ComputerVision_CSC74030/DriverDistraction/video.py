import cv2
import numpy as np


class Video:
    def __init__(self, cap):
        '''cap: 0 or title of video to read'''
        # if cap != 0:
        #     self.rescale(cap)
        #     cap = 'output.avi'

        self.vid = cv2.VideoCapture(cap)

        self.window_title = 'Testing...'

        self.width = 640.0
        self.focal_length = self.height = 480.0
        self.center = (self.width // 2, self.height // 2)
        self.intrinsic = np.array([[self.focal_length, 0, self.center[0]],
                                   [0, self.focal_length, self.center[1]],
                                   [0, 0, 1]])

    def get_frame(self):
        self.ret, self.img = self.vid.read()
        if self.ret:
            self.img = cv2.resize(self.img, (640, 480), interpolation=cv2.INTER_AREA)

    # def show_frame(self):
    #     while True:
    #         self.get_frame()
    #         cv2.imshow(self.window_title,self.img)
    #         if cv2.waitKey(1) & 0xFF == 27:
    #             print(self.width, self.height)
    #             break
    #     self.vid.release()
    #     cv2.destroyAllWindows()

    def show_frame(self):
        cv2.imshow(self.window_title, self.img)

    def rescale(self, cap):
        vid = cv2.VideoCapture(cap)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, 15.0, (640, 480))

        while True:
            ret, frame = vid.read()
            if ret == True:
                if (frame.shape[0] == 480) and (frame.shape[1] == 640):
                    break
                b = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
                out.write(b)
            else:
                break
        vid.release()
        out.release()
        cv2.destroyAllWindows()

    def show_point(self, point):
        cv2.circle(self.img, point, 5, (0, 0, 255), -5)


if __name__ == "__main__":
    vid = Video('test.mp4')
    # count = 0
    while True:
        # print(count)
        vid.get_frame()
        vid.show_frame()
        # count+=1

        if cv2.waitKey(1) & 0xFF == 27:
            print(vid.width, vid.height)
            break

    vid.vid.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # vid = Video(0)
#     vid = Video('test.mp4')
#     vid.show_frame()