import numpy as np
import cv2

class Pairs:
    def __init__(self, im1='pic410.png', im2='pic430.png', path='G://CUNY/CV/Assignments/HW3/'):
        self.path = path
        self.img1 = cv2.imread(path+im1)
        self.img2 = cv2.imread(path+im2)
        self.width = self.img1.shape[1]
        self.imgComb = np.concatenate((self.img1, self.img2), axis=1)
        self.p = np.zeros((40, 2))          # temp store the coordinates for selected points
        self.nClick = 0                     # used to count the number of points are selected
        print(self.img1.shape[0], self.img1.shape[1])
        print(self.imgComb.shape[0], self.imgComb.shape[1])


    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(self.nClick, end='   ')
            if self.nClick < 40:
                print(x, y)
                self.p[self.nClick, 1] = y
                if self.nClick%2 == 0:
                    self.p[self.nClick, 0] = x
                else:
                    self.p[self.nClick, 0] = x - self.width
                self.imgComb = cv2.circle(self.imgComb, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                # cv2.imshow('Campus', self.imgComb)
            else:
                self.saveCoor()
            self.nClick += 1



    def captureCoor(self):
        text = "Please select 21 pairs of points on left image and right image: (press 'c' after finishing)"
        self.imgComb = cv2.putText(self.imgComb, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        while True:
            cv2.imshow('Campus', self.imgComb)
            cv2.setMouseCallback('Campus', self.click_event)
            k = cv2.waitKey(10) & 0XFF
            if k == 99:
                break


    def saveCoor(self):
        pl = np.zeros((20, 2))
        pr = np.zeros((20, 2))
        for i in range(self.p.shape[0]):
            if i%2 == 0:
                pl[i//2, :] = self.p[i, :]
            else:
                pr[i//2, :] = self.p[i, :]
        np.save("G://CUNY/CV/Assignments/HW3/pl.npy", pl)
        np.save("G://CUNY/CV/Assignments/HW3/pr.npy", pr)



p = Pairs()
p.captureCoor()
print("Well Done!")


