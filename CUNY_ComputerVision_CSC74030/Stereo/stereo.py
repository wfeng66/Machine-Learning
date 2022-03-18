import numpy as np
import cv2

class Pairs:
    def __init__(self, im1='pic410.png', im2='pic430.png', path='G://CUNY/CV/Assignments/HW3/'):
        self.path = path
        self.img1 = cv2.imread(path+im1)
        self.img2 = cv2.imread(path+im2)
        self.imgComb = np.concatenate((self.img1, self.img2), axis=1)
        self.p = np.zeros((40, 2))          # temp store the coordinates for selected points
        self.nClick = 0                     # used to count the number of points are selected


    def click_event(self, event, x, y, flags, params):
        if self.nClick < 40:
            self.p[self.nClick, 0] = x
            self.p[self.nClick, 1] = y
        else:
            self.saveCoor()
        self.nClick += 1



    def captureCoor(self):
        text = "Please select 21 pairs of points on left image and right image: "
        self.imgComb = cv2.putText(self.imgComb, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        cv2.imshow('Campus', self.imgComb)
        cv2.setMouseCallback('Campus', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def saveCoor(self):
        pl = np.zeros((20, 2))
        pr = np.zeros((20, 2))
        for i in range(self.p.shape[0]):
            if i%2 == 0:
                pl[i//2, :] = self.p[i, :]
            else:
                pr[i//2, :] = self.p[i, :]
        np.savetxt(self.path+'pl.txt', pl)
        np.savetxt(self.path + 'pr.txt', pr)



p = Pairs()
p.captureCoor()


