import numpy as np
import cv2

class Pairs:
    def __init__(self, rfile, lfile, nPnt=40, im1='pic410.png', im2='pic430.png', path='G://CUNY/CV/Assignments/HW3/'):
        self.path = path
        self.nPoints = nPnt  # number of points to pick up
        self.img1 = cv2.imread(path+im1)
        self.img2 = cv2.imread(path+im2)
        self.width = self.img1.shape[1]
        self.imgComb = np.concatenate((self.img1, self.img2), axis=1)
        self.p = np.zeros((self.nPoints, 2))          # temp store the coordinates for selected points
        self.nClick = 0                     # used to count the number of points are selected
        self.rfile = rfile                  # the file name for storing points on right image
        self.lfile = lfile                  # the file name for storing points on left image



    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(self.nClick, end='   ')
            if self.nClick < self.nPoints:
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
        pl = np.zeros((self.nPoints//2, 2))
        pr = np.zeros((self.nPoints//2, 2))
        for i in range(self.p.shape[0]):
            if i%2 == 0:
                pl[i//2, :] = self.p[i, :]
            else:
                pr[i//2, :] = self.p[i, :]
        np.save(self.path+self.lfile, pl)
        np.save(self.path+self.rfile, pr)



class getF:
    def __init__(self, path='G://CUNY/CV/Assignments/HW3/', lfile = 'ctrl_pl.npy', rfile = 'ctrl_pr.npy' ):
        self.pl = np.load(path + lfile)
        self.pr = np.load(path + rfile)
        # convert to homogeneous coordinate
        self.pl = np.append(self.pl, np.ones((self.pl.shape[0], 1)), axis=1)
        self.pr = np.append(self.pr, np.ones((self.pr.shape[0], 1)), axis=1)
        # normalize points
        self.T = self.conNormMat()
        self.pl = self.pl@self.T.T
        self.pr = self.pr@self.T.T
        # create coefficient matrix A by using corresponding points in two images
        self.A = self.conA()


    def conA(self):
        A = [np.kron(self.pl[n, :], self.pr[n, :]) for n in range(self.pl.shape[0])]
        return np.array(A)


    def conNormMat(self):
        Xc = (np.mean(self.pl[:, 0]) + np.mean(self.pr[:, 0]))/2
        Yc = (np.mean(self.pl[:, 1]) + np.mean(self.pr[:, 1]))/2
        Ds = np.sqrt(Xc**2 + Yc**2)
        scale = np.sqrt(2) / Ds
        T = np.eye(3, 3)
        T[:, -1] = [-Xc, -Yc, 1/scale]
        T = scale*T
        return T


    def findF(self):
        U, D, V = np.linalg.svd(self.A)
        Fa = V[-1, :].reshape(3,3)
        # enforce singularity constraint
        u, d, v = np.linalg.svd(Fa)
        d[-1] = 0
        F = u@d@v
        return F


    def


gF = getF()
Fb = gF.findF()





