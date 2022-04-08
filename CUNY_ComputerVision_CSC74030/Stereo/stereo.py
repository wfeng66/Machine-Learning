import numpy as np
import cv2

class Pairs:
    def __init__(self, rfile, lfile, nPnt=40, im1='pic410.png', im2='pic430.png', path='G://CUNY/CV/Assignments/HW3/'):
        self.path = path
        self.nPoints = nPnt                 # number of points to pick up
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
        self.img1 = cv2.imread(path+'pic410.png')
        self.pl = np.load(path + lfile)
        self.pr = np.load(path + rfile)
        self.plo = self.pl.copy()
        self.pro = self.pr.copy()
        # convert to homogeneous coordinate
        self.pl = np.append(self.pl, np.ones((self.pl.shape[0], 1)), axis=1)
        self.pr = np.append(self.pr, np.ones((self.pr.shape[0], 1)), axis=1)
        # normalize points
        self.Tl = self.conNormMat(self.pl[0], self.pl[1])
        self.Tr = self.conNormMat(self.pr[0], self.pr[1])
        self.pl = self.pl@self.Tl.T
        self.pr = self.pr@self.Tr.T
        # create coefficient matrix A by using corresponding points in two images
        self.A = self.conA()


    def conA(self):
        # A = [np.kron(self.pl[n, :], self.pr[n, :]) for n in range(self.pl.shape[0])]
        A = np.zeros((self.pl.shape[0], 9))
        for i in range(self.pl.shape[0]):
            # [xl*xr, yl*xr, xr, xl*yr, yl*yr, yr, xl, yl, 1]
            A[i, :] = [self.pl[i, 0]*self.pr[i, 0], self.pl[i, 1]*self.pr[i, 0], self.pr[i, 0], \
                       self.pl[i, 0]*self.pr[i, 1], self.pl[i, 1]*self.pr[i, 1], self.pr[i, 1], \
                       self.pl[i, 0], self.pl[i, 1], 1]

            #[xl*xr, yr*xl, xl, xr*yl, yl*yr, yl, xr, yr, 1]
            # A[i, :] = [self.pl[i, 0]*self.pr[i, 0], self.pr[i, 1]*self.pl[i, 0], self.pl[i, 0], \
            #            self.pr[i, 0]*self.pl[i, 1], self.pl[i, 1]*self.pr[i, 1], self.pl[i, 1], \
            #            self.pr[i, 0], self.pr[i, 1], 1]
        return A


    def conNormMat(self, x, y):
        Xc = np.mean(x)
        Yc = np.mean(y)
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
        Fn = u@np.diag(d)@v
        # F = u @ np.diag(d) @ v
        # denormalize
        F = self.Tr.T@Fn@self.Tl
        self.F = F
        return F


    def findEpipoles(self):
        U, D, V = np.linalg.svd(self.F)
        Epl = V[2, :]
        Epr = U[2, :]
        Epl = Epl / Epl[-1]
        Epr = Epr / Epr[-1]
        return Epl, Epr


    def epLine(self, p):
        return self.F@p




def drawlines(img1,img2,lines,p1,p2):
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,p1,p2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
    return img1


def getEpLine(r2l, F, p):
    if r2l == 1:
        lines = F.T @ p.T
    else:
        lines = F @ p.T
    return lines.T


def load():
    # load the images
    img1 = cv2.imread('G://CUNY/CV/Assignments/HW3/pic410.png', 0)
    img2 = cv2.imread('G://CUNY/CV/Assignments/HW3/pic430.png', 0)

    # load the points data
    ctrl_l = np.load('G://CUNY/CV/Assignments/HW3/ctrl_pl.npy')
    ctrl_r = np.load('G://CUNY/CV/Assignments/HW3/ctrl_pr.npy')
    test_l = np.load('G://CUNY/CV/Assignments/HW3/test_pl.npy')
    test_r = np.load('G://CUNY/CV/Assignments/HW3/test_pr.npy')
    ctrl_l = np.int32(ctrl_l)
    ctrl_r = np.int32(ctrl_r)
    test_l = np.int32(test_l)
    test_r = np.int32(test_r)
    return img1, img2, ctrl_l, ctrl_r, test_l, test_r


def homo(pts_list):
    rslts = []
    for pts in pts_list:
        homo = np.append(pts, np.ones((pts.shape[0], 1)), axis=1)
        rslts.append(homo)
    return rslts



"""

path='G://CUNY/CV/Assignments/HW3/'
im1='pic410.png'
im2='pic430.png'
img1 = cv2.imread(path+im1)
img2 = cv2.imread(path+im2)
imgComb = np.concatenate((img1, img2), axis=1)
gF = getF()
F = gF.findF()
# gF.findEpipoles()
# print(F)

# F, mask = cv2.findFundamentalMat(gF.pl, gF.pr, cv2.FM_LMEDS)
# gF.F = F
# gF.plo = np.int32(gF.plo)

# ll_prime = cv2.computeCorrespondEpilines(gF.plo[:, :2], 1, F)

for i in range(gF.pl.shape[0]):
# i = 5
    imgComb = cv2.circle(imgComb, (int(gF.pro[i, 0])+gF.img1.shape[1], int(gF.pro[i, 1])), radius=3, color=(0, 0, 255), thickness=-1)
    l_prime = gF.epLine(gF.plo[i, :])
    y1 = -l_prime[0]*0/l_prime[1] - l_prime[2]/l_prime[1]
    y2 = -l_prime[0]*gF.img1.shape[1]/l_prime[1] - l_prime[2]/l_prime[1]
    # y1 = -ll_prime[i, 0, 0] * 0 / ll_prime[i, 0, 1] - ll_prime[i, 0, 2] / ll_prime[i, 0, 1]
    # y2 = -ll_prime[i, 0, 0] * gF.img1.shape[1] / ll_prime[i, 0, 1] - ll_prime[i, 0, 2] / ll_prime[i, 0, 1]
    imgComb = cv2.line(imgComb, (gF.img1.shape[1], round(y1)), (gF.img1.shape[1]*2, round(y2)), (0, 255, 0), 1)
    cv2.imshow('Campus', imgComb)
    k = cv2.waitKey(10000000) & 0XFF
    if k == 99:
        continue




cv2.imshow('Campus', imgComb)
cv2.waitKey(0)


"""



