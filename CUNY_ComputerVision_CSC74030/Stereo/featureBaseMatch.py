import numpy as np
import cv2
import stereo

class featureMatch:
    def __init__(self, rfile='ctrl_pr.npy', lfile='ctrl_pl.npy', im1='pic410.png', im2='pic430.png', path='G://CUNY/CV/Assignments/HW3/'):
        self.path = path
        self.img1 = cv2.imread(path+im1, 0)
        self.img2 = cv2.imread(path+im2, 0)
        self.width = self.img1.shape[1]
        self.ctrl_l = np.load(path+lfile)
        self.ctrl_r = np.load(path+rfile)
        self.F, _ = cv2.findFundamentalMat(np.int32(self.ctrl_l), np.int32(self.ctrl_r), cv2.FM_LMEDS)



    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.match(y, x, 5, 5)            # switch x and y
            self.img1 = cv2.circle(self.img1, (x, y), radius=5, color=(0, 0, 255), thickness=-1)


    def match(self, x, y, x_win, y_win):
        # edges = cv2.Canny(self.img1, 100, 200)
        # cv2.imshow('edges', edges)

        # extract feature descriptor from img1
        ftr1 = self.img1[x-x_win: x+x_win+1, y-y_win: y+y_win+1]
        ftr1 = (ftr1-np.mean(ftr1))/np.std(ftr1)

        # find the epipolar line for the coordinate
        [pt_homo] = stereo.homo([np.array([y, x]).reshape((-1,2))])
        print(pt_homo)
        line = stereo.getEpLine(0, self.F, pt_homo)
        [line] = line

        # search matching
        dst = {}
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x_low, x_high = x-150, x+150
        for x_s in range(x_low, x_high):
            y_c = int(-(line[0]*x_s+line[2])/line[1])
            y_low, y_high = y_c - 5, y_c + 5
            for y_s in range(y_low, y_high):
                ftr2 = self.img2[x_s-x_win: x_s+x_win+1, y_s-y_win: y_s+y_win+1]
                dst[(x_s, y_s)] = np.sum(np.abs(ftr1-ftr2))
        corr_pt = min(dst, key=dst.get)
        self.img2 = cv2.circle(self.img2, tuple(corr_pt), 5, color, -1)
        cv2.imshow('Campus2', self.img2)
        cv2.waitKey(0)


    def captureCoor(self):
        text = "Select the points on the left image: (press 'c' after finishing)"
        self.img1 = cv2.putText(self.img1, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        cv2.imshow('Campus2', self.img2)
        while True:
            cv2.imshow('Campus1', self.img1)
            cv2.setMouseCallback('Campus1', self.click_event)
            k = cv2.waitKey(10) & 0XFF
            if k == 99:
                break



# cv2.imshow('Campus_left', img_l)
# cv2.imshow('Campus_right', img_r)
# cv2.waitKey(0)

mat = featureMatch()
mat.captureCoor()

