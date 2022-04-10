import numpy as np
import cv2
import stereo


"""
# load the images and point pairs
img1, img2, ctrl_l, ctrl_r, test_l, test_r = stereo.load()

F, mask = cv2.findFundamentalMat(ctrl_l,ctrl_r,cv2.FM_LMEDS)

[ctrl_l_homo, ctrl_r_homo, test_l_homo, test_r_homo] = stereo.homo([ctrl_l, ctrl_r, test_l, test_r])
print(F)

lines_r = getEpLine(0, F, test_l_homo)
lines_l = getEpLine(1, F, test_r_homo)


# cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)


img_l = drawlines(img1,img2,lines_l,test_l,test_r)
img_r = drawlines(img2,img1,lines_r,test_r,test_l)
# img_l = drawlines(img1,img2,lines_l,ctrl_l,ctrl_r)
# img_r = drawlines(img2,img1,lines_r,ctrl_r,ctrl_l)
"""

class featureMatch:
    def __init__(self, rfile='ctrl_pr.npy', lfile='ctrl_pl.npy', im1='pic410.png', im2='pic430.png', path='G://CUNY/CV/Assignments/HW3/'):
        self.path = path
        self.img1 = cv2.imread(path+im1)
        self.img2 = cv2.imread(path+im2)
        self.width = self.img1.shape[1]
        self.ctrl_l = np.load(path+lfile)
        self.ctrl_r = np.load(path+rfile)
        self.F, _ = cv2.findFundamentalMat(np.int32(self.ctrl_l), np.int32(self.ctrl_r), cv2.FM_LMEDS)



    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.match(x, y)
            self.img1 = cv2.circle(self.img1, (x, y), radius=5, color=(0, 0, 255), thickness=-1)



    def match(self, x, y):
        edges = cv2.Canny(self.img1, 100, 200)
        print(type(edges))
        print(edges.shape)
        print(edges)
        print(x, y)
        print(edges[x-3: x+3, y-3: y+3])
        cv2.imshow('edges', edges)



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

