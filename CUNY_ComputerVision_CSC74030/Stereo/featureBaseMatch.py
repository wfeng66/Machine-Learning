import numpy as np
import cv2
import stereo

# load the images
img1 = cv2.imread('G://CUNY/CV/Assignments/HW3/pic410.png',0)
img2 = cv2.imread('G://CUNY/CV/Assignments/HW3/pic430.png',0)

ctrl_l = np.load('G://CUNY/CV/Assignments/HW3/ctrl_pl.npy')
ctrl_r = np.load('G://CUNY/CV/Assignments/HW3/ctrl_pr.npy')
test_l = np.load('G://CUNY/CV/Assignments/HW3/test_pl.npy')
test_r = np.load('G://CUNY/CV/Assignments/HW3/test_pr.npy')
ctrl_l = np.int32(ctrl_l)
ctrl_r = np.int32(ctrl_r)
test_l = np.int32(test_l)
test_r = np.int32(test_r)

F, mask = cv2.findFundamentalMat(ctrl_l,ctrl_r,cv2.FM_LMEDS)

ctrl_l_homo = np.append(ctrl_l, np.ones((ctrl_l.shape[0], 1)), axis=1)
ctrl_r_homo = np.append(ctrl_r, np.ones((ctrl_l.shape[0], 1)), axis=1)
test_l_homo = np.append(test_l, np.ones((test_l.shape[0], 1)), axis=1)
test_r_homo = np.append(test_r, np.ones((test_l.shape[0], 1)), axis=1)

def getEpLine(r2l, F, p):
    if r2l == 1:
        lines = F.T @ p.T
    else:
        lines = F @ p.T
    return lines.T


def drawlines(img1,img2,lines,p1,p2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,p1,p2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        # img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1

lines_r = getEpLine(0, F, test_l_homo)
lines_l = getEpLine(1, F, test_r_homo)


# cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)


img_l = drawlines(img1,img2,lines_l,test_l,test_r)
img_r = drawlines(img2,img1,lines_r,test_r,test_l)
# img_l = drawlines(img1,img2,lines_l,ctrl_l,ctrl_r)
# img_r = drawlines(img2,img1,lines_r,ctrl_r,ctrl_l)


cv2.imshow('Campus_left', img_l)
cv2.imshow('Campus_right', img_r)
cv2.waitKey(0)


