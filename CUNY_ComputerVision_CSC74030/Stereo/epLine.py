import numpy as np
import cv2
import stereo


# load the images and point pairs
img1, img2, ctrl_l, ctrl_r, test_l, test_r = stereo.load()

# calculate the fundamental matrix
F, mask = cv2.findFundamentalMat(ctrl_l,ctrl_r,cv2.FM_LMEDS)

[ctrl_l_homo, ctrl_r_homo, test_l_homo, test_r_homo] = stereo.homo([ctrl_l, ctrl_r, test_l, test_r])

# find epolar lines
lines_r = stereo.getEpLine(0, F, test_l_homo)
lines_l = stereo.getEpLine(1, F, test_r_homo)

# draw the epolar lines
img_l = stereo.drawlines(img1,img2,lines_l,test_l,test_r)
img_r = stereo.drawlines(img2,img1,lines_r,test_r,test_l)

cv2.imshow('Campus_left', img_l)
cv2.imshow('Campus_right', img_r)
cv2.waitKey(0)