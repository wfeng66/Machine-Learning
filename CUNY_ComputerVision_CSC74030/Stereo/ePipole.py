import cv2
import numpy as np
import stereo



def ePipole(F):
    U, D, V = np.linalg.svd(F)
    return V[-1, :], U[:, -1]


img1, img2, ctrl_l, ctrl_r, test_l, test_r = stereo.load()

F, mask = cv2.findFundamentalMat(ctrl_l,ctrl_r,cv2.FM_LMEDS)
# gF = stereo.getF()
# F = gF.findF()

epl, epr = ePipole(F)
epl = epl/epl[-1]
epr = epr/epr[-1]
print(epl, epr)





