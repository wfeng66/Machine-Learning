import cv2
import numpy as np




def ePipole(F):
    U, D, V = np.linalg.svd(F)
    return V[-1, :], U[:, -1]


path='G://CUNY/CV/Assignments/HW3/'
lfile = 'ctrl_pl.npy'
rfile = 'ctrl_pr.npy'
im1='pic410.png'
im2='pic430.png'
img1 = cv2.imread(path + im1)
img2 = cv2.imread(path + im2)
pl = np.load(path + lfile)
pr = np.load(path + rfile)
pl = np.int32(pl)
pr = np.int32(pr)

F, mask = cv2.findFundamentalMat(pl,pr,cv2.FM_LMEDS)


epl, epr = ePipole(F)
epl = epl/epl[-1]
epr = epr/epr[-1]
print(epl, epr)





