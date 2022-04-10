import cv2
import numpy as np
from matplotlib import pyplot as plt
import stereo

'''
path='G://CUNY/CV/Assignments/HW3/'
im1='pic410.png'
im2='pic430.png'
img1 = cv2.imread(path+im1)
img2 = cv2.imread(path+im2)
'''

gF = stereo.getF()
F = gF.findF()
# F, mask = cv2.findFundamentalMat(gF.plo,gF.pro,cv2.FM_LMEDS)
"""
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# lines = gF.epLine(gF.plo[i, :])
lines = cv2.computeCorrespondEpilines(gF.plo.reshape(-1,1,2), 2,F)
lines = lines.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines,gF.plo,gF.pro)
plt.subplot(121),plt.imshow(img5)
plt.show()
"""

print('The fundamental matrix is: ')
print(F)