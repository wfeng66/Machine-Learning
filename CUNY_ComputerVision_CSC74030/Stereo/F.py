import cv2
import numpy as np
from matplotlib import pyplot as plt
import stereo

gF = stereo.getF()
F = gF.findF()
# F, mask = cv2.findFundamentalMat(gF.plo,gF.pro,cv2.FM_LMEDS)

print('The fundamental matrix is: ')
print(F)