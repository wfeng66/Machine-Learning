import random
import numpy as np
import matplotlib.pyplot as plt

class calib:
    def __init__(self, f=0.016, ox=256, oy=256, sx=0.0088 / 512, sy=0.0066 / 512, alpha=-120, beta=0, gamma=40, T=[0, 0, 5]):
        # set the intrinsic and extrinsic parameters
        self.f = f
        self.ox = ox
        self.oy = oy
        self.sx = sx
        self.sy = sy
        self.fx, self.fy = f/self.sx, f/self.sy
        self.alpha = alpha*np.pi/180
        self.beta = beta*np.pi/180
        self.gamma = gamma*np.pi/180
        self.T = np.array(T).reshape((3,1))
        # 3d coordinate of vertices on the cube in homogeneous format
        self.A = np.array([1, 0, 1, 1])
        self.B = np.array([0, 0, 1, 1])
        self.C = np.array([0, 1, 1, 1])
        self.D = np.array([1, 1, 1, 1])
        self.E = np.array([1, 0, 0, 1])
        self.F = np.array([0, 0, 0, 1])
        self.G = np.array([0, 1, 0, 1])
        self.H = np.array([1, 1, 0, 1])
        # rotation matrix
        self.Rr = np.array([
            [np.cos(self.gamma), -np.sin(self.gamma), 0],
            [np.sin(self.gamma), np.cos(self.gamma), 0],
            [0, 0, 1]
        ])
        self.Rb = np.array([
            [np.cos(self.beta), 0, -np.sin(self.beta)],
            [0, 1, 0],
            [np.sin(self.beta), 0, np.cos(self.beta)]
        ])
        self.Ra = np.array([
            [1, 0, 0],
            [0, np.cos(self.alpha), -np.sin(self.alpha)],
            [0, np.sin(self.alpha), np.cos(self.alpha)]
        ])
        self.R = self.Ra@self.Rb@self.Rr

        # initialize the world 3d coordinates of the simulated points
        a = np.arange(0.2, 1.0, 0.2)
        p1 = np.append(np.full(((4,1)), 1), a.reshape((4,1)), 1)
        p2 = np.append(a.reshape((4,1)),np.full(((4,1)), 1), 1)
        self.Pw = np.empty((0,3), float)
        for z in np.arange(0.2, 1.0, 0.2):                  # generate world coordinate for marked points
            self.Pw = np.concatenate((self.Pw, np.append(p1, np.full((4,1), z), 1)), axis=0)
            self.Pw = np.concatenate((self.Pw, np.append(p2, np.full((4,1), z), 1)), axis=0)


    def showSetting(self):
        print('focal length: ', self.f)
        print('image center: ', self.ox, ',', self.oy)
        print('pixel size: ', self.sx, ',', self.sy)
        print('rotation angles: ')
        print('alpha: ', self.alpha * 180 / np.pi)
        print('beta: ', self.beta * 180 / np.pi)
        print('gamma: ', self.gamma * 180 / np.pi)
        print('Rotation Matrix: ', self.R)


    def proj(self):
        # this method is used to project the world 3d coordinate to the image plane 2d coordinate
        # parameters: no
        # return: no, all projected coordinates are stored in object self properties
        # extrinsic matrix
        self.Mext = np.append(self.R, self.T, 1)
        self.Mint = np.array([
            [-self.fx, 0, self.ox],
            [0, -self.fy, self.oy],
            [0, 0, 1]
        ])
        self.M = self.Mint@self.Mext
        # project marked points' world coordinate to image frame coordinate
        self.Pwh = np.append(self.Pw, np.full((self.Pw.shape[0], 1), 1), 1)
        self.Pim = self.M@self.Pwh.T
        self.Pim = self.Pim.T
        self.Pim[:, 0] = self.Pim[:, 0]/self.Pim[:, 2]
        self.Pim[:, 1] = self.Pim[:, 1]/self.Pim[:, 2]
        self.Pim = self.Pim[:, :2]
        # project world coordinate of vertices on the cube to frame coordinate
        self.Aim, self.Bim, self.Cim, self.Dim = self.M @ self.A, self.M @ self.B, self.M @ self.C, self.M @ self.D
        self.Eim, self.Fim, self.Gim, self.Him = self.M @ self.E, self.M @ self.F, self.M @ self.G, self.M @ self.H
        self.Aim[0], self.Aim[1] = self.Aim[0] / self.Aim[2], self.Aim[1] / self.Aim[2]
        self.Bim[0], self.Bim[1] = self.Bim[0] / self.Bim[2], self.Bim[1] / self.Bim[2]
        self.Cim[0], self.Cim[1] = self.Cim[0] / self.Cim[2], self.Cim[1] / self.Cim[2]
        self.Dim[0], self.Dim[1] = self.Dim[0] / self.Dim[2], self.Dim[1] / self.Dim[2]
        self.Eim[0], self.Eim[1] = self.Eim[0] / self.Eim[2], self.Eim[1] / self.Eim[2]
        self.Fim[0], self.Fim[1] = self.Fim[0] / self.Fim[2], self.Fim[1] / self.Fim[2]
        self.Gim[0], self.Gim[1] = self.Gim[0] / self.Gim[2], self.Gim[1] / self.Gim[2]
        self.Him[0], self.Him[1] = self.Him[0] / self.Him[2], self.Him[1] / self.Him[2]


    def visual(self):
        # this method is used to visualize the marked points and cube
        # parameters: no
        # return: no, output the chart
        # plot the checking points
        plt.scatter(self.Pim[:, 0], self.Pim[:, 1])
        # plot the edge of the cube
        plt.plot([self.Aim[0], self.Bim[0]], [self.Aim[1], self.Bim[1]])
        plt.plot([self.Bim[0], self.Cim[0]], [self.Bim[1], self.Cim[1]])
        plt.plot([self.Cim[0], self.Dim[0]], [self.Cim[1], self.Dim[1]])
        plt.plot([self.Dim[0], self.Aim[0]], [self.Dim[1], self.Aim[1]])
        plt.plot([self.Eim[0], self.Him[0]], [self.Eim[1], self.Him[1]])
        plt.plot([self.Him[0], self.Gim[0]], [self.Him[1], self.Gim[1]])
        plt.plot([self.Aim[0], self.Eim[0]], [self.Aim[1], self.Eim[1]])
        plt.plot([self.Dim[0], self.Him[0]], [self.Dim[1], self.Him[1]])
        plt.plot([self.Cim[0], self.Gim[0]], [self.Cim[1], self.Gim[1]])
        # define the axis on the frame
        plt.xlim(0, 512)
        plt.ylim(0, 512)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.show()


    def calibrate(self):
        # this method is used to calibrate the camera parameters
        # parameters: no
        # return: rotation matrix, aspect ratio, translate vector, fx, fy, rotation angles
        # define A matrix
        A = np.zeros((self.Pim.shape[0], 8))
        A[:, 0] = (self.Pim[:, 0]-self.ox) * self.Pw[:, 0]        # x*X
        A[:, 1] = (self.Pim[:, 0]-self.ox) * self.Pw[:, 1]        # x*Y
        A[:, 2] = (self.Pim[:, 0]-self.ox) * self.Pw[:, 2]        # x*Z
        A[:, 3] = (self.Pim[:, 0]-self.ox)                        # x
        A[:, 4] = -(self.Pim[:, 1]-self.oy) * self.Pw[:, 0]       # -y*X
        A[:, 5] = -(self.Pim[:, 1]-self.oy) * self.Pw[:, 1]       # -y*Y
        A[:, 6] = -(self.Pim[:, 1]-self.oy) * self.Pw[:, 2]       # -y*Z
        A[:, 7] = -(self.Pim[:, 1]-self.oy)                       # -y
        # SVD
        u, s, v = np.linalg.svd(A)
        # v = v.T
        gamma = np.sqrt(v[7, 0]**2 + v[7, 1]**2 + v[7, 2]**2)
        alpha = np.sqrt(v[7, 4]**2 + v[7, 5]**2 + v[7, 6]**2)/gamma

        # find the sign of gamma
        # if self.Pim[0, 0] == abs(self.Pim[0, 0]):       # x is positive
        #     gamma = -gamma
        X_prime = (v[7, 4:7].T@self.Pw[0] + v[7, 7])/(gamma*alpha)
        if X_prime != self.Pim[0, 0]:
            gamma = -gamma
        # calculate R3 by the cross product of R1 and R2
        R1 = v[7, 4:7] / (gamma * alpha)
        R2 = v[7, :3] / gamma
        R3 = np.cross(R1, R2)
        # find the Tx and Ty
        Tx = v[7, -1] / (gamma*alpha)
        Ty = v[7, 3] / gamma
        # enforce orthogonality constraint
        R = np.zeros((3,3))
        R[0, :], R[1, :], R[2, :] = R1, R2, R3
        U, S, V = np.linalg.svd(R)
        I = np.identity(3)
        R = U@I@V
        print('estimating R: ', R)
        print('original R: ', self.R)
        print('the sum of error between estimated R and oringal R: ', np.sum(self.R-R))
        # find Tz, fx and fy
        a2 = R@self.Pw.T                                                # RPw.T
        AA = np.zeros((self.Pw.shape[0], 2))                            # Initiate empty A matrix
        AA[:, 0] = self.Pim[:, 0]-self.ox                               # 1st column of A matrix
        AA[:, 1] = a2[0, :].T + Tx                                      # 2nd column of A matrix
        BB = -(self.Pim[:, 0]-self.ox)*a2[2, :].T
        [Tz, fx] = np.linalg.inv(AA.T@AA)@AA.T@BB
        fy = fx/alpha
        print('estimating T:', [Tx, Ty, Tz], 'original T: ', self.T)
        print('estimating fx: ', fx, 'original fx: ', self.fx)
        print('estimating fy: ', fy, 'original fy: ', self.fy)
        # compute the rotation angles
        beta_R = np.arcsin(-R[0, -1])
        alpha_R = np.arcsin(-R[1,-1]/np.cos(beta_R))
        gamma_R = np.arcsin(-R[0, 1]/np.cos(beta_R))
        print('estimating alpha, beta, gammar: ', alpha_R*180/np.pi, beta_R*180/np.pi, gamma_R*180/np.pi)
        print('original alpha, beta, gammar: ', self.alpha*180/np.pi, self.beta*180/np.pi, self.gamma*180/np.pi)
        return R, alpha, [Tx, Ty, Tz], fx, fy, alpha_R, beta_R, gamma_R


    def locAccAnalysis(self):
        # this method is used to analysis the accuracy sensitive caused by measurement error on 3d and 2d environment
        # parameters: no
        # return: no, all output on the screen
        # create random error
        noise_w = np.array([random.uniform(0.1/1000, 0.2/1000) for _ in range(self.Pw.shape[0]*3)]).reshape(self.Pw.shape)
        noise_im = np.array([random.uniform(6.6/(512*1000), 10/(512*1000)) for _ in range(self.Pw.shape[0]*2)]).reshape(self.Pw.shape[0], 2)
        # calibration
        self.proj()
        self.Pw = self.Pw + noise_w
        self.Pim = self.Pim + noise_im
        R, alpha, [Tx, Ty, Tz], fx, fy, alpha_R, beta_R, gamma_R = c.calibrate()
        # accuracy sensitive analysis
        sum_noise = np.sum(noise_w) + np.sum(noise_im)
        print("Error sensitive: ")
        print('R: ', np.sum(R-self.R)/sum_noise)
        print('alpha: ', (alpha-(self.fx/self.fy))/sum_noise)
        print('Tx: ', (Tx-self.T[0])/sum_noise)
        print('Ty: ', (Ty - self.T[1]) / sum_noise)
        print('Tz: ', (Tz - self.T[2]) / sum_noise)
        print('fx: ', (fx - self.fx) / sum_noise)
        print('fy: ', (fy - self.fy) / sum_noise)


    def orthAccAnalysis(self):
        # this method is used to analysis the accuracy sensitive caused by error of image center evaluation
        # parameters: no
        # return: no, all output on the screen
        # projection
        self.proj()
        # error appear on axis of x
        self.ox -= 0.5
        # calibration
        R, alpha, [Tx, Ty, Tz], fx, fy, alpha_R, beta_R, gamma_R = c.calibrate()
        # output accuracy sensitive
        print("Image center error sensitive (ox): ")
        print('R: ', np.sum(R - self.R) / 0.5)
        print('Tx: ', (Tx - self.T[0]) / 0.5)
        print('Ty: ', (Ty - self.T[1]) / 0.5)
        print('Tz: ', (Tz - self.T[2]) / 0.5)
        # error appear on axis of y
        self.ox += 0.5
        self.oy -= 0.5
        # calibration
        R, alpha, [Tx, Ty, Tz], fx, fy, alpha_R, beta_R, gamma_R = c.calibrate()
        # output accuracy sensitive
        print("Image center error sensitive (oy): ")
        print('R: ', np.sum(R - self.R) / 0.5)
        print('Tx: ', (Tx - self.T[0]) / 0.5)
        print('Ty: ', (Ty - self.T[1]) / 0.5)
        print('Tz: ', (Tz - self.T[2]) / 0.5)



# design and visualization
# c = calib()
# c.proj()
# c.visual()

# calibration
c = calib()
c.proj()
c.calibrate()

# acurracy analysis
# c = calib()
# c.locAccAnalysis()
# c.orthAccAnalysis()






