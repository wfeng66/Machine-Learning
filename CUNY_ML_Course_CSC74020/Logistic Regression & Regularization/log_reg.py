'''
create a logistic regression from scratch
including feature transform, PCA
apply Breast cancer data set 
'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class log_reg():
    def __init__(self, lr=0.01, epoch=100000, batch_size=4, fold=5, init='random') -> None:
        self.init = init
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.folding = fold
        self.loss_in = []
        self.loss_out = []
        
    def fold(self):
        # self.X_val, self.Y_val= self.X[400:], self.Y[400:]
        # self.X, self.Y = self.X[:400], self.Y[:400]
        self.X_val, self.Y_val= self.X[50:], self.Y[50:]
        self.X, self.Y = self.X[:50], self.Y[:50]
    
    def init_w(self):
        if self.init == 'zero':
            self.W = np.zeros((self.n, ))
        elif self.init == 'random':
            self.W = np.random.normal(0, 0.01, (self.n, ))
        else:
            print('The initilizer only accept "zero" or "random"!')
            return  
    
    def sigmoid(self, z):
        return 1.0/(1+np.exp(-z))
    
    def z(self, x):
        # zz = np.dot(x, self.W)
        # print("z:", x.shape, self.W.shape, zz.shape)
        return np.dot(x, self.W)
    
    def gradient(self, X, Y):
        numerator = np.dot(Y, X)
        # print('W: ', self.W.shape)
        # print('X.T: ', X.T.shape)
        # print('Y: ', Y.shape)
        #tmp = np.dot(X.T, self.W)
        #e_exp = np.dot(Y, np.dot(X, self.W))
        #return -np.sum(numerator/(1+np.exp(e_exp)))/X.shape[0]
        prev_y = self.sigmoid(self.z(X))
        # print('prev:', prev_y.shape, 'Y:', Y.shape, 'X.T:', X.T.shape)
        return (1/self.m)*np.dot(X.T, self.sigmoid(self.z(X))-Y)+2*self.wd*self.W
    
    def update(self, X, Y):
        # delata = self.gradient(X, Y)
        # print('W:', self.W.shape, 'delata:', delata.shape, delata)
        self.W = self.W - self.lr*self.gradient(X, Y)

        
    def loss(self, x, y):
        # print(self.W.T.shape)
        # print(x.shape)
        # l = np.sum(np.log(1+np.exp(-y.dot(np.dot(x, self.W)))))/x.shape[0]
        # print(x.shape, y.shape)
        # z = self.z(x)
        # print(z[:3])
        # prev_y = self.sigmoid(self.z(x))
        # print(prev_y[:5])
        # print(y[:5])
        l = -(1/x.shape[0])*np.sum(y*np.log(self.sigmoid(self.z(x)))+\
            (1-y)*np.log(1-self.sigmoid(self.z(x)))) + self.wd*np.dot(self.W, self.W.T)
        # print(l)
        return l
    
    def fit(self, X, Y, wd=0.0): 
        """Train the model

        Args:
            X (np.array): training data set
            Y (np.array): label
            wd (float): the weight decay supermeter - lambda. Defaults to 0.

        Returns:
            weights(np.array): the final weights
            loss_in(list): the list of history loss in sample
            loss_out(list): the list of history loss out of sample
        """
        self.wd = wd
        self.X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
        self.Y = Y
        self.fold()
        self.m, self.n = self.X.shape
        self.init_w()
        for _ in range(self.epoch):
            # if _ == 1000:
            #     self.lr = self.lr/10
            for i in range((self.m-1)//self.batch_size+1):
                xb = self.X[i*self.batch_size:(i+1)*self.batch_size]
                yb = self.Y[i*self.batch_size:(i+1)*self.batch_size]
                self.update(xb, yb)
            # print('in: ', end=' ')
            l_in = self.loss(self.X, self.Y)
            # print('out: ', end=' ')
            l_out = self.loss(self.X_val, self.Y_val)
            self.loss_in.append(l_in)
            self.loss_out.append(l_out)
            print('In: {}, Out: {}...'.format(l_in, l_out))
        return self.W, self.loss_in, self.loss_out
    

def load_bc():
    import sklearn.datasets as ds 
    bc = ds.load_breast_cancer()
    X_bc = bc.data
    y_bc = bc.target
    # y_bc[y_bc==0] = -1                # convert 0 to -1 in target
    return X_bc, y_bc

import pandas as pd
# df = pd.read_csv("G://temp/marks.txt", header=None)
# X_bc = np.array(df.iloc[:, :-1])
# y_bc = np.array(df.iloc[:, -1])
X_bc, y_bc = load_bc()              # load data

def pca_tran(X, n):
    pca = PCA(n_components=n)
    pcaComponets = pca.fit_transform(X)
    pca_arr = np.array(pcaComponets)
    return pca_arr
    
    

# np.random.seed(12)
# num_observations = 5000
# x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
# x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
# simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
# simulated_labels = np.hstack((np.zeros(num_observations),
#                               np.ones(num_observations)))



lr = log_reg()
# X_bc = pca_tran(X_bc, 5)
scaler = StandardScaler() 
X_bc = scaler.fit_transform(X_bc)
W, loss_in, loss_out = lr.fit(X_bc, y_bc, 0.00001)



# import matplotlib.pyplot as plt
# x = range(0, 1001, 1)
# plt.plot(x, loss_in, label='loss_in')
# plt.plot(x, loss_out, label='loss_out')
# plt.legend(['train', 'val'])
# plt.show()

            

    

        
    
    
    

