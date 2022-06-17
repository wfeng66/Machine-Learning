'''
create a logistic regression from scratch
including feature transform, PCA
apply Breast cancer data set 
'''

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class log_reg():
    def __init__(self, lr=0.0003, epoch=3000, batch_size=4, fold=5, init='random') -> None:
        self.init = init                # initialization methods
        self.X_train, self.X_val, self.Y_train, self.Y_val = [], [], [], []     
        self.epoch = epoch              # training epochs
        self.lr = lr                    # learning rate
        self.batch_size = batch_size    # batch size   
        self.loss_in = []               # store loss history in sample
        self.loss_out = []              # store loss history out of sample, validation loss
        self.v = fold                   # control how many folds
        
    def fold(self):
        '''
        create n-folding data set
        the number of folding decided by the self.v which can be set at initialization of the instance
        no input 
        return the 4 lists: X_train, X_val, Y_train, Y_val
        each list includes self.v entries, default 5 entries which are numpy arrays
        '''
        X_train, X_val, Y_train, Y_val = [], [], [], []
        size = self.X.shape[0]//self.v      # claculate the size of the validation data set
        for i in range(self.v):             # use loop to move the position of validation set
            # choose different part of data set as validation set, combines other as training
            X_val.append(self.X[i*size:(i+1)*size])
            X_train.append(np.concatenate((self.X[:i*size], self.X[(i+1)*size:])))
            Y_val.append(self.Y[i*size:(i+1)*size])
            Y_train.append(np.concatenate((self.Y[:i*size], self.Y[(i+1)*size:])))
        # set the size of training and validation data set
        self.X_train_m, self.X_val_m = X_train[0].shape[0], X_val[0].shape[0]       
        return X_train, X_val, Y_train, Y_val

    
    def init_w(self):
        '''
        initialize the weights
        no input, no return
        the initializing method is decided by self.init, which can be set at initialization of the instance
        the default of initializing method is random
        the initialized weights are stored in self.W
        '''
        if self.init == 'zero':
            self.W = np.zeros((self.n, ))
        elif self.init == 'random':
            self.W = np.random.normal(0, 0.01, (self.X.shape[1], ))
        else:
            print('The initilizer only accept "zero" or "random"!')
            return  
    
    def sigmoid(self, z):
        '''
        sigmoid function
        input:  z - the only one independent variable
        output: the value of sigmoid function
        '''
        return 1.0/(1+np.exp(-z))
       
    def z(self, x):
        '''
        linear regression, matrix product
        input:  x - inpute feature matrix
        output: the result of dot product
        '''
        return np.dot(x, self.W)
    
    def gradient(self, X, Y):
        '''
        gradient function
        including 2 versions: uniform regularizer and low-order regularizer
        input:  X - feature matrix of training data
                Y - label matrix of training data 
        output: the gradient matrix whose shape is same as that of weights 
        uniform: sum(w_q^2), where q is the order of power
        low-order: sum(q*w_q^2), where q is the order of power
        '''
        # prev_y = self.sigmoid(self.z(X))
        # Uniform regularizer
        # online formular version
        # return (1/X.shape[0])*np.dot(X.T, self.sigmoid(self.z(X))-Y) + (2/X.shape[0])*self.wd*self.W
        # textbook formular version
        # return (-1/X.shape[0])*np.sum(((Y.reshape(-1, 1)*X)/(1 + np.exp(Y*self.z(X))).reshape((-1,1))) , axis=0) + (2/X.shape[0])*self.wd*self.W
        # other version
        # return -(1/self.m)*np.dot(X.T, prev_y*(1-prev_y)*(Y-prev_y))
        # Low-order regularizer
        q = np.arange(1, self.W.shape[0]+1)         # create a q array corresponding to the order of power
        decay = np.multiply(q, self.W)
        # online formular version
        return (1/X.shape[0])*np.dot(X.T, self.sigmoid(self.z(X))-Y)+2*self.wd*decay
        # textbook formular version
        # return (-1/X.shape[0])*np.sum(((Y.reshape(-1, 1)*X)/(1 + np.exp(Y*self.z(X))).reshape((-1,1))) , axis=0) +\
        #        (2/X.shape[0])*self.wd*decay
    
    
    def update(self, X, Y):
        '''
        update the weights
        W_new = W_old - LearningRate*gradient
        input:  X - feature matrix of training data
                Y - label matrix of training data 
        return: no return, the result is stored to self.W
        '''
        self.W = self.W - self.lr*self.gradient(X, Y)

        
    def loss(self, x, y):
        '''
        loss function, calculate the loss value
        '''
        # Uniform regularizer
        # online formular version
        # l = -(1/x.shape[0])*np.sum(y*np.log(self.sigmoid(self.z(x)))+\
        #     (1-y)*np.log(1-self.sigmoid(self.z(x)))) + (1/x.shape[0])*self.wd*np.dot(self.W, self.W.T)
        # textbook formular version
        # l = (1/x.shape[0])*np.sum(np.log(1 + np.exp(-y*self.z(x))) , axis=0) + (1/x.shape[0])*self.wd*np.dot(self.W, self.W.T)
        # Low-order regularizer
        q = np.arange(1, self.W.shape[0]+1)         # create a q array corresponding to the order of power
        decay = np.dot(self.W.T, np.multiply(q, self.W))
        # online formular version
        l = -(1/x.shape[0])*np.sum(y*np.log(self.sigmoid(self.z(x)))+\
             (1-y)*np.log(1-self.sigmoid(self.z(x)))) + self.wd*decay
        # textbook formular version
        # l = (1/x.shape[0])*np.sum(np.log(1 + np.exp(-y*self.z(x))) , axis=0) + \
        #     (1/x.shape[0])*self.wd*decay
        return l
    
    def fit(self, X, Y, wd=0.01): 
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
        # split data sets
        self.X_test, self.Y_test = self.X[500:], self.Y[500:]
        self.X, self.Y = self.X[:500], self.Y[:500]
        X_train, X_val, Y_train, Y_val = self.fold()
        for v in range(self.v):             # iterate folding
            self.init_w()                   # initialize weights
            lst_in, lst_out = [], []        # store loss in and loss out
            # get training and validation data in specific folding
            x_tr_v, y_tr_v, x_val_v, y_val_v = X_train[v], Y_train[v], X_val[v], Y_val[v]   
            for ep in range(self.epoch):    # iterate epochs
                # variant learning rate
                if ep == 10000:
                    self.lr = self.lr/10
                elif ep == 50000:
                    self.lr = self.lr/10
                # train in batches
                for i in range((self.X_train_m-1)//self.batch_size+1):
                    xb = x_tr_v[i*self.batch_size:(i+1)*self.batch_size]
                    yb = y_tr_v[i*self.batch_size:(i+1)*self.batch_size]
                    self.update(xb, yb)
                # calculate the loss in and loss out
                l_in = self.loss(x_tr_v, y_tr_v)
                l_out = self.loss(x_val_v, y_val_v)
                if ep%100 == 0:         # store the loss values each 100 epochs
                    lst_in.append(l_in)
                    lst_out.append(l_out)
                    # output in training
                    print('wd:', wd, 'v:{}, In: {}, Out: {}...'.format(v, l_in, l_out))
            # backup the loss histories after each folding training
            self.loss_in.append(lst_in)
            self.loss_out.append(lst_out)
        '''
        After all, the self.loss_in and self.loss_out include 5 lists respectively, 
        each one include loss history in one folding.
        '''
        return self.W, self.loss_in, self.loss_out
    

def load_bc():
    '''
    load breast cancer data set
    input: no
    return: X_bc - feature matrix
            y_bc - label matrix
    '''
    import sklearn.datasets as ds 
    bc = ds.load_breast_cancer()
    X_bc = bc.data
    y_bc = bc.target
    return X_bc, y_bc

import pandas as pd
X_bc, y_bc = load_bc()              # load data

def pca_tran(X, n):
    '''
    PCA transformation, dimensions reduction
    input:  X - feature matrix
            n - the n_components parameter
    return  the transferred feature matrix
    '''
    pca = PCA(n_components=n)
    pcaComponets = pca.fit_transform(X)
    pca_arr = np.array(pcaComponets)
    return pca_arr
    


l_in, l_out = [], []
# iterate weight decays
# create lambda values need to try
d0 = np.arange(1e-10, 1e-8, 1e-10)
d1 = np.arange(1e-8, 1e-6, 1e-8)
d2 = np.arange(1e-6, 1e-5, 1e-6)
decays = list(d0) + list(d1) + list(d2) 
X_bc = pca_tran(X_bc, 5)                # PCA
# Polynomial transformation
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=10)
X_bc = poly.fit_transform(X_bc)
# Standard scaler
scaler = StandardScaler() 
X_bc = scaler.fit_transform(X_bc)
for d in decays:                        # iterate lambda values
    lr = log_reg()                      # instantiate a logistic regression
    W, loss_in, loss_out = lr.fit(X_bc, y_bc, d)        # train the model with specific lambda value
    # only keep the final loss value in specific lambda
    l_in.append(np.mean(loss_in, axis=0)[-1])           
    l_out.append(np.mean(loss_out, axis=0)[-1])


# visualization of the loss in and loss out with respect to various lambdas of regularization
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(decays, l_in, 'b', label='train loss')
plt.plot(decays, l_out, 'r', label='val loss')
plt.xlabel('Regularization Parameter')
plt.ylabel('E_out')
plt.title('Low-order Regularizer')
plt.legend()
plt.show()



            

    

        
    
    
    

