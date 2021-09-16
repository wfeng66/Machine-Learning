import numpy as np
import time
import matplotlib.pyplot as plt

iter_n = 10             # the number of iteration
ncols = 10              # the number of columns in matrix, corresponding to the number of features in a data set
np.random.seed(2021)

# create a matrix with random integers
def createX(m, n):
  # m is the number of rows in the matrix, corresponding to the number of instances
  # n is the number of columns in the matrix, corresponding to the number of features
  # a mxn matrix would be return
  return np.random.randint(100, size=(m, n))


# calculate the Euclidean Distance Matrix by nested loop
def ED_loop(A, B):
  m1 = A.shape[0]   # the number of rows of matrix A
  m2 = B.shape[0]   # the number of rows of matrix B
  n1 = A.shape[1]   # the number of columns of matrix A
  n2 = B.shape[1]   # the number of columns of matrix B
  if n1 != n2:
    raise Exception("The numbers of columns between two inputed matrices doen't match!")
  Z = np.zeros((m1, m2))

  for i in range(m1):
    for j in range(m2):
      Z[i, j] = np.sqrt(sum([(A[i, k] - A[j, k])**2 for k in range(n1)]))

  return Z


# calculate the Euclidean Distance Matrix by vectorization operations
def ED_vec(A, B):
  # A, B is the two matrices to be calculated the distance between them
  # return the Euclidean Distance Matrix
  p1 = np.sum(A**2, axis=1)[:, np.newaxis]
  p2 = np.sum(B**2, axis=1)
  p3 = -2*np.dot(A, B.T)
  return np.sqrt(p1+p2+p3)


def corr_loop(X):
  n = X.shape[1]              # the number of columns in matrix X
  S = np.zeros((n, n))        # initialize covariation matrix S
  # calculate columne mean of X
  u = np.zeros((1, n))
  for k in range(n):
    u[0, k] = np.mean(X[:, k])
  # calculate the covariation matrix and standard deviation
  sd = np.zeros((1, n))       # initialize the standard deviation matrix
  for i in range(n):
    for j in range(n):
      for m in range(X.shape[0]):
        S[i,j] += (X[m, i] - u[0, i])*(X[m, j] - u[0, j])
      S[i,j] /= (n-1)
      if i == j:                       # on the diagonal of the matrix
        sd[0,i] = np.sqrt(S[i, j])     # calculate the standard deviations
  # initialize correlation matrix R
  R = np.zeros((n, n))
  # calculate the correlation matrix
  for i in range(n):
    for j in range(n):
      R[i,j] = S[i,j]/(sd[0, i]*sd[0, j])
  return R


def corr_vec(X):
  # this function calculate the correlation matrix by matrix operations
  # instead of nesting loops
  n = X.shape[1]              # the number of columns in matrix X
  u = np.mean(X, axis=0)      # u is the column means
  dev = X-u                   # dev is the deviations, xi-column_mean
  S = np.dot(dev.transpose(), dev)   # S is the covariance matrix
  sd = np.sqrt(np.diagonal(S))       # sd is the standard diviation matrix
  sd = np.reshape(sd, (-1, sd.shape[0]))
  R = (S/sd)/np.transpose(sd)        # R is the correlation matrix 
  return R



if __name__ == "__main__":
    rows = range(10, 501, 50)
    nrows = len(rows)
    # define 2d list to record the performances
    perf_ed_loop = [[]]         # for the euclidean distance in nested loop approach
    perf_ed_vec = [[]]          # for the euclidean distance in matrices operation approach
    perf_corr_loop = [[]]       # for the correlation matrix in nested loop approach
    perf_corr_vec = [[]]        # for the correlation matrix in matrices operation approach

    # run 2 approaches in various sizes of matrix iter_n times
    for m in rows:
        p_ed_loop = []
        p_ed_vec = []
        p_corr_loop = []
        p_corr_vec = []
        X = createX(m, ncols)
        for i in range(iter_n):
            
            begin_t = time.time()
            Z_loop = ED_loop(X, X)
            end_t = time.time()
            p_ed_loop.append(end_t-begin_t)

            begin_t = time.time()
            Z_vec = ED_vec(X, X)
            end_t = time.time()
            p_ed_vec.append(end_t-begin_t)  
            
            assert np.allclose(Z_loop, Z_vec, atol=1e-06) # check the identity of two matrices produced in two ways
            
            begin_t = time.time()
            R_loop = corr_loop(X)
            end_t = time.time()
            p_corr_loop.append(end_t-begin_t)

            begin_t = time.time()
            R_vec = corr_vec(X)
            end_t = time.time()
            p_corr_vec.append(end_t-begin_t)  
            
            assert np.allclose(R_loop, R_vec, atol=1e-06) # check the identity of two matrices produced in two ways

        perf_ed_loop.append(p_ed_loop)
        perf_ed_vec.append(p_ed_vec) 
        perf_corr_loop.append(p_corr_loop)
        perf_corr_vec.append(p_corr_vec) 

    # convert the 2d list to np.array
    perf_ed_loop.pop(0)    # get rid of the null list at the top
    perf_ed_vec.pop(0)
    perf_corr_loop.pop(0)    
    perf_corr_vec.pop(0)    
    perf_ed_loop = np.array(perf_ed_loop)
    perf_ed_vec = np.array(perf_ed_vec)
    perf_corr_loop = np.array(perf_corr_loop)
    perf_corr_vec = np.array(perf_corr_vec)

    # next 14 line codes refer to ComputeMatrices.py by Dr. Chia-Ling Tsai
    u_ed_loop = np.mean(perf_ed_loop, axis = 1)
    u_ed_vec = np.mean(perf_ed_vec, axis = 1)
    u_corr_loop = np.mean(perf_corr_loop, axis = 1)
    u_corr_vec = np.mean(perf_corr_vec, axis = 1)
    std_ed_loop = np.std(perf_ed_loop, axis = 1)
    std_ed_vec = np.std(perf_ed_vec, axis = 1)
    std_corr_loop = np.std(perf_corr_loop, axis = 1)
    std_corr_vec = np.std(perf_corr_vec, axis = 1)

    plt.figure(1)
    plt.errorbar(rows, u_ed_loop, yerr=std_ed_loop, color='red',label = 'Loop Solution for Distance Comp')
    plt.errorbar(rows, u_ed_vec, yerr=std_ed_vec, color='blue', label = 'Matrix Solution for Distance Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Distance Computation Methods')
    plt.legend()
    plt.savefig('CompareDistanceCompFig.pdf')
    plt.show()    # uncomment this if you want to see it right way
    print("result is written to CompareDistanceCompFig.pdf")

    plt.figure(2)
    plt.errorbar(rows, u_corr_loop, yerr=std_corr_loop, color='red',label = 'Loop Solution for Correlation Comp')
    plt.errorbar(rows, u_corr_vec, yerr=std_corr_vec, color='blue', label = 'Matrix Solution for Correlation Comp')
    plt.xlabel('Number of Rows of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Correlation Computation Methods')
    plt.legend()
    plt.savefig('CompareCorrelationCompFig.pdf')
    plt.show()    # uncomment this if you want to see it right way
    print("result is written to CompareCorrelationCompFig.pdf")