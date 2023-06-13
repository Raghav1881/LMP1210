# %%
import numpy as np

# %%
def run_linear_regression():
  """
  This function sets up a toy linear regression problem and then
  solves it analytically and with gradient descent, and prints
  both of their solutions.
  The two solutions should be quite similar.
  """

  N = 100 # number of data points
  D = 2 # dimension of each point
  # randomly sample data from gaussian
  X = np.random.normal(loc=0.,scale=2.0,size=(N,D)) 
  # generate targets
  true_w = np.random.uniform(low=-1.,high=1.,size=(2,))
  y = np.sum(true_w.reshape(1,-1)*X,axis=1)

  print(f"true weights: {true_w}")
  print(f"analytic optimal weights: {analytic_solution(X,y)}")
  print(f"gradient descent weights: {gradient_descent_solution(X,y)}")


def gradient_descent_solution(X,y):
  """
  arguments:
    X: (N,D) numpy float array of input data
    y: (N,) numpy float array of targets
  returns:
    w: (D,) numpy float array of optimal weights
  """

  # initialize weights and bias randomly for gradient descent
  D = X.shape[1]
  w = np.random.normal(loc=0.,scale=0.1,size=(D,))
  b = np.random.normal(loc=0.,scale=0.1,size=1)
  alpha = 0.01 # learning rate
  num_iters = 100
  for t in range(num_iters):
    w = gradient_descent_update(X,y,w,alpha)
  return w

def compute_gradients(X,y,w):
  """
  Implement this function. Using a for loop (from i=1 to N) is allowed.
  Challenge (not for credit): can you think of a vectorized version?
  arguments:
    X: (N,D) numpy float array of input data
    y: (N,) numpy float array of targets
    w: (D,) numpy float array of initial weights
  returns:
    dL_dw: (D,) numpy float array of weight gradients
  """

  N, D = X.shape
  dL_dw = 2/N * X.T.dot(X.dot(w) - y)

  return dL_dw

def gradient_descent_update(X,y,w,alpha):
  """
  Implement this function (using compute_gradients).
  arguments:
    X: (N,D) numpy float array of input data
    y: (N,) numpy float array of targets
    w: (D,) numpy float array of initial weights
    alpha: float, learning rate
  returns:
    w: (D,) numpy float array of updated weights
  """

  dL_dw = compute_gradients(X,y,w)
  w -= alpha * dL_dw
  
  return w

def analytic_solution(X,y):
  """
  Implement this function (you will need numpy.linalg).
  arguments:
    X: (N,D) numpy float array of input data
    y: (N,) numpy float array of targets
  returns:
    w: (D,) numpy float array of optimal weights
  """

  w = np.linalg.inv(X.T @ X) @ X.T @ y

  return w

# %%
run_linear_regression()

# %%



