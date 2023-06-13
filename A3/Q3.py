# %% [markdown]
# # Adaboost Implementation
# 
# In this question, you will finish an implementation of the Adaboost algorithm (using decision stumps as the weak classifier). For your convenience, I have re-written the algorithm steps below:

# %% [markdown]
# **Input**: Data $x^{(n)}$ ($N$ points, $D$ dimensions) and $y^{(n)}$ (N points, binary labels), weak classifier training procedure `weak_learn`, number of training iterations $T$
# 
# **Output**: $H(x)$, an ensemble of weak classifiers $h_t(x), t = 1, ..., T$ 
# 
# Note: for binary classification, $h_t(x)$ outputs a value in $\{0,1\}$
# 
# 1. Initialize sample weights $w^{(n)} = \frac{1}{N}, n = 1, ..., N$, initialize empty ensemble
# 
# Repeat steps 2-5 $T$ times:
# 
# 2. Fit a weak classifier $h_t$ to the weighted data $(x^{(n)},y^{(n)},w^{(n)})$, using `weak_learn`
# 
# 3. Compute weighted classification error $$e_t \leftarrow \frac{\sum_{n=1}^N w^{(n)} \mathbb{I} \{ h(x^{(n)}) \neq y^{(n)} \}}{\sum_{n=1}^N w^{(n)}}$$ where $\mathbb{I} \{ h(x^{(n)}) \neq x^{(n)} \}$ is an expression that means "return 1 if the prediction $h_t(x^{(n)})$ does not equal the true label $y^{(n)}$, otherwise return 0"
# 
# 4. Compute classifier coefficient $$\alpha_t \leftarrow \frac{1}{2} \log \frac{1-e_t}{e_t}$$ Note:  if implemented correctly $\alpha_t > 0$, the logarithm is base $e$
# 
# 5. Update sample weights $$w^{(n)} \leftarrow w^{(n)} \exp(2 \alpha_t \mathbb{I} \{ h(x^{(n)}) \neq t^{(n)} \})$$
# 
# 6. Return the ensemble $$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$$

# %% [markdown]
# In class, we described this algorithm in lecture 5, slide 34.

# %% [markdown]
# Fill in the code below (indicated with the TODO comments). Do not change the functions that have DO NOT MODIFY.

# %%
import numpy as np
import sklearn
import sklearn.tree
import sklearn.datasets
import math
# random seed
np.random.seed(1211)

# %%
def weak_learn(X,y,w):
  """ 
  DO NOT MODIFY
  train a weak classifier (decision tree with depth of 1)
  """
  
  dt = sklearn.tree.DecisionTreeClassifier(max_depth=1)
  dt.fit(X,y,w)
  return dt

def prepare_data():
  """ 
  DO NOT MODIFY
  generate data and randomly split 80/20 train/test
  """

  X, y = sklearn.datasets.make_classification(n_samples=500, n_features=20, n_informative=5)
  print(f"Number of datapoints = {X.shape[0]}")
  print(f"Number of features = {X.shape[1]}")
  X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.2)
  return X_train, y_train, X_test, y_test

def ensemble_predict(X,ensemble_classifiers,ensemble_alphas):
  """ 
  DO NOT MODIFY
  get predictions for an ensemble
  """

  ensemble_predictions = []
  for i in range(len(ensemble_classifiers)):
    classifier = ensemble_classifiers[i]
    alpha = ensemble_alphas[i]
    predictions = classifier.predict(X)
    ensemble_predictions.append(alpha*predictions)
  ensemble_predictions = np.sum(np.stack(ensemble_predictions,axis=0),axis=0)
  return ensemble_predictions > 0.5

def ensemble_score(X,y,ensemble_classifiers,ensemble_alphas):
  """ 
  DO NOT MODIFY
  compute accuracy for an ensemble's predictions
  """

  ensemble_predictions = ensemble_predict(X,ensemble_classifiers,ensemble_alphas)
  ensemble_accuracy = np.mean(ensemble_predictions==y)
  return ensemble_accuracy


def run_adaboost(X_train,y_train,num_iters):
  """
  create an ensemble of weighted weak classifiers using the adaboost algorithm
  """

  # data dimensions 
  # (N is number of training points, D is number of input components)
  N = X_train.shape[0]
  D = X_train.shape[1]
  # TODO initialize weights
  weights = np.ones(N) / N

  ensemble_classifiers = []
  ensemble_alphas = []
  
  for i in range(num_iters):
    # train a weak classifier
    classifier = weak_learn(X_train, y_train, weights)

    # get classifier predictions
    predictions = classifier.predict(X_train)

    # compute weighted error
    weighted_err = np.sum(weights * (predictions != y_train)) / np.sum(weights)

    # compute classifier coefficient
    alpha = 0.5 * np.log((1 - weighted_err) / weighted_err)
    assert alpha >= 0., alpha

    # add classifier to ensemble
    ensemble_classifiers.append(classifier)
    ensemble_alphas.append(alpha)

    # update weights
    weights *= np.exp(2 * alpha * (predictions != y_train))
        
  return ensemble_classifiers, ensemble_alphas

# %% [markdown]
# You can test out your implementation here. If you did it correctly, you should get a training accuracy of about 0.64 and a test accuracy of about 0.74

# %%
X_train, y_train, X_test, y_test = prepare_data()

# %%
num_iters = 5
ensemble_classifiers, ensemble_alphas = run_adaboost(X_train,y_train,num_iters)
# if you want to test your algorithm, best to restart and run all cells
print(ensemble_score(X_train,y_train,ensemble_classifiers,ensemble_alphas)) # should be around 0.64
print(ensemble_score(X_test,y_test,ensemble_classifiers,ensemble_alphas)) # should be around 0.74

# %%



