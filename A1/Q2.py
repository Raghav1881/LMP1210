# %%
# numpy import statement
import numpy as np
np.random.seed(1210)

# %% [markdown]
# **Note**: Each question has some test cases, to help you figure out if your implementation is correct
# 

# %% [markdown]
# ## Algorithm questions

# %% [markdown]
# Implement the following algorithms according to their docstring.
# 
# Please, only use on basic list/dictionary operations, do not use any fancy libraries that solve this problem for you!

# %%
def extract_odd_values(input_list):
  """
  Return a new list which includes odd elements from input_list.
  Arguments:
    input_list: list, a list of elements
  Returns:
    new_list: list, only including odd values of the input_list
  """
  new_list = [x for x in input_list if x % 2 != 0]
  return new_list

# %%
# test cases: extract_odd_values
l1 = [1,3,2,4,5,6]
l1_invert = extract_odd_values(l1)
print(l1) # should be [1,3,2,4,5,6]
print(l1_invert) # should be [1,3,5]
l2 = []
l2_invert = []
print(l2) # should be []
print(l2_invert) # should be []

# %%
def has_duplicate(input_list):
  """
  Return True if input_list has any duplicates, otherwise return False.
  Arguments:
    input_list: list, a list of elements
  Returns:
    duplicate: boolean, whether or not input_list has duplicates
  """
  temp_list = []
  for k in input_list:
    if k not in temp_list:
      temp_list.append(k)
    else:
      return True
  return False

# %%
# test cases for has_duplicate
l1 = ['A','T','C','G']
print(has_duplicate(l1)) # should be False
l2 = ['T','T','C','A']
print(has_duplicate(l2)) # should be True
l3 = ['A','X','Y','A']
print(has_duplicate(l3)) # should be True

# %%
def invert_dict(input_dict):
  """
  Return a new dictionary whose key, value pairs are equal to the 
  value, key pairs of input_dict.
  This only works if input_dict has no duplicates in its values.
  if input_dict has any duplicate values, raise a ValueError.
  Arguments:
    input_dict: dict, a dictionary
  Returns:
    new_dict: dict, a dictionary that is the inverse of input_dict
  """
  temp_list = list(input_dict.values())
  if has_duplicate(temp_list):
    raise ValueError
  else:
    new_dict = {v: k for k, v in input_dict.items()}
    return new_dict

# %%
# test cases for invert_dict
d1 = {"A":0,"C":1,"T":2,"G":3}
d1_invert = invert_dict(d1)
print(d1) # should be {"A":0,"C":1,"T":2,"G":3}
print(d1_invert) # should be {0:"A", 1:"C", 2:"T", 3:"G"}
d2 = {"A":0,"C":0,"T":2,"G":3}
d2_invert = invert_dict(d2) # should raise ValueError
print(d2) # should not be executed
print(d2_invert) # should not be executed

# %%
def mutagenesis(seq,mut_dict):
  """
  Return a new sequence that is identical to seq (a DNA sequence) but is 
  mutated according to mut_dict, a dictionary that maps nucleotide bases 
  ('A','C','T','G') to other (mutated) bases.
  Raise a ValueError if input_string is not a valid DNA sequence (contains
  a character other than ACTG).
  If a base is absent from mut_dict, do NOT mutate it.
  Note: this is mutation in the biological sense, not referring to python
  mutability.
  Arguments:
    seq: str, input sequence of bases
    mut_dict: dict, maps original bases to new bases
  Returns:
    new_seq: str, sequence where the bases have been mutated
  """
  new_seq = ""
  if not all(c in "ACTG" for c in seq):
    raise ValueError

  for base in seq:
    new_base = mut_dict.get(base, base)
    new_seq += new_base
  return new_seq


# %%
# test cases for mutatagenesis
# A becomes T, T becomes A, other bases are unchanged
mut_dict = {
    "A": "T",
    "T": "A",
}
seq1 = "AACTGACTTAGCA"
mut_seq1 = mutagenesis(seq1,mut_dict)
print(seq1) # should be "AACTGACTTAGCA"
print(mut_seq1) # should be "TTCAGTCAATGCT"
seq2 = "ACTGM5"
mut_seq2 = mutagenesis(seq2,mut_dict) # should raise ValueError
print(seq2) # should not be executed
print(mut_seq2) # should not be executed

# %% [markdown]
# ## Numpy Programming Question

# %% [markdown]
# ### Mins and Sums
# 
# In this question, you must implement two related functions. The first function, `min_then_sum`, computes the min over user-specified axes `axes` for two different arrays, then it adds the resulting arrays together (elementwise). The second function, `sum_then_min`, does the same thing but in opposite order: first it computes the sum over user-specified axes `axes`, then it takes the elementwise min of the resulting arrays.

# %%
def min_then_sum(array_1,array_2,axes):
  """
  Take min of array_1 and array_2 over axes, then return the elementwise 
  sum of the resulting arrays.
  Hint: if a, b are arrays with same shape, a+b is elementwise sum of a and b.
  """ 
  assert array_1.shape == array_2.shape
  min_1, min_2 = np.min(array_1, axis = axes), np.min(array_2, axis = axes)
  array_sum = min_1 + min_2
  return array_sum


# %%
# min_then_sum tests
a1 = np.array([1.,2.,3.])
b1 = np.array([1.,1.,1.])
mts1 = min_then_sum(a1,b1,(0,))
print(mts1) # should be 2.0
a2 = np.arange(40).reshape(2,4,5) # 3-dimensional array with shape (2,4,5)
b2 = np.ones([40]).reshape(2,4,5) # 3-dimensional array with shape (2,4,5)
mts2 = min_then_sum(a2,b2,(0,2)) # 1-dimensional array with shape (4,)
print(mts2) # should be [ 1.  6. 11. 16.]

# %%
def sum_then_min(array_1,array_2,axes):
  """
  Take sum of array_1 and array_2 over axes, then return the elementwise
  min of the resulting arrays.
  Hint: if a, b are arrays with same shape, np.minimum(a,b) is elementwise min 
  of a and b.
  """
  assert array_1.shape == array_2.shape
  min_1, min_2 = np.sum(array_1, axis = axes), np.sum(array_2, axis = axes)
  total_min = np.minimum(min_1, min_2)
  return total_min
  

# %%
# sum_then_min tests
a1 = np.array([1.,2.,3.])
b1 = np.array([1.,1.,1.])
stm1 = sum_then_min(a1,b1,(0,))
print(stm1) # should be 3.0
a2 = np.arange(40).reshape(2,4,5) # 3-dimensional array with shape (2,4,5)
b2 = np.ones([40]).reshape(2,4,5) # 3-dimensional array with shape (2,4,5)
stm2 = sum_then_min(a2,b2,(0,2)) # 1-dimensional array with shape (4,)
print(stm2) # should be [10. 10. 10. 10.]

# %% [markdown]
# ### Computing Mean and Variance of a Discrete Distribution
# 
# A discrete distribution is defined over a list of values, often called its **support**. Each item in the support has an associated **probability mass function**, which we will shorten to **pmf**.
# 
# The mean of the distribution (a scalar) can be computed with the following equation:
# 
# $$ \texttt{mean} = \sum_i \texttt{pmf[i]} * \texttt{support[i]}$$
# 
# Similary, the variance of the distribution (a scalar) can be computed with this equation:
# 
# $$ \texttt{variance} = \sum_i \texttt{pmf[i]} * (\texttt{support[i]} - \texttt{mean})^2 $$
# 
# Implement these equations as the functions `compute_mean` and `compute_variance`.
# 
# Note: you should not use a loop, instead use the `np.sum()` function!

# %%
def compute_mean(support,pmf):
  """
  Compute mean of the multinomial distribution
  Arguments:
    support: 1D numpy array, support of distribution
    pmf: 1D numpy array, probability mass function
  Returns:
    mean: float, the mean of the distribution
  """
  assert len(support) == len(pmf)
  mean = np.sum(pmf * support)
  return mean

# %%
# compute_mean tests
support1 = np.arange(5)
pmf1 = np.array([0.2,0.2,0.2,0.2,0.2])
mean1 = compute_mean(support1,pmf1)
print(mean1) # should be 2.0
support2 = np.arange(3)
pmf2 = np.array([0.3,0.2,0.5])
mean2 = compute_mean(support2,pmf2)
print(mean2) # should be 1.2

# %%
def compute_variance(support,pmf):
  """
  Compute variance of the multinomial distribution
  Arguments:
    support: 1D numpy array, support of distribution
    pmf: 1D numpy array, probability mass function
  Returns:
    variance: float, the variance of the distribution
  """
  assert len(support) == len(pmf)
  variance = np.sum(pmf * (support - compute_mean(support, pmf))**2)
  return variance

# %%
# compute_variance tests
support1 = np.arange(5)
pmf1 = np.array([0.2,0.2,0.2,0.2,0.2])
var1 = compute_variance(support1,pmf1)
print(var1) # should be 2.0
support2 = np.arange(3)
pmf2 = np.array([0.3,0.2,0.5])
var2 = compute_variance(support2,pmf2)
print(var2) # should be 0.76

# %% [markdown]
# Now, apply your `compute_mean` and `compute_variance` functions to estimate the mean and variance of a distribution from samples, by completing the `estimate_distribution` function below:

# %%
from numpy.core.fromnumeric import mean

def estimate_distribution(support, true_pmf, num_samples):
  """
  Draw IID samples from the distribution, estimate the distribution's parameters,
  and compute and print the true and estimated mean and variance.
  Arguments:
    support: 1D numpy array, support of distribution
    true_pmf: 1D numpy array, true probability mass function
    num_samples: int, number of samples to draw
  Returns:
    None
  """
  assert len(support) == len(true_pmf)
  assert num_samples > 0
  # compute true mean and variance using helper functions
  true_mean = compute_mean(support, true_pmf)
  true_variance = compute_variance(support, true_pmf)
  # sample data from the true distribution
  # samples is an array of size [num_samples,len(support)]
  samples = np.random.multinomial(1,true_pmf,size=(num_samples,))
  # estimate pmf by taking mean across samples (the 0th axis)
  est_pmf = np.sum(samples, axis = 0)/num_samples
  # compute estimated mean and variance using helper functions
  est_mean = compute_mean(support, est_pmf)
  est_variance = compute_variance(support, est_pmf)
  # print out the results
  print(f"True Statistics: Mean = {true_mean}, Variance = {true_variance}")
  print(f"Estimated Statistics: Mean = {est_mean}, Variance = {est_variance}")

# %%
# estimate_distribution tests
# the estimated means and variances should be close to the real ones
# there might be some inaccuracy since the estimate has some randomness
num_samples = 1000
support1 = np.arange(5)
pmf1 = np.array([0.2,0.2,0.2,0.2,0.2])
estimate_distribution(support1,pmf1,num_samples)
support2 = np.arange(3)
pmf2 = np.array([0.3,0.2,0.5])
estimate_distribution(support2,pmf2,num_samples)


