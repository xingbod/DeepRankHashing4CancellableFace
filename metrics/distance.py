#FV-Benchmark
import numpy as np


def cosineDistance(x, y):
    return np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
  
def chi2Distance(x, y):
  feat_div = (x - y)**2
  for i in range(x.shape[0]):
    sum_feat = (x[i] + y[i])
    if sum_feat != 0.0:
      feat_div[i] /= sum_feat
  return feat_div
  
def chi2DistanceSum(x, y):
   return -np.sum(chi2Distance(x, y))
  
def jointBayesianDistance(x, y):
    return x[0] + y[0] + 2*np.dot(x[1:].T, y[1:])