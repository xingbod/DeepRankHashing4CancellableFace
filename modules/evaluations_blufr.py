"""
This script was modified from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
"""

import time
def loadBLUFR():
    # Read LFW data
    features_lfw, label_ver_blufr = data.loadBLUFR()
    data_ver = load_cPickle(label_ver_blufr[0])
    # print "data readed"
    return data_ver, features_lfw

def computeStatsMulti(descriptors, ground_truth, distance=cosineDistance):
    scores = threadComputeMatrix(descriptors, ground_truth, distance)
    acc, roc = computeAccuracyROC(scores)
    return acc, roc


def calc_acc_multi(data_ver, features):
  time1 = time.time()
  acc, roc = computeStatsMulti(features, data_ver)
  time2 = time.time()
  print('%s function took %0.3f ms' % ("Multi", (time2-time1)*1000.0))
  print("ACC", acc)

