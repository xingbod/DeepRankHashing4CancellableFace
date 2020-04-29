"""
This script was modified from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
"""
import time
import bcolz
import numpy as np
import os

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_val_data(data_path):
    """get validation data"""
    lfw, lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')
    agedb_30, agedb_30_issame = get_val_pair(data_path,
                                             'agedb_align_112/AgeDB/agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_align_112/cfp_fp')

    return lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame



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

