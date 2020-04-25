'''
@xingbo dong, xingbod@gmail.com
This verification is used to evaluate the eer, roc, accuracy for verification protocol

Example:
    roc_eer,acc, roc = computeAccuracyROC([0.8,0.9,0.6,0.3,0.6],[0.1,0.2,0.4,0.1,0.6])
    print('roc_eer:',roc_eer)
    print('acc:',acc)
    print('roc:',roc)
    plotROC(roc, [], title=None, show_grid=True)

Output roc_eer: 0.20
        acc: 0.8
'''

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.pylab import cm
import pylab
from distance import cosineDistance
from scipy.optimize import brentq
from scipy.interpolate import interp1d

'''Plot ROC curve'''
def plot_result(probas_data,labels,acc):
  fpr, tpr, thresholds = roc_curve(labels, probas_data)
  roc_auc = auc(fpr, tpr)
  plt.plot(fpr, tpr, lw=1, label='ROC  (area = %0.2f)' % ( roc_auc))
  plt.legend(loc="lower right")
  plt.title("Accuracy: " + str(acc))
  plt.grid()
  plt.show()
  

def plotROC(rocs, labels, title=None, show_grid=True):
    prev_figsize = pylab.rcParams['figure.figsize']

    pylab.rcParams['figure.figsize'] = (8.0, 6.0)

    for roc,label in zip(rocs,labels):
        plt.plot(roc[0], roc[1], label=label)
    
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")

    if title:
        plt.title(title)
    if show_grid:
        plt.grid()

    pylab.rcParams['figure.figsize'] = prev_figsize

def plot_confusion_matrix(true_labels,pred_labels):
  # Compute confusion matrix and save to drive
  conf_mat = confusion_matrix(np.array(true_labels), np.array(pred_labels))
  print(conf_mat)
  # Show confusion matrix in a separate window
  #plt.matshow(conf_mat, cmap=cm.jet)
  #plt.title('Confusion matrix')
  #plt.colorbar()
  #plt.ylabel('True label')
  #plt.xlabel('Predicted label')

def statsForThreshold(matches_scores, mismatches_scores, threshold):
    tp = np.sum(matches_scores >= threshold)
    fn = len(matches_scores) - tp
                
    tn = np.sum(mismatches_scores < threshold)
    fp = len(mismatches_scores) - tn
    
    return (fp / float(fp+tn), tp / float(tp+fn))


def computeROC(matches_scores, mismatches_scores, thresholds=None):
    if thresholds is None:
        score_distribution = np.concatenate([matches_scores, mismatches_scores])
        thresholds = np.linspace(np.min(score_distribution), np.max(score_distribution), num=100)    
    fprs,tprs = [],[]
    for threshold in thresholds:
        fpr,tpr = statsForThreshold(matches_scores, mismatches_scores, threshold)
        fprs.append(fpr)
        tprs.append(tpr)
        
    return fprs,tprs

def computeROCThreshold(matches_scores, mismatches_scores,):
   
    score_distribution = np.concatenate([matches_scores, mismatches_scores])
    thresholds = np.linspace(np.min(score_distribution), np.max(score_distribution), num=100)    
    fprs,tprs = [],[]
    for threshold in thresholds:
        fpr,tpr = statsForThreshold(matches_scores, mismatches_scores, threshold)
        fprs.append(fpr)
        tprs.append(tpr)
        
    return fprs,tprs, thresholds


def computeAccuracyROC(matches_scores, mismatches_scores):
    fprs, tprs = computeROC(matches_scores, mismatches_scores)
    eer = brentq(lambda x: 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
    return eer,(np.max(((1-np.asarray(fprs)) + np.asarray(tprs))/2)), (fprs, tprs)
  
def getBestThreshold(scores):
    fprs, tprs, threshold = computeROCThreshold(scores)
    arg_max = np.argmax(((1-np.asarray(fprs)) + np.asarray(tprs))/2)
    print("argmax: ",arg_max)
    return threshold[arg_max]
    
def computeDistanceMatrix(descs, sets_ground_truth, distance=cosineDistance):  
    #Compute distance for given set using predefinded distance
    matches_scores = []
    mismatches_scores = []
    if len(sets_ground_truth[0]) > 0:
      for matches in sets_ground_truth[0]:
          matches_scores.append( distance(descs[matches[0]], descs[matches[1]]) )
      # print "Size matches_scores ", len(matches_scores), matches_scores[-1].shape
    if len(sets_ground_truth[1]) > 0:
      for matches in sets_ground_truth[1]:
          mismatches_scores.append( distance(descs[matches[0]], descs[matches[1]]) )

    print(len(matches_scores), len(mismatches_scores))

    return (matches_scores, mismatches_scores)

from multiprocessing import Process, Pool, Queue,JoinableQueue
import threading

class ThreadDistanceMatrix(threading.Thread):
    '''Worker for Parallel extracting features using Caffe and Multi-GPU'''
    def __init__(self, tasks, results, distance):
      threading.Thread.__init__(self)
      self.result       = results
      self.task_queue   = tasks
      self.distance     = distance
     
    
    def run(self):
      '''Run Taks until you will meet 'exit' as as task
         Result put at list'''
      proc_name = self.name
      while True:
        next_task = self.task_queue.get()
        # print "Next"
        if str(next_task) == "exit":
            #exit means shutdown
            # print '%s: Exiting' % proc_name
            self.task_queue.task_done()
            break
        tmp = self.distance (next_task[0], next_task[1])
        self.task_queue.task_done()
        self.result.put(tmp)
    
      return

class Worker(Process):
    
    def __init__(self, tasks, results, distance, descriptor):
      Process.__init__(self)
      self.result       = results
      self.task_queue   = tasks
      self.distance     = distance
      self.descriptor   = descriptor 

    def run(self):
      proc_name = self.name
      while True:
        next_task = self.task_queue.get()
        # print "Next"
        if str(next_task) == "exit":
          #exit means shutdown
          # print '%s: Exiting' % proc_name
          self.task_queue.task_done()
          break
        tmp = self.computeDistance (next_task)
        self.task_queue.task_done()
        self.result.put(tmp)

    def computeDistance(self, data):
      matches_scores = list()
      for matches in data:
        matches_scores.append( self.distance(self.descriptor[matches[0]], self.descriptor[matches[1]]) )

      return matches_scores

def runParallel(descs, sets_ground_truth, distance=cosineDistance):
  results = Queue()
  tasks = JoinableQueue()
  num_cpu = 6
  work_divider = 6
  #for each thread, create worker
  jobs = [ Worker(tasks, results, distance, descs)
                  for i in range(num_cpu) ]
  

  in_one_thread = int( len(sets_ground_truth)/ (num_cpu * work_divider))
  num_work = 0
  for i in range(0, len(sets_ground_truth), in_one_thread):
    num_work += 1
    if (i + in_one_thread) < len(sets_ground_truth):
      tasks.put(sets_ground_truth[i:i+in_one_thread])
    else:
      tasks.put(sets_ground_truth[i:])

  # num_work = len(sets_ground_truth)    
  #Add a exit for each consumer to end processing
  for i in range(num_cpu):
      tasks.put("exit")

  for j in jobs:
    j.start()
  # Wait for all of the tasks to finish
  tasks.join()
  list_object_pos = list()
  while num_work:
    list_object_pos.extend(results.get())
    num_work -= 1
  
  print(len(list_object_pos))
  list_object_pos =  np.asarray(list_object_pos)
  return list_object_pos

def threadComputeMatrix(descs, sets_ground_truth, distance=cosineDistance): 
  if len(sets_ground_truth[0]) > 0:
    matches    = runParallel(descs, sets_ground_truth[0], distance)
  else:
    matches = list()
  if len(sets_ground_truth[1]) > 0:
    mismatches = runParallel(descs, sets_ground_truth[1], distance)
  else:
    mismatches = list()
  return (matches, mismatches)

if __name__ == '__main__':
    roc_eer,acc, roc = computeAccuracyROC([0.8,0.9,0.6,0.3,0.6],[0.1,0.2,0.4,0.1,0.6])
    print('roc_eer:',roc_eer)
    print('acc:',acc)
    print('roc:',roc)
    plotROC(roc, [], title=None, show_grid=True)
