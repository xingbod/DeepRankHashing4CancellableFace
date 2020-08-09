This MATLAB package provides an evaluation toolkit for the Benchmark of Large-scale Unconstrained Face Recognition (BLUFR) introduced in our IJCB paper [1]. The Labeled Faces in the Wild (LFW) database [2] is the main benchmark database in this package. Besides, we also provide a similar evaluation protocol on the FRGCv2 database to verify algorithms' performance with controlled face images.

The package contains the following components.

(1) BLUFR configuration files contained in the config subfolder. The .mat configuration file provides basic image lists or indexes of the training, test, and development sets for the benchmark.

(2) List files contained in the list subfolder. These list files are text files, provided as an alternative of the configuration file. This is particularly useful if you are using other programming languages like C++ and Python.

(3) Evaluation utilities and demo codes included in the code subfolder.

(4) Basic features contained in the data subfolder. Due to the size of the feature files, they are provided as separate downloads.

For a quick start, download the lfw.mat and place it in the data subfolder, then run the demo_pca.m and demo_supervised.m examples. For the benchmark of your own algorithm, replace the basic features, and integrate your own algorithm in the demo codes.

For algorithm comparison, we suggest to report the verification rates at FAR=0.1%, and the open-set identification rates (DIR) at FAR=1% and rank=1. Besides, ROC curves for verification and ROC curves for open-set identification at rank 1 can also be illustrated. Please make sure to report the (mu-sigma) performance, not the average.

We will maintain the up to date top 10 results, ranked by verification rates at FAR=0.1%, of this benchmark in our project page. If you published a new result within the top 10 results we maintained, please send your result (as a mat file illustrated in the demo codes) to us so that we can update the top 10 results to include your algorithm's performance.


IMPORTANT: You must also cite the original LFW technical report [2] which provides the LFW database as well as the original evaluation protocols.


Version: 1.0
Date: 2014-07-22
 
Author: Shengcai Liao
Institute: National Laboratory of Pattern Recognition, 	Institute of Automation, Chinese Academy of Sciences

Email: scliao@nlpr.ia.ac.cn

Project page: http://www.cbsr.ia.ac.cn/users/scliao/projects/blufr/


References:

[1] Shengcai Liao, Zhen Lei, Dong Yi, Stan Z. Li, "A Benchmark Study of Large-scale Unconstrained Face Recognition." In IAPR/IEEE International Joint Conference on Biometrics, Sep. 29 - Oct. 2, Clearwater, Florida, USA, 2014.

[2] G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst, October 2007. http://vis-www.cs.umass.edu/lfw/.
