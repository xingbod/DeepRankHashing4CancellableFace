'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
# coding: utf-8

import os
import numpy as np
# import cPickle
import pickle
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import sklearn.metrics
import skimage.transform
import argparse
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import cv2
import sys
import glob

from recognition.embedding import Embedding
from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import warnings

import tqdm
import os
import numpy as np
from modules.utils import load_yaml
from modules.models import build_or_load_IoMmodel,build_or_load_Random_IoMmodel,build_iom_model

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--model-prefix', default='', help='path to load model.')
parser.add_argument('--model-epoch', default=1, type=int, help='')
parser.add_argument('--gpu', default=7, type=int, help='gpu id')
parser.add_argument('--batch-size', default=32, type=int, help='')
parser.add_argument('--cfg_path', default='configs/config_random/iom_res100_random_insightface.yaml', type=str, help='your config file')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
parser.add_argument('--is_only_arc', default=0, type=int, help='is ArcFace only? Or IoM added')
args = parser.parse_args()

target = args.target
model_path = args.model_prefix
batch_size = args.batch_size
is_only_arc = args.is_only_arc
cfg_path = args.cfg_path
gpu_id = args.gpu
epoch = args.model_epoch
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job
# is_only_arc = True
print("is_only_arc?",is_only_arc)
def read_template_media_list(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


# In[ ]:


def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


# In[ ]:


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# In[ ]:


def get_image_feature(img_path, img_list_path, model):
    img_list = open(img_list_path)
    embedding = Embedding(model)
    files = img_list.readlines()
    print('files:', len(files))
    faceness_scores = []
    img_feats = []
    crop_imgs = []
    img_index = 1
    for each_line in tqdm.tqdm(files):
        # if img_index % 500 == 0:
        #     print('processing', img_index)
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]], dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        crop_img = embedding.getCropImg(img, lmk)
        crop_imgs.append(crop_img)
        # img_feats.append(embedding.get(img, lmk))
        faceness_scores.append(name_lmk_score[-1])
        if len(crop_imgs) == batch_size:
            # print('processing', img_index,len(crop_imgs))
            feats = embedding.getFeat(np.array(crop_imgs))
            if len(img_feats) == 0:
                img_feats = feats
            else:
                img_feats = np.concatenate((img_feats, feats), axis=0)
            crop_imgs = []
        img_index = img_index + 1
    if len(crop_imgs) > 0:
        print('processing', img_index)
        feats = embedding.getFeat(np.array(crop_imgs))
        img_feats = np.concatenate((img_feats, feats), axis=0)
    img_feats = np.array(img_feats)
    faceness_scores = np.array(faceness_scores).astype(np.float32)

    # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
    # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
    return img_feats, faceness_scores


# In[ ]:



def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# # Step1: Load Meta Data

# In[ ]:

assert target == 'IJBC' or target == 'IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(os.path.join('%s/meta' % target, '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:


# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(os.path.join('%s/meta' % target, '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 2: Get Image Features

# In[ ]:

# load model


cfg = load_yaml(cfg_path)  # cfg = load_yaml(FLAGS.cfg_path)
if is_only_arc:
    model = build_or_load_IoMmodel(cfg, is_only_arc=is_only_arc)
else:
    model = build_iom_model(cfg)
model.summary(line_length=80)

# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
img_path = './%s/loose_crop' % target
img_list_path = './%s/meta/%s_name_5pts_score.txt' % (target, target.lower())
img_feats, faceness_scores = get_image_feature(img_path, img_list_path, model)
# img_feats = np.load("img_feats_" + cfg['backbone_type'] + '_' + str(is_only_arc) + '_' + str(cfg['m']) + 'x' + str(
#     cfg['q']) + ".npy")
# faceness_scores = np.load("faceness_scores.npy")
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))

if is_only_arc:
    cfg['m'] = 0
    cfg['q'] = 0

np.save("data/ijbc_img_feats_learning_"+cfg['backbone_type']+'_'+ str(cfg['m']) + 'x' + str(cfg['q'])+".npy", img_feats)
np.save("data/ijbc_faceness_scores.npy", faceness_scores)
# # Step3: Get Template Features
