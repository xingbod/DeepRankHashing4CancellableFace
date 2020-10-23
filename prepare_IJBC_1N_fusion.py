'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
# !/usr/bin/env python
# coding: utf-8
import pickle
import os
import numpy as np
import timeit
import sklearn
import sklearn.metrics
import cv2
import sys
import argparse
import glob
import numpy.matlib
import heapq
import math
from datetime import datetime as dt
import tqdm
from sklearn import preprocessing

sys.path.append('recognition')
from embedding import Embedding
from menpo.visualize import print_progress
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap

from modules.utils import load_yaml
from modules.models import build_or_load_IoMmodel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def read_template_subject_id_list(path):
    ijb_meta = np.loadtxt(path, dtype=str, skiprows=1, delimiter=',')
    templates = ijb_meta[:, 0].astype(np.int)
    subject_ids = ijb_meta[:, 1].astype(np.int)
    return templates, subject_ids


def read_template_media_list(path):
    ijb_meta = np.loadtxt(path, dtype=str)
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    pairs = np.loadtxt(path, dtype=str)
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


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
            img_feats.append(feats)
            crop_imgs = []
        img_index = img_index + 1
    if len(crop_imgs) > 0:
        print('processing', img_index)
        feats = embedding.getFeat(crop_imgs)
        img_feats.append(feats)
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)

    # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
    # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
    return img_feats, faceness_scores


def image2template_feature(img_feats=None, templates=None, medias=None, choose_templates=None, choose_ids=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates, indices = np.unique(choose_templates, return_index=True)
    unique_subjectids = choose_ids[indices]
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
    template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    return template_norm_feats, unique_templates, unique_subjectids


def image2template_feature_hash(img_feats=None, templates=None, medias=None, choose_templates=None, choose_ids=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates, indices = np.unique(choose_templates, return_index=True)
    unique_subjectids = choose_ids[indices]
    print('***img_feats**', img_feats[0])
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.median(face_norm_feats[ind_m], 0, keepdims=True)]# using sum to try median can achieve good perf 40%  sum can not 3% mean can also 30%
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.median(media_norm_feats, 0)# median can achieve good perf sum-mean can not.median-sum cannot
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(count_template))
    # print('***template_feats',template_feats[0])
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    # template_feats = np.round(template_feats)
    print('***template_feats***',template_feats[0])
    # template_norm_feats = template_feats
    template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    print('***finaltemplate***',template_norm_feats[0])
    return template_norm_feats, unique_templates, unique_subjectids


def verification(template_norm_feats=None, unique_templates=None, p1=None, p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do ijb 1n test')
    # general
    parser.add_argument('--model-prefix', default='', help='path to load model.')
    parser.add_argument('--model-epoch', default=1, type=int, help='')
    parser.add_argument('--gpu', default=7, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--cfg_path', default='configs/config_random/iom_res100_random_insightface.yaml', type=str,
                        help='your config file')
    parser.add_argument('--job', default='insightface', type=str, help='job name')
    parser.add_argument('--target', default='IJBC', type=str, help='target, set to IJBC or IJBB')
    parser.add_argument('--gallery_mark', default=1, type=int, help='Using which gallery?,default 0: combine and close-set, 1 G1, 2 G2')
    parser.add_argument('--is_only_arc', default=1, type=int, help='is ArcFace only? Or IoM added')
    args = parser.parse_args()
    target = args.target
    model_path = args.model_prefix
    gpu_id = args.gpu
    cfg_path = args.cfg_path
    batch_size = args.batch_size
    epoch = args.model_epoch
    gallery_mark = args.gallery_mark
    is_only_arc = args.is_only_arc



    meta_dir = "%s/meta" % args.target  # meta root dir
    if target == 'IJBC':
        gallery_s1_record = "%s_1N_gallery_G1.csv" % (args.target.lower())
        gallery_s2_record = "%s_1N_gallery_G2.csv" % (args.target.lower())
    else:
        gallery_s1_record = "%s_1N_gallery_S1.csv" % (args.target.lower())
        gallery_s2_record = "%s_1N_gallery_S2.csv" % (args.target.lower())
    gallery_s1_templates, gallery_s1_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s1_record))
    print(gallery_s1_templates.shape, gallery_s1_subject_ids.shape)

    gallery_s2_templates, gallery_s2_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, gallery_s2_record))
    print(gallery_s2_templates.shape, gallery_s2_templates.shape)

    if gallery_mark == 1:
        print('Using G1!',gallery_s1_templates.shape)
        gallery_templates = gallery_s1_templates
        gallery_subject_ids = gallery_s1_subject_ids
    elif gallery_mark == 2:
        print('Using G2',gallery_s1_templates.shape)
        gallery_templates = gallery_s2_templates
        gallery_subject_ids = gallery_s2_subject_ids
    else:
        gallery_templates = np.concatenate([gallery_s1_templates, gallery_s2_templates])
        gallery_subject_ids = np.concatenate([gallery_s1_subject_ids, gallery_s2_subject_ids])

    print(gallery_templates.shape, gallery_subject_ids.shape)


    media_record = "%s_face_tid_mid.txt" % args.target.lower()
    total_templates, total_medias = read_template_media_list(os.path.join(meta_dir, media_record))
    print("total_templates", total_templates.shape, total_medias.shape)
    # load image features
    start = timeit.default_timer()
    feature_path = ''  # feature path
    face_path = ''  # face path
    img_path = './%s/loose_crop' % target
    img_list_path = './%s/meta/%s_name_5pts_score.txt' % (target, target.lower())
    # img_feats, faceness_scores = get_image_feature(feature_path, face_path)

    cfg = load_yaml(cfg_path)  # cfg = load_yaml(FLAGS.cfg_path)
    # model = build_or_load_IoMmodel(cfg, is_only_arc=is_only_arc)
    # model.summary(line_length=80)
    #
    # # img_feats, faceness_scores = get_image_feature(img_path, img_list_path, model)


    if is_only_arc:
        cfg['m'] = 0
        cfg['q'] = 0
    else:
        cfg['m'] = 512
        cfg['q'] = 8

    img_feats = np.load("data_ijbc/img_feats_" + cfg['backbone_type'] + '_' + str(is_only_arc) + '_' + str(cfg['m']) + 'x' + str(
        cfg['q']) + ".npy")
    faceness_scores = np.load("faceness_scores.npy")

    # compute template features from image features.
    start = timeit.default_timer()
    # ==========================================================
    # Norm feature before aggregation into template feature?
    # Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
    # ==========================================================
    use_norm_score = True  # if True, TestMode(N1)
    use_detector_score = False  # if True, TestMode(D1)
    use_flip_test = True  # if True, TestMode(F1)

    if use_flip_test:
        # concat --- F1
        img_input_feats = img_feats
        # add --- F2
        # img_input_feats = img_feats[:, 0:int(img_feats.shape[1] / 2)] + img_feats[:, int(img_feats.shape[1] / 2):]
    else:
        img_input_feats = img_feats[:, 0:int(img_feats.shape[1] / 2)]

    if use_norm_score:
        img_input_feats = img_input_feats
    else:
        # normalise features to remove norm information
        img_input_feats = img_input_feats / np.sqrt(np.sum(img_input_feats ** 2, -1, keepdims=True))

    if use_detector_score:
        img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:, np.newaxis], 1,
                                                             img_input_feats.shape[1])
    else:
        img_input_feats = img_input_feats
    print("input features shape", img_input_feats.shape)

    # load gallery feature # image2template_feature_hash image2template_feature
    gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = image2template_feature_hash(
        img_input_feats, total_templates, total_medias, gallery_templates, gallery_subject_ids)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print("gallery_templates_feature", gallery_templates_feature.shape)
    print("gallery_unique_subject_ids", gallery_unique_subject_ids.shape)
    # np.savetxt("gallery_templates_feature.txt", gallery_templates_feature)
    # np.savetxt("gallery_unique_subject_ids.txt", gallery_unique_subject_ids)

    # load prope feature
    probe_mixed_record = "%s_1N_probe_mixed.csv" % target.lower()
    probe_mixed_templates, probe_mixed_subject_ids = read_template_subject_id_list(
        os.path.join(meta_dir, probe_mixed_record))
    print(probe_mixed_templates.shape, probe_mixed_subject_ids.shape)
    probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = image2template_feature_hash(
        img_input_feats, total_templates, total_medias, probe_mixed_templates, probe_mixed_subject_ids)
    print("probe_mixed_templates_feature", probe_mixed_templates_feature.shape)
    print("probe_mixed_unique_subject_ids", probe_mixed_unique_subject_ids.shape)
    # np.savetxt("probe_mixed_templates_feature.txt", probe_mixed_templates_feature)
    # np.savetxt("probe_mixed_unique_subject_ids.txt", probe_mixed_unique_subject_ids)

    # root_dir = "" #feature root dir
    # gallery_id_path = "" #id filepath
    # gallery_feats_path = "" #feature filelpath
    # print("{}: start loading gallery feat {}".format(dt.now(), gallery_id_path))
    # gallery_ids, gallery_feats = load_feat_file(root_dir, gallery_id_path, gallery_feats_path)
    # print("{}: end loading gallery feat".format(dt.now()))
    #
    # probe_id_path = "probe_mixed_unique_subject_ids.txt" #probe id filepath
    # probe_feats_path = "probe_mixed_templates_feature.txt" #probe feats filepath
    # print("{}: start loading probe feat {}".format(dt.now(), probe_id_path))
    # probe_ids, probe_feats = load_feat_file(root_dir, probe_id_path, probe_feats_path)
    # print("{}: end loading probe feat".format(dt.now()))

    gallery_ids = gallery_unique_subject_ids
    gallery_feats = gallery_templates_feature
    probe_ids = probe_mixed_unique_subject_ids
    probe_feats = probe_mixed_templates_feature
    #

    # np.savetxt("data/"+"IJBC1N_" + cfg['backbone_type'] + '_' + str(is_only_arc) + '_' + str(cfg['m']) + 'x' + str(
    #     cfg['q'])+"gallery_ids.csv", gallery_ids, delimiter=",")
    # np.savetxt("data/"+"IJBC1N_" + cfg['backbone_type'] + '_' + str(is_only_arc) + '_' + str(cfg['m']) + 'x' + str(
    #     cfg['q'])+"gallery_feats.csv", gallery_feats, delimiter=",")
    # np.savetxt("data/"+"IJBC1N_" + cfg['backbone_type'] + '_' + str(is_only_arc) + '_' + str(cfg['m']) + 'x' + str(
    #     cfg['q'])+"probe_ids.csv", probe_ids, delimiter=",")
    # np.savetxt("data/"+"IJBC1N_" + cfg['backbone_type'] + '_' + str(is_only_arc) + '_' + str(cfg['m']) + 'x' + str(
    #     cfg['q'])+"probe_feats.csv", probe_feats, delimiter=",")
    np.savetxt("data/gallery_ids.csv", gallery_ids, delimiter=",")
    np.savetxt("data/gallery_feats.csv", gallery_feats, delimiter=",")
    np.savetxt("data/probe_ids.csv", probe_ids, delimiter=",")
    np.savetxt("data/probe_feats.csv", probe_feats, delimiter=",")
    print("[**]Completed!!!")