'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import modules
import csv
import math
import sklearn
import sklearn.metrics as metrics
import tqdm
from scipy.spatial import distance
import numpy as np
from scipy.optimize import brentq
from scipy import interpolate
from modules.evaluations import get_val_data, perform_val, perform_val_yts
from modules.models import ArcFaceModel, IoMFaceModelFromArFace, IoMFaceModelFromArFaceMLossHead,IoMFaceModelFromArFace2,IoMFaceModelFromArFace3,IoMFaceModelFromArFace_T,IoMFaceModelFromArFace_T1
from modules.utils import set_memory_growth, load_yaml, l2_norm
import urllib
from sklearn.model_selection import KFold

# modules.utils.set_memory_growth()
flags.DEFINE_string('cfg_path', './configs/iom_res50.yaml', 'config file path')
flags.DEFINE_string('ckpt_epoch', '', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')


def load_data_from_dir(save_path, BATCH_SIZE=128, subset='Sadie_Frost/1', img_ext='jpg'):
    def transform_test_images(img):
        img = tf.image.resize(img, (112, 112))
        img = img / 255
        return img

    def get_label_withname(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        wh = parts[-2]
        return wh

    def process_path_withname(file_path):
        label = get_label_withname(file_path)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = transform_test_images(img)
        return img, label

    #     list_gallery_ds = tf.data.Dataset.list_files(save_path +'/'+subset+'/*.'+img_ext).shuffle(100).take(5)
    list_gallery_ds = tf.data.Dataset.list_files(save_path + '/' + subset + '/*.' + img_ext)
    labeled_gallery_ds = list_gallery_ds.map(lambda x: process_path_withname(x),
                                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = labeled_gallery_ds.batch(BATCH_SIZE)
    return dataset


def eucliden_dist(embeddings1, embeddings2):
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    return dist

def extractFeat(dataset, model, feature_dim=512):
    final_feature = np.zeros(feature_dim)
    feats = []
    names = []
    n = 0
    for image_batch, label_batch in dataset:
        feature = model(image_batch)
        for i in range(feature.shape[0]):
            n = n + 1
            feats.append(feature[i])
            mylabel = label_batch[i].numpy()
            names.append(mylabel)
            if feature[i] is not None:
                final_feature += feature[i] / np.linalg.norm(feature[i], ord=2)
    #         print(f"[*] finanly we have {n} extracted samples features"
    final_feature /= np.linalg.norm(final_feature, ord=2)
    return final_feature

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def computeEER(issames,scores):
    nrof_pairs = len(issames)
    thresholds = np.arange(0, 4, 0.01)
    nrof_thresholds = len(thresholds)
    nrof_folds = 10
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    print(nrof_pairs)

    dist = np.array(scores)
    actual_issame = np.array(issames)
    indices = np.arange(nrof_pairs)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = \
                calculate_accuracy(threshold,
                                   dist[test_set],
                                   actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    auc = metrics.auc(fpr, tpr)
    # print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)  # 512 8 10.9% Original 9.8%
    return eer,auc

def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    # cfg = load_yaml('./config_arc/arc_lres100ir.yaml')  #
    cfg = load_yaml(FLAGS.cfg_path)
    permKey = None
    if cfg['head_type'] == 'IoMHead':  #
        # permKey = generatePermKey(cfg['embd_shape'])
        permKey = tf.eye(cfg['embd_shape'])  # for training, we don't permutate, won't influence the performance

    arcmodel = ArcFaceModel(size=cfg['input_size'],
                            embd_shape=cfg['embd_shape'],
                            backbone_type=cfg['backbone_type'],
                            head_type='ArcHead',
                            training=False,
                            cfg=cfg)
    if cfg['backbone_type'] == 'ResNet50':
        ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_res50')

    elif cfg['backbone_type'] == 'InceptionResNetV2':
        ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_InceptionResNetV2')
    elif cfg['backbone_type'] == 'Xception':
        ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_Xception')
    elif cfg['backbone_type'] == 'lresnet100e_ir':
        ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_lresnet100e_ir')
    else:
        ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_res50')

    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcmodel.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()

    ###### get matching protocol
    link = "https://www.cs.tau.ac.il/~wolf/ytfaces/splits.txt"
    file = urllib.request.urlopen(link)
    listmy = []
    for line in file:
        decoded_line = line.decode("utf-8")
        listmy.append(decoded_line.split(","))
    def getScore(arcmodel):
        scores = []
        issames = []
        dict = {}
        for i in tqdm.tqdm(range(1, 5001)):
            first_name = listmy[i][2].strip()
            second_name = listmy[i][3].strip()
            issame = int(listmy[i][4].strip())
            if not dict.__contains__(first_name.replace("/", "_")):
                try:
                    dataset_1 = load_data_from_dir('./data/test_dataset/aligned_images_DB_YTF/160x160',
                                                   subset=first_name)
                    feats1 = extractFeat(dataset_1, arcmodel)
                    dict[first_name.replace("/", "_")] = feats1
                except Exception:
                    print('[*]', first_name, second_name, 'failed')
                    continue
            if not dict.__contains__(second_name.replace("/", "_")):
                try:
                    dataset_2 = load_data_from_dir('./data/test_dataset/aligned_images_DB_YTF/160x160',
                                                   subset=second_name)
                    feats2 = extractFeat(dataset_2, arcmodel)
                    dict[second_name.replace("/", "_")] = feats2
                except Exception:
                    print('[*]', first_name, second_name, 'failed')
                    continue
            # feats1 = extractFeat(dataset_1, arcmodel)
            # feats2 = extractFeat(dataset_2, arcmodel)
            if dict.__contains__(first_name.replace("/", "_")) and dict.__contains__(second_name.replace("/", "_")):
                feats1 = dict[first_name.replace("/", "_")]
                feats2 = dict[second_name.replace("/", "_")]
                #     dist = sklearn.metrics.pairwise_distances(feats1, feats2, metric='hamming')
                score = distance.euclidean(feats1, feats2)
                # dist = distance.hamming(embeddings1, embeddings2)
                #     dist = tf.linalg.diag_part(dist)
                #     dist = dist.numpy()
                #     score = np.average(dist)
                # print('issame', issame, 'score', score)
                scores.append(score)
                issames.append(issame)
        return scores,issames

    # scores, issames = getScore(arcmodel)
    # eer_orig, auc_orig = computeEER(issames, scores)
    # eer_orig = 0
    # auc_orig = 0
    # print(eer_orig,auc_orig)
    model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                   arcmodel=arcmodel, training=False,
                                   permKey=permKey, cfg=cfg)
    scores, issames = getScore(model)
    eer_r_iom, auc_r_iom = computeEER(issames, scores)


    if FLAGS.ckpt_epoch == '':
        ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    else:
        ckpt_path = './checkpoints/' + cfg['sub_name'] + '/' + FLAGS.ckpt_epoch
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()
    model.summary(line_length=80)
    scores, issames = getScore(model)
    eer_dl_iom, auc_dl_iom = computeEER(issames, scores)

    log_str2 = '''backbone={} \t {:.4f}\t {:.4f}\t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n\n '''.format(
        cfg['backbone_type'],cfg['m'], cfg['q'], eer_orig, auc_orig,eer_r_iom, auc_r_iom,eer_dl_iom, auc_dl_iom)
    with open('./logs/YTF_' + cfg['sub_name'] + "_Output.md", "a") as text_file:
        text_file.write(log_str2)
    print(log_str2)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
