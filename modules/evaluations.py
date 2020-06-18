"""
This script was modified from https://github.com/ZhaoJ9014/face.evoLVe.PyTorch
"""
import os
import cv2
import bcolz
import numpy as np
import tqdm
from sklearn.model_selection import KFold
import tensorflow as tf
from modules.utils import l2_norm
# sklearn scipy used for EER cal.
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from modules.dataset import load_data_split
from metrics.retrieval import streaming_mean_averge_precision, streaming_mean_cmc_at_k
from sklearn import preprocessing
from modules.LUT import genLUT
import scipy.spatial.distance as dist
import sklearn.metrics


def pdist(a, b=None):
    """Compute element-wise squared distance between `a` and `b`.
    Parameters
    ----------
    a : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    b : tf.Tensor
        A matrix of shape MxL with M row-vectors of dimensionality L.
    Returns
    -------
    tf.Tensor
        A matrix of shape NxM where element (i, j) contains the squared
        distance between elements `a[i]` and `b[j]`.
    """
    sq_sum_a = tf.reduce_sum(tf.square(a), axis=1)
    if b is None:
        return -2 * tf.matmul(a, tf.transpose(a)) + \
               tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_a, (1, -1))
    sq_sum_b = tf.reduce_sum(tf.square(b), axis=1)
    return -2 * tf.matmul(a, tf.transpose(b)) + \
           tf.reshape(sq_sum_a, (-1, 1)) + tf.reshape(sq_sum_b, (1, -1))


def cosine_distance(a, b=None):
    """Compute element-wise cosine distance between `a` and `b`.
    Parameters
    ----------
    a : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    b : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    Returns
    -------
    tf.Tensor
        A matrix of shape NxM where element (i, j) contains the cosine distance
        between elements `a[i]` and `b[j]`.
    """
    # print('***********a*******',a)
    a = tf.cast(a, tf.float32)  # cast to float before norm it
    a_normed = tf.nn.l2_normalize(a, axis=1)
    b_normed = a_normed if b is None else tf.nn.l2_normalize(tf.cast(b, tf.float32), axis=1)
    return (
            tf.constant(1.0, tf.float32) -
            tf.matmul(a_normed, tf.transpose(b_normed)))


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


def ccrop_batch(imgs):
    assert len(imgs.shape) == 4
    resized_imgs = np.array([cv2.resize(img, (128, 128)) for img in imgs])
    ccropped_imgs = resized_imgs[:, 8:-8, 8:-8, :]

    return ccropped_imgs


def hflip_batch(imgs):
    assert len(imgs.shape) == 4
    return imgs[:, :, ::-1, :]


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


def eucliden_dist(embeddings1, embeddings2):
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    return dist


import numpy as np
import scipy.spatial.distance as dist


def Hamming_dist(embeddings1, embeddings2):
    def cal(embeddings1, embeddings2):
        diff = np.subtract(embeddings1, embeddings2)
        smstr = np.nonzero(diff)
        # print('smstr',smstr)  # 不为0 的元素的下标
        dist = np.shape(smstr)[1] / np.shape(embeddings1)[0]
        return dist
    dist = np.zeros(np.shape(embeddings1)[0])
    for i in range(np.shape(embeddings1)[0]):
        dist[i] = cal(embeddings1[i, :], embeddings2[i, :])
    return dist


def eucliden_dist(embeddings1, embeddings2):
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    return dist


def cosin_dist(embeddings1, embeddings2):
    def cal(embeddings1, embeddings2):
        dist = np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * (np.linalg.norm(embeddings2)))
        return dist

    dist = np.zeros(np.shape(embeddings1)[0])
    for i in range(np.shape(embeddings1)[0]):
        dist[i] = cal(embeddings1[i, :], embeddings2[i, :])
    return np.array(1-dist)


def Jaccard_dist(embeddings1, embeddings2):
    def cal(embeddings1, embeddings2):
        matv = np.array([embeddings1, embeddings2])
        dist2 = dist.pdist(matv, 'jaccard')
        return dist2
    dist = np.zeros(np.shape(embeddings1)[0])
    for i in range(np.shape(embeddings1)[0]):
        dist[i] = cal(embeddings1[i, :], embeddings2[i, :])
    return np.array(dist)


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10, cfg=None, measure='Euclidean'):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    # diff = np.subtract(embeddings1, embeddings2)
    # dist = np.sum(np.square(diff), 1)
    # if cfg['head_type'] == 'IoMHead':
    #     # dist = dist/(cfg['q']*cfg['embd_shape']) # should divide by the largest distance
    #     dist = dist / (tf.math.reduce_max(dist).numpy()+10)  # should divide by the largest distance
    if measure == 'Euclidean':
        dist = sklearn.metrics.pairwise_distances(embeddings1, embeddings2, metric='euclidean')
        dist = tf.linalg.diag_part(dist)
        # dist = eucliden_dist(embeddings1, embeddings2)
        dist = dist / (tf.math.reduce_max(dist).numpy() + 1)
    elif measure == 'Hamming':
        dist = sklearn.metrics.pairwise_distances(embeddings1, embeddings2, metric='hamming')
        dist = tf.linalg.diag_part(dist)
        # dist = Hamming_dist(embeddings1, embeddings2)
    elif measure == 'Cosine':
        dist = sklearn.metrics.pairwise_distances(embeddings1, embeddings2, metric='cosine')
        dist = tf.linalg.diag_part(dist)
        # dist = cosin_dist(embeddings1, embeddings2)
        dist[dist < 0] = 0

    print("[*] dist {}".format(dist))
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
    # print('Equal Error Rate (EER): %1.3f' % eer)

    return tpr, fpr, accuracy, best_thresholds, auc, eer

def evaluate(embeddings, actual_issame, nrof_folds=10,measure='Hamming', cfg=None):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]  # 隔行采样
    embeddings2 = embeddings[1::2]  # 隔行采样
    tpr, fpr, accuracy, best_thresholds, auc, eer = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
        nrof_folds=nrof_folds, cfg=cfg,measure=measure)

    return tpr, fpr, accuracy, best_thresholds, auc, eer


def perform_val(embedding_size, batch_size, model,
                carray, issame, nrof_folds=10, is_ccrop=False, is_flip=False, cfg=None, isLUT=0,measure='Hamming'):
    """perform val"""
    if cfg['head_type'] == 'IoMHead':
        embedding_size = int(embedding_size / cfg['q'])
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size), ascii=True):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
        if is_ccrop:
            batch = ccrop_batch(batch)

        if is_flip:
            fliped = hflip_batch(batch)
            emb_batch = model(batch) + model(fliped)
            # embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            batch = ccrop_batch(batch)
            emb_batch = model(batch)
        # print(emb_batch)
        if cfg['head_type'] == 'IoMHead':
            embeddings[idx:idx + batch_size] = emb_batch  # not working? why
        else:
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        # embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        # print(embeddings)
    if isLUT:  # length of bin
        # here do the binary convert
        # # here convert the embedding to binary
        LUT1 = genLUT(q=cfg['q'], bin_dim=isLUT)
        embeddings = tf.cast(embeddings, tf.int32)
        LUV = tf.gather(LUT1, embeddings)
        embeddings = tf.reshape(LUV, (embeddings.shape[0], isLUT * embeddings.shape[1]))

        ##### end ########
    tpr, fpr, accuracy, best_thresholds, auc, eer = evaluate(
        embeddings, issame, nrof_folds,measure, cfg)

    return accuracy.mean(), best_thresholds.mean(), auc, eer, embeddings


'''
below if for archead with IoM layer
'''


def perform_val2(embedding_size, batch_size, model,
                 carray, issame, nrof_folds=10, is_ccrop=False, is_flip=False, cfg=None):
    """perform val"""
    embedding_size = int(embedding_size / cfg['q'])
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size), ascii=True):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1]) * 0.5 + 0.5
        if is_ccrop:
            batch = ccrop_batch(batch)

        if is_flip:
            fliped = hflip_batch(batch)
            emb_batch = model(batch) + model(fliped)
            # embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            batch = ccrop_batch(batch)
            emb_batch = model(batch)
        # print(emb_batch)
        embeddings[idx:idx + batch_size] = emb_batch  # not working? why
        # embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        # print(embeddings)
    tpr, fpr, accuracy, best_thresholds, auc, eer = evaluate(
        embeddings, issame, nrof_folds, cfg)

    return accuracy.mean(), best_thresholds.mean(), auc, eer, embeddings


'''
new add, in case val during train
'''


def val_LFW(model, cfg):
    lfw, lfw_issame = get_val_pair(cfg['test_dataset'], 'lfw_align_112/lfw')
    return perform_val(
        cfg['q'] * cfg['m'], 32, model, lfw, lfw_issame,
        is_ccrop=cfg['is_ccrop'], cfg=cfg)


'''
2020/04/29 new add by Xingbo, evaluate y.t.f and f.s
ds_path = 'E:/my research/etri2020/facedataset/facescrub_images_112x112'

'''


def perform_val_yts(batch_size, model, ds_path, is_ccrop=False, is_flip=False, cfg=None, img_ext='png', isLUT=False):
    """perform val for youtube face and facescrb"""

    def extractFeat(dataset, model):
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
        print(f"[*] finanly we have {n} extracted samples features")
        return feats, names

    gallery = load_data_split(ds_path, batch_size, subset='train_gallery', img_ext=img_ext)
    probes = load_data_split(ds_path, batch_size, subset='test', img_ext=img_ext)
    gallery_feats, gallery_names = extractFeat(gallery, model)
    probes_feats, probes_names = extractFeat(probes, model)
    if isLUT:
        # here do the binary convert
        LUT1 = genLUT(q=cfg['q'], bin_dim=isLUT)
        gallery_feats = tf.cast(gallery_feats, tf.int32)
        LUV = tf.gather(LUT1, gallery_feats)
        gallery_feats = tf.reshape(LUV, (gallery_feats.shape[0], isLUT * gallery_feats.shape[1]))

        probes_feats = tf.cast(probes_feats, tf.int32)
        LUV = tf.gather(LUT1, probes_feats)
        probes_feats = tf.reshape(LUV, (probes_feats.shape[0], isLUT * probes_feats.shape[1]))

        ##### end ########

    mAp = streaming_mean_averge_precision(probes_feats, probes_names, gallery_feats, gallery_names, k=50)
    rr = streaming_mean_cmc_at_k(probes_feats, probes_names, gallery_feats, gallery_names, 10)
    return mAp, rr


if __name__ == '__main__':
    embeddings1 = tf.constant([[1.0, 2, 3], [-1.0, 4, 3], [-9.0, 9, 3]])
    embeddings2 = tf.constant([[-3, 4, 5], [1.0, 3, 2], [1, 0, 2]])

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    print('diff', diff)
    print('dist', dist)
    pd = pdist(embeddings1, embeddings2)
    print(pd)
