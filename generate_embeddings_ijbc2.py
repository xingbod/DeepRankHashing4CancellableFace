from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
if tf.__version__.startswith('1'):# important is you want to run with tf1.x,
    print('[*] enable eager execution')
    tf.compat.v1.enable_eager_execution()
import modules
import csv
import math
import tqdm

from modules.evaluations import get_val_data, perform_val, perform_val_yts
from modules.models import ArcFaceModel, IoMFaceModelFromArFace, IoMFaceModelFromArFaceMLossHead,IoMFaceModelFromArFace2,IoMFaceModelFromArFace3,IoMFaceModelFromArFace_T,IoMFaceModelFromArFace_T1
from modules.utils import set_memory_growth, load_yaml, l2_norm
from modules.embedding_util import load_data_from_dir,extractFeat,extractFeatAppend
from modules.LUT import genLUT


# modules.utils.set_memory_growth()
flags.DEFINE_string('cfg_path', './configs/iom_res50.yaml', 'config file path')
flags.DEFINE_string('ckpt_epoch', '', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_integer('isLUT', 0, 'isLUT length of the look up table entry, 0 mean not use')


def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    isLUT = FLAGS.isLUT

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)
    permKey = None
    if cfg['head_type'] == 'IoMHead':  #
        # permKey = generatePermKey(cfg['embd_shape'])
        permKey = tf.eye(cfg['embd_shape'])  # for training, we don't permutate, won't influence the performance
    m = cfg['m']
    q = cfg['q']
    arcmodel = ArcFaceModel(size=cfg['input_size'],
                            embd_shape=cfg['embd_shape'],
                            backbone_type=cfg['backbone_type'],
                            head_type='ArcHead',
                            training=False,
                            # here equal false, just get the model without acrHead, to load the model trained by arcface
                            cfg=cfg)

    model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                   arcmodel=arcmodel, training=False,
                                   permKey=permKey, cfg=cfg)

    if FLAGS.ckpt_epoch == '':
        ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    else:
        ckpt_path = './checkpoints/' + cfg['sub_name'] + '/' + FLAGS.ckpt_epoch
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        # exit()
    model.summary(line_length=80)
    cfg['embd_shape'] = m * q


    ##########################################
    dataset = load_data_from_dir('/media/Storage/facedata/ijbc/images_cropped_G1', BATCH_SIZE=cfg['eval_batch_size'],
                                 img_ext='png', ds='IJBC_CROP')
    feats1, names1, n = extractFeat(dataset, model)

    dataset2 = load_data_from_dir('/media/Storage/facedata/ijbc/images_cropped_G2', BATCH_SIZE=cfg['eval_batch_size'],
                                 img_ext='png', ds='IJBC_CROP')
    feats2, names2, n = extractFeat(dataset2, model)

    dataset3 = load_data_from_dir('/media/Storage/facedata/ijbc/images_cropped', BATCH_SIZE=cfg['eval_batch_size'],
                                 img_ext='png', ds='IJBC_CROP')
    feats_prob, names_prob, n = extractFeat(dataset3, model)

    with open('embeddings_dl/' + cfg['backbone_type'] + '_ijbc_G1_feat_dlIoM_' + str(cfg['m']) + 'x' + str(
            cfg['q']) + '.csv',
              'w') as f:
        write = csv.writer(f)
        write.writerows(feats1)
    with open('embeddings_dl/' + cfg['backbone_type'] + '_ijbc_G1_name_dl_' + str(cfg['m']) + 'x' + str(
            cfg['q']) + '.txt', 'w') as outfile:
        for i in names1:
            outfile.write(i + "\n")

    with open('embeddings_dl/' + cfg['backbone_type'] + '_ijbc_G2_feat_dlIoM_' + str(cfg['m']) + 'x' + str(
            cfg['q']) + '.csv',
              'w') as f:
        write = csv.writer(f)
        write.writerows(feats2)
    with open('embeddings_dl/' + cfg['backbone_type'] + '_ijbc_G2_name_dl_' + str(cfg['m']) + 'x' + str(
            cfg['q']) + '.txt', 'w') as outfile:
        for i in names2:
            outfile.write(i + "\n")

    with open('embeddings_dl/' + cfg['backbone_type'] + '_ijbc_P_feat_dlIoM_' + str(cfg['m']) + 'x' + str(
            cfg['q']) + '.csv',
              'w') as f:
        write = csv.writer(f)
        write.writerows(feats_prob)
    with open('embeddings_dl/' + cfg['backbone_type'] + '_ijbc_P_name_dl_' + str(cfg['m']) + 'x' + str(
            cfg['q']) + '.txt', 'w') as outfile:
        for i in names_prob:
            outfile.write(i + "\n")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
