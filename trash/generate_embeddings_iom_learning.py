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
from modules.models import build_or_load_IoMmodel


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
    model = build_or_load_IoMmodel(cfg)
    m = cfg['m']
    q = cfg['q']
    cfg['embd_shape'] = m * q
    embedding_lfw = True
    embedding_vgg2 = True
    embedding_ijbc = False # No need, as we use specifical protocol.
    if embedding_lfw:
        dataset = load_data_from_dir('../data/lfw_mtcnnpy_160', BATCH_SIZE=cfg['eval_batch_size'])
        feats, names, n = extractFeat(dataset, model, )
        with open(
                'embeddings_dl/' + cfg['backbone_type'] + '_lfw_feat_dlIoM_' + str(cfg['m']) + 'x' + str(
                    cfg['q']) + '.csv',
                'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerows(feats)
        with open('embeddings_dl/' + cfg['backbone_type'] + '_lfw_name_dl_' + str(cfg['m']) + 'x' + str(
                cfg['q']) + '.txt', 'w') as outfile:
            for i in names:
                outfile.write(i + "\n")

    ###########################
    if embedding_vgg2:
        dataset = load_data_from_dir('/media/Storage/facedata/vgg_mtcnnpy_160_shuffled',
                                     BATCH_SIZE=cfg['eval_batch_size'],
                                     img_ext='png', ds='VGG2')
        feats, names, n = extractFeat(dataset, model)
        with open('embeddings_dl/' + cfg['backbone_type'] + '_VGG2_feat_dlIoM_' + str(cfg['m']) + 'x' + str(
                cfg['q']) + '.csv',
                  'w') as f:
            # using csv.writer method from CSV package
            print('embeddings_dl/' + cfg['backbone_type'] + '_VGG2_feat_dlIoM_' + str(cfg['m']) + 'x' + str(
                cfg['q']) + '.csv')
            write = csv.writer(f)
            write.writerows(feats)
        with open('embeddings_dl/' + cfg['backbone_type'] + '_VGG2_name_dl_' + str(cfg['m']) + 'x' + str(
                cfg['q']) + '.txt', 'w') as outfile:
            for i in names:
                outfile.write(i + "\n")

    if embedding_ijbc:
        ##########################################
        dataset = load_data_from_dir('/media/Storage/facedata/ijbc_mtcnn_160/images/img',
                                     BATCH_SIZE=cfg['eval_batch_size'],
                                     img_ext='png', ds='IJBC')
        feats1, names1, n = extractFeat(dataset, model)
        dataset2 = load_data_from_dir('/media/Storage/facedata/ijbc_mtcnn_160/images/frames',
                                      BATCH_SIZE=cfg['eval_batch_size'],
                                      img_ext='png', ds='IJBC')
        feats, names, n = extractFeatAppend(dataset2, model, feats1, names1)

        with open('embeddings_dl/' + cfg['backbone_type'] + '_ijbc_feat_dlIoM_' + str(cfg['m']) + 'x' + str(
                cfg['q']) + '.csv',
                  'w') as f:
            # using csv.writer method from CSV package
            print('embeddings_dl/' + cfg['backbone_type'] + '_ijbc_feat_dlIoM_' + str(cfg['m']) + 'x' + str(
                cfg['q']) + '.csv')
            write = csv.writer(f)
            write.writerows(feats)
        with open('embeddings_dl/' + cfg['backbone_type'] + '_ijbc_name_dl_' + str(cfg['m']) + 'x' + str(
                cfg['q']) + '.txt', 'w') as outfile:
            for i in names:
                outfile.write(i + "\n")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
