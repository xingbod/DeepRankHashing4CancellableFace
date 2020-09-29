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
if tf.__version__.startswith('1'):# important is you want to run with tf1.x,
    print('[*] enable eager execution')
    tf.compat.v1.enable_eager_execution()
import modules
import csv
import math

from modules.evaluations import get_val_data, perform_val, perform_val_yts
from modules.models import ArcFaceModel, IoMFaceModelFromArFace, IoMFaceModelFromArFaceMLossHead,IoMFaceModelFromArFace2,IoMFaceModelFromArFace3,IoMFaceModelFromArFace_T,IoMFaceModelFromArFace_T1
from modules.utils import set_memory_growth, load_yaml, l2_norm,generatePermKey

# modules.utils.set_memory_growth()
flags.DEFINE_string('cfg_path', './config_15/cfg15_allloss_res100_512x8.yaml', 'config file path')
flags.DEFINE_string('ckpt_epoch', '', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')


def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)
    permKey = None
    if cfg['head_type'] == 'IoMHead':  #
        permKey = generatePermKey(cfg['embd_shape'])
        # permKey = tf.eye(cfg['embd_shape'])  # for training, we don't permutate, won't influence the performance
    m = cfg['m']
    q = cfg['q']
    arcmodel = ArcFaceModel(size=cfg['input_size'],
                            embd_shape=cfg['embd_shape'],
                            backbone_type=cfg['backbone_type'],
                            head_type='ArcHead',
                            training=False,
                            # here equal false, just get the model without acrHead, to load the model trained by arcface
                            cfg=cfg)
    if cfg['loss_fun'] == 'margin_loss':
        model = IoMFaceModelFromArFaceMLossHead(size=cfg['input_size'],
                                                arcmodel=arcmodel, training=False,
                                                permKey=permKey, cfg=cfg)
    else:
        # here I add the extra IoM layer and head
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
        print("[*] Warning!!!! Cannot find ckpt from {}, random weight of IoM layer will be used.".format(ckpt_path))
        # exit()
    model.summary(line_length=80)
    cfg['embd_shape'] = m * q

    # load img
    img = cv2.imread("./data/demo_img/Bill_Gates_0001.png")
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.
    input_data1 = np.expand_dims(img, 0)

    img = cv2.imread("./data/demo_img/Bill_Gates_0002.png")
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.
    input_data2 = np.expand_dims(img, 0)

    img = cv2.imread("./data/demo_img/Wang_Yi_0001.png")
    img = cv2.resize(img, (112, 112))
    img = img.astype(np.float32) / 255.
    input_data_negative = np.expand_dims(img, 0)

    embedding_1 = model(input_data1)
    embedding_2 = model(input_data2)
    embedding_negative = model(input_data_negative)

    def Hamming(embeddings1, embeddings2):
        diff = np.subtract(embeddings1, embeddings2)
        smstr = np.nonzero(diff)
        dist = np.shape(smstr)[1] / np.shape(embeddings1)[0]
        return dist
    dist_positive = Hamming(embedding_1, embedding_2)
    dist_negative = Hamming(embedding_1, embedding_negative)

    print("dist_positive: {}, dist_negative: {}".format(dist_positive,dist_negative))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
