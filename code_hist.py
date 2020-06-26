'''
Copyright ? 2020 by Xingbo Dong
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
import matplotlib.pyplot as plt
import numpy as np
import collections

from modules.evaluations import get_val_data, perform_val, perform_val_yts
from modules.models import ArcFaceModel, IoMFaceModelFromArFace, IoMFaceModelFromArFaceMLossHead, \
    IoMFaceModelFromArFace2, IoMFaceModelFromArFace3, IoMFaceModelFromArFace_T, IoMFaceModelFromArFace_T1
from modules.utils import set_memory_growth, load_yaml, l2_norm

# modules.utils.set_memory_growth()
flags.DEFINE_string('cfg_path', './configs/iom_res50.yaml', 'config file path')
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
    ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_res50')
    print("[*] load ckpt from {}".format(ckpt_path))
    arcmodel.load_weights(ckpt_path)
    if cfg['loss_fun'] == 'margin_loss':
        model = IoMFaceModelFromArFaceMLossHead(size=cfg['input_size'],
                                                arcmodel=arcmodel, training=False,
                                                permKey=permKey, cfg=cfg)
    else:
        # here I add the extra IoM layer and head
        if cfg['hidden_layer_remark'] == '1':
            model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                           arcmodel=arcmodel, training=False,
                                           permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == '2':
            model = IoMFaceModelFromArFace2(size=cfg['input_size'],
                                            arcmodel=arcmodel, training=False,
                                            permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == '3':
            model = IoMFaceModelFromArFace3(size=cfg['input_size'],
                                            arcmodel=arcmodel, training=False,
                                            permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == 'T':  # 2 layers
            model = IoMFaceModelFromArFace_T(size=cfg['input_size'],
                                             arcmodel=arcmodel, training=False,
                                             permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == 'T1':
            model = IoMFaceModelFromArFace_T1(size=cfg['input_size'],
                                              arcmodel=arcmodel, training=False,
                                              permKey=permKey, cfg=cfg)
        else:
            model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                           arcmodel=arcmodel, training=False,
                                           permKey=permKey, cfg=cfg)
    cfg['embd_shape'] = m * q
    model.summary(line_length=80)
    def evl(isLUT, measure, model, logremark):

        print("[*] Loading LFW, AgeDB30 and CFP-FP...",logremark)
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            get_val_data(cfg['test_dataset'])


        print("[*] Perform Evaluation on LFW...",logremark)
        acc_lfw, best_th_lfw, auc_lfw, eer_lfw, embeddings_lfw = perform_val(
            cfg['embd_shape'], cfg['eval_batch_size'], model, lfw, lfw_issame,
            is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=0, measure=measure)
        print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_lfw, best_th_lfw, auc_lfw, eer_lfw))
        x = np.asarray(embeddings_lfw)
        x = x.astype(int)
        reshaped_array = x.reshape(x.size)
        counter = collections.Counter(reshaped_array)
        x = counter.keys()
        frequency = counter.values()
        y = [x / reshaped_array.size for x in frequency]
        plt.bar(x, y)
        plt.ylabel('Probability')
        plt.xlabel('Code value')
        # plt.show()
        plt.savefig('plots/histogram_'+logremark+'_iom_' + cfg['sub_name']  +'.svg', format='svg')
        plt.close('all')

        with open('embeddings/' +logremark  + cfg['sub_name'] + '_embeddings_lfw.csv', 'w', newline='') as file:
            writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
            writer.writerows(embeddings_lfw)
        #
        # acc_lfw, best_th_lfw, auc_lfw, eer_lfw, embeddings_lfw_bin = perform_val(
        #     cfg['embd_shape'], cfg['eval_batch_size'], model, lfw, lfw_issame,
        #     is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=q, measure=measure)
        # print(" Binary   acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_lfw, best_th_lfw, auc_lfw, eer_lfw))
        #
        # x = np.asarray(embeddings_lfw_bin)
        # x = x.astype(int)
        # reshaped_array = x.reshape(x.size)
        # counter = collections.Counter(reshaped_array)
        # x = counter.keys()
        # frequency = counter.values()
        # y = [x / reshaped_array.size for x in frequency]
        # plt.bar(x, y)
        # plt.ylabel('Probability')
        # plt.xlabel('Code value')
        # plt.savefig('histogram_'+logremark+'_iom_binary_' +  cfg['sub_name'] +'.svg', format='svg')

        # with open('embeddings/' + cfg['sub_name'] + '_embeddings_bin_lfw.csv', 'w', newline='') as file:
        #     writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
        #     writer.writerows(embeddings_lfw_bin)
    evl(q, 'Euclidean',model,'random')

    if cfg['loss_fun'] == 'margin_loss':
        model = IoMFaceModelFromArFaceMLossHead(size=cfg['input_size'],
                                                arcmodel=arcmodel, training=False,
                                                permKey=permKey, cfg=cfg)
    else:
        # here I add the extra IoM layer and head
        if cfg['hidden_layer_remark'] == '1':
            model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                           arcmodel=arcmodel, training=False,
                                           permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == '2':
            model = IoMFaceModelFromArFace2(size=cfg['input_size'],
                                            arcmodel=arcmodel, training=False,
                                            permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == '3':
            model = IoMFaceModelFromArFace3(size=cfg['input_size'],
                                            arcmodel=arcmodel, training=False,
                                            permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == 'T':  # 2 layers
            model = IoMFaceModelFromArFace_T(size=cfg['input_size'],
                                             arcmodel=arcmodel, training=False,
                                             permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == 'T1':
            model = IoMFaceModelFromArFace_T1(size=cfg['input_size'],
                                              arcmodel=arcmodel, training=False,
                                              permKey=permKey, cfg=cfg)
        else:
            model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                           arcmodel=arcmodel, training=False,
                                           permKey=permKey, cfg=cfg)
    cfg['embd_shape'] = m * q
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

    evl(q, 'Euclidean',model,'after_train')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
