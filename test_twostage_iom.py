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
        elif cfg['hidden_layer_remark'] == 'T':# 2 layers
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
    cfg['embd_shape'] = m * q

    def evl(isLUT, measure):
        # print("[*] Perform Retrieval Evaluation on Y.T.F and F.S...")
        # mAp_ytf, rr_ytf = perform_val_yts(cfg['eval_batch_size'], model, cfg['test_dataset_ytf'], img_ext='jpg',
        #                                   isLUT=isLUT, cfg=cfg)
        # mAp_fs, rr_fs = perform_val_yts(cfg['eval_batch_size'], model, cfg['test_dataset_fs'], img_ext='png',
        #                                 isLUT=isLUT, cfg=cfg)
        # print("    Y.T.F mAP {:.4f}, F.S mAP: {:.2f}".format(mAp_ytf, mAp_fs))
        # print("    Y.T.F CMC-1 {:.4f}, F.S CMC-1: {:.2f}".format(rr_ytf[0], rr_fs[0]))
        mAp_fs = mAp_ytf = 0
        rr_ytf = rr_fs = [0]
        if isLUT == 0 and measure == 'Jaccard':
            isLUT = q

        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            get_val_data(cfg['test_dataset'])

        print("[*] Perform Evaluation on LFW...")
        acc_lfw, best_th_lfw, auc_lfw, eer_lfw, embeddings_lfw = perform_val(
            cfg['embd_shape'], cfg['eval_batch_size'], model, lfw, lfw_issame,
            is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=isLUT, measure=measure)
        print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_lfw, best_th_lfw, auc_lfw, eer_lfw))
        # with open('embeddings/' + cfg['sub_name'] + '_embeddings_lfw.csv', 'w', newline='') as file:
        #     writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
        #     writer.writerows(embeddings_lfw)
        print("[*] Perform Evaluation on AgeDB30...")
        acc_agedb30, best_th_agedb30, auc_agedb30, eer_agedb30, embeddings_agedb30 = perform_val(
            cfg['embd_shape'], cfg['eval_batch_size'], model, agedb_30,
            agedb_30_issame, is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=isLUT, measure=measure)
        print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_agedb30, best_th_agedb30, auc_agedb30,
                                                                          eer_agedb30))

        print("[*] Perform Evaluation on CFP-FP...")
        acc_cfp_fp, best_th_cfp_fp, auc_cfp_fp, eer_cfp_fp, embeddings_cfp_fp = perform_val(
            cfg['embd_shape'], cfg['eval_batch_size'], model, cfp_fp, cfp_fp_issame,
            is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=isLUT, measure=measure)
        print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_cfp_fp, best_th_cfp_fp, auc_cfp_fp,
                                                                          eer_cfp_fp))

        log_str = '''| q = {:.2f}, m = {:.2f},LUT={} | LFW    | AgeDB30 | CFP - FP |
        |------------------------|--------|---------|----------|
        | Accuracy               | {:.4f} | {:.4f}  | {:.4f}   |
        | EER                    | {:.4f} | {:.4f}  | {:.4f}   |
        | AUC                    | {:.4f} | {:.4f}  | {:.4f}   |
        | Threshold              | {:.4f} | {:.4f}  | {:.4f}   |
        |                        | mAP    | CMC-1   |          |
        | Y.T.F                  | {:.4f} | {:.4f}  |          |
        | F.S                    | {:.4f} | {:.4f}  |          | \n\n '''.format(q, m, isLUT,
                                                                                 acc_lfw, acc_agedb30, acc_cfp_fp,
                                                                                 eer_lfw, eer_agedb30, eer_cfp_fp,
                                                                                 auc_lfw, auc_agedb30, auc_cfp_fp,
                                                                                 best_th_lfw, best_th_agedb30,
                                                                                 best_th_cfp_fp,
                                                                                 mAp_ytf, rr_ytf[0],
                                                                                 mAp_fs, rr_fs[0])
        with open('./logs/' + cfg['sub_name'] + "_Output.md", "a") as text_file:
            text_file.write(log_str)
        print(log_str)
        log_str2 = '''| q = {:.2f}, m = {:.2f},LUT={},dist={} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n\n '''.format(
            q, m, isLUT, measure, mAp_ytf, mAp_fs, rr_ytf[0], rr_fs[0], eer_lfw, eer_agedb30, eer_cfp_fp, acc_lfw,
            acc_agedb30, acc_cfp_fp, auc_lfw, auc_agedb30, auc_cfp_fp)
        with open('./logs/' + cfg['sub_name'] + "_Output.md", "a") as text_file:
            text_file.write(log_str2)

        print(log_str2)

    evl(0, 'Hamming')
    evl(3, 'Hamming')
    evl(2, 'Hamming')
    evl(1, 'Hamming')
    # evl(int(math.log2(q)), 'Hamming')
    # evl(0, 'Euclidean')
    # # evl(q, 'Jaccard')
    # # evl(8, 'Hamming')
    # evl(int(math.log2(q)), 'Euclidean')
    # evl(8, 'Jaccard')
    # evl(q, 'Euclidean')
    # evl(8)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
