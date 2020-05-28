from absl import app, flags, logging
from absl.flags import FLAGS
# import cv2
import os
import numpy as np
import tensorflow as tf
import modules
import csv

from modules.evaluations import get_val_data, perform_val,perform_val_yts
from modules.models import ArcFaceModel, IoMFaceModelFromArFace
from modules.utils import set_memory_growth, load_yaml, l2_norm


flags.DEFINE_string('cfg_path', './configs/iom_res50_random.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')

# modules.utils.set_memory_growth()

mycfg = {'m':0, 'q': 0}



def callMe():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml('./configs/iom_res50_random.yaml')#    cfg = load_yaml(FLAGS.cfg_path)
    permKey = None
    if cfg['head_type'] == 'IoMHead':#
        #permKey = generatePermKey(cfg['embd_shape'])
        permKey = tf.eye(cfg['embd_shape']) # for training, we don't permutate, won't influence the performance

    arcmodel = ArcFaceModel(size=cfg['input_size'],
                         embd_shape=cfg['embd_shape'],
                         backbone_type=cfg['backbone_type'],
                         head_type='ArcHead',
                         training=False,
                         cfg=cfg)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_res50' )
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        arcmodel.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()
    m = cfg['m'] = mycfg['m']
    q = cfg['q'] = mycfg['q']
    model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                   arcmodel=arcmodel, training=False,
                                   permKey=permKey, cfg=cfg)
    model.summary(line_length=80)
    model.layers[0].trainable = False
    # for layer in model.layers:
    #     print(layer.name)
    #     layer.trainable = False
    cfg['embd_shape'] = m * q
    if False:
        print("[*] Encode {} to ./output_embeds.npy".format(FLAGS.img_path))
        # img = cv2.imread(FLAGS.img_path)
        # img = cv2.resize(img, (cfg['input_size'], cfg['input_size']))
        # img = img.astype(np.float32) / 255.
        # if len(img.shape) == 3:
        #     img = np.expand_dims(img, 0)
        # embeds = l2_norm(model(img))
        # np.save('./output_embeds.npy', embeds)
    else:
        def evl(isLUT):
            print("[*] Perform Retrieval Evaluation on Y.T.F and F.S...")
            mAp_ytf, rr_ytf = perform_val_yts(cfg['eval_batch_size'], model, cfg['test_dataset_ytf'], img_ext='jpg',
                                              isLUT=isLUT,cfg=cfg)
            mAp_fs, rr_fs = perform_val_yts(cfg['eval_batch_size'], model, cfg['test_dataset_fs'], img_ext='png',
                                            isLUT=isLUT,cfg=cfg)
            print("    Y.T.F mAP {:.4f}, F.S mAP: {:.2f}".format(mAp_ytf, mAp_fs))
            print("    Y.T.F CMC-1 {:.4f}, F.S CMC-1: {:.2f}".format(rr_ytf[0], rr_fs[0]))

            print("[*] Loading LFW, AgeDB30 and CFP-FP...")
            lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
                get_val_data(cfg['test_dataset'])

            print("[*] Perform Evaluation on LFW...")
            acc_lfw, best_th_lfw, auc_lfw, eer_lfw, embeddings_lfw = perform_val(
                cfg['embd_shape'], cfg['eval_batch_size'], model, lfw, lfw_issame,
                is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=isLUT)
            print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_lfw, best_th_lfw, auc_lfw, eer_lfw))

            print("[*] Perform Evaluation on AgeDB30...")
            acc_agedb30, best_th_agedb30, auc_agedb30, eer_agedb30, embeddings_agedb30 = perform_val(
                cfg['embd_shape'], cfg['eval_batch_size'], model, agedb_30,
                agedb_30_issame, is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=isLUT)
            print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_agedb30, best_th_agedb30, auc_agedb30,
                                                                              eer_agedb30))

            print("[*] Perform Evaluation on CFP-FP...")
            acc_cfp_fp, best_th_cfp_fp, auc_cfp_fp, eer_cfp_fp, embeddings_cfp_fp = perform_val(
                cfg['embd_shape'], cfg['eval_batch_size'], model, cfp_fp, cfp_fp_issame,
                is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=isLUT)
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
                                                                                           acc_lfw, acc_agedb30,
                                                                                           acc_cfp_fp,
                                                                                           eer_lfw, eer_agedb30,
                                                                                           eer_cfp_fp,
                                                                                           auc_lfw, auc_agedb30,
                                                                                           auc_cfp_fp,
                                                                                           best_th_lfw, best_th_agedb30,
                                                                                           best_th_cfp_fp,
                                                                                           mAp_ytf, rr_ytf[0],
                                                                                           mAp_fs, rr_fs[0])
            with open('./logs/' + cfg['sub_name'] + "_Output.md", "a") as text_file:
                text_file.write(log_str)
            print(log_str)

            log_str2 = '''| q = {:.2f}, m = {:.2f},LUT={}\t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n\n '''.format(
                q, m, isLUT, mAp_ytf, mAp_fs, rr_ytf[0], rr_fs[0], eer_lfw, eer_agedb30, eer_cfp_fp, auc_lfw,
                auc_agedb30, auc_cfp_fp, auc_lfw, auc_agedb30, auc_cfp_fp)

            with open('./logs/' + cfg['sub_name'] + "_Output.md", "a") as text_file:
                text_file.write(log_str2)

        evl(0)# no LUT
        evl(4)
        evl(8)
        evl(16)


# def main(_argv):
#     for m in [32, 64, 128, 256, 512]:
#         for q in [2, 4, 8, 16]:
#             print(m,q,'****')
#             mycfg['m'] = m
#             mycfg['q'] = q
#             app.run(callMe2())

# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass
# 32,64,128,
for m in [512]:
    for q in [ 8, 16]:
        print(m,q,'****')
        mycfg['m'] = m
        mycfg['q'] = q
        callMe()

for m in [1024,2048]:
    for q in [2, 4]:
        print(m,q,'****')
        mycfg['m'] = m
        mycfg['q'] = q
        callMe()