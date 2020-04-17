from absl import app, flags, logging
from absl.flags import FLAGS
# import cv2
import os
import numpy as np
import tensorflow as tf
import modules
import csv

from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel, IoMFaceModelFromArFace
from modules.utils import set_memory_growth, load_yaml, l2_norm


flags.DEFINE_string('cfg_path', './configs/iom_res50_random.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)
    permKey = None
    if cfg['head_type'] == 'IoMHead':#
        #permKey = generatePermKey(cfg['embd_shape'])
        permKey = tf.eye(cfg['embd_shape']) # for training, we don't permutate, won't influence the performance

    model = ArcFaceModel(size=cfg['input_size'],
                         embd_shape=cfg['embd_shape'],
                         backbone_type=cfg['backbone_type'],
                         head_type='ArcHead',
                         training=False,
                         permKey=permKey, cfg=cfg)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_res50' )
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()
    m = cfg['m']
    q = cfg['q']
    model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                   embd_shape=cfg['embd_shape'],
                                   arcmodel=model,
                                   head_type='ArcHead',
                                   training=False,
                                   # here equal false, just get the model without acrHead, to load the model trained by arcface
                                   permKey=permKey, cfg=cfg)
    model.layers[0].trainable = False
    for layer in model.layers:
        print(layer.name)
        layer.trainable = False
    cfg['embd_shape'] = m * q
    if FLAGS.img_path:
        print("[*] Encode {} to ./output_embeds.npy".format(FLAGS.img_path))
        # img = cv2.imread(FLAGS.img_path)
        # img = cv2.resize(img, (cfg['input_size'], cfg['input_size']))
        # img = img.astype(np.float32) / 255.
        # if len(img.shape) == 3:
        #     img = np.expand_dims(img, 0)
        # embeds = l2_norm(model(img))
        # np.save('./output_embeds.npy', embeds)
    else:
        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            get_val_data(cfg['test_dataset'])

        print("[*] Perform Evaluation on LFW...")
        acc_lfw, best_th_lfw, auc_lfw, eer_lfw, embeddings_lfw = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, lfw, lfw_issame,
            is_ccrop=cfg['is_ccrop'], cfg=cfg)
        print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_lfw, best_th_lfw, auc_lfw, eer_lfw))

        print("[*] Perform Evaluation on AgeDB30...")
        acc_agedb30, best_th_agedb30, auc_agedb30, eer_agedb30, embeddings_agedb30 = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, agedb_30,
            agedb_30_issame, is_ccrop=cfg['is_ccrop'], cfg=cfg)
        print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_agedb30, best_th_agedb30, auc_agedb30,
                                                                          eer_agedb30))

        print("[*] Perform Evaluation on CFP-FP...")
        acc_cfp_fp, best_th_cfp_fp, auc_cfp_fp, eer_cfp_fp, embeddings_cfp_fp = perform_val(
            cfg['embd_shape'], cfg['batch_size'], model, cfp_fp, cfp_fp_issame,
            is_ccrop=cfg['is_ccrop'], cfg=cfg)
        print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_cfp_fp, best_th_cfp_fp, auc_cfp_fp,
                                                                          eer_cfp_fp))
        with open('./embeddings/embeddings_lfw.csv', 'w', newline='') as file:
            writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
            writer.writerows(embeddings_lfw)
        with open('./embeddings/embeddings_agedb30.csv', 'w', newline='') as file:
            writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
            writer.writerows(embeddings_agedb30)
        with open('./embeddings/embeddings_cfp_fp.csv', 'w', newline='') as file:
            writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
            writer.writerows(embeddings_cfp_fp)

        print(''' q = {:.2f}, m = {:.2f} | LFW | AgeDB30 | CFP - FP
        --- | --- | --- | ---
        Accuracy | {:.4f} | {:.4f} | {:.4f} 
        EER  | {:.4f} | {:.4f} | {:.4f} 
        AUC  | {:.4f} | {:.4f} | {:.4f} 
        Threshold  | {:.4f} | {:.4f} | {:.4f} '''.format(q, m,
                                                         acc_lfw, acc_agedb30, acc_cfp_fp ,
                                                         eer_lfw, eer_agedb30, eer_cfp_fp ,
                                                         auc_lfw, auc_agedb30, auc_cfp_fp ,
                                                         best_th_lfw, best_th_agedb30,best_th_cfp_fp))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
