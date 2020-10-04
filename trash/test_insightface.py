from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
import csv
from modules.keras_resnet50 import KitModel

from modules.evaluations import get_val_data, perform_val
from modules.utils import set_memory_growth, load_yaml

flags.DEFINE_string('cfg_path', './config_arc/arc_res50.yaml', 'config file path')
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

    # import converted model
    model = KitModel('pre_models/resnet50/resnet50.npy')
    # model.load_weights('pre_models/resnet50/variables/variables.index')
    for layer in model.layers:
        layer.trainable = False
    model.summary()

    print("[*] Perform Retrieval Evaluation on Y.T.F and F.S...")
    # mAp_ytf, rr_ytf = perform_val_yts(cfg['eval_batch_size'], model, cfg['test_dataset_ytf'],img_ext='jpg')
    # mAp_fs, rr_fs = perform_val_yts(cfg['eval_batch_size'], model, cfg['test_dataset_fs'],img_ext='png')
    # print("    Y.T.F mAP {:.4f}, F.S mAP: {:.2f}".format(mAp_ytf, mAp_fs))
    # print("    Y.T.F CMC-1 {:.4f}, F.S CMC-1: {:.2f}".format(rr_ytf[0], rr_fs[0]))
    mAp_fs = mAp_ytf = 0
    rr_ytf = rr_fs = [0]
    print("[*] Loading LFW, AgeDB30 and CFP-FP...")
    lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
        get_val_data(cfg['test_dataset'])

    print("[*] Perform Evaluation on LFW...")
    acc_lfw, best_th_lfw, auc_lfw, eer_lfw, embeddings_lfw = perform_val(
        cfg['embd_shape'], cfg['batch_size'], model, lfw, lfw_issame,
        is_ccrop=cfg['is_ccrop'], cfg=cfg, measure='Euclidean')
    print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_lfw, best_th_lfw, auc_lfw, eer_lfw))

    with open('embeddings/' + cfg['sub_name'] + '_embeddings_orig_lfw.csv', 'w', newline='') as file:
        writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
        writer.writerows(embeddings_lfw)

    print("[*] Perform Evaluation on AgeDB30...")
    acc_agedb30, best_th_agedb30, auc_agedb30, eer_agedb30, embeddings_agedb30 = perform_val(
        cfg['embd_shape'], cfg['batch_size'], model, agedb_30,
        agedb_30_issame, is_ccrop=cfg['is_ccrop'], cfg=cfg, measure='Euclidean')
    print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_agedb30, best_th_agedb30, auc_agedb30,
                                                                      eer_agedb30))

    print("[*] Perform Evaluation on CFP-FP...")
    acc_cfp_fp, best_th_cfp_fp, auc_cfp_fp, eer_cfp_fp, embeddings_cfp_fp = perform_val(
        cfg['embd_shape'], cfg['batch_size'], model, cfp_fp, cfp_fp_issame,
        is_ccrop=cfg['is_ccrop'], cfg=cfg, measure='Euclidean')
    print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_cfp_fp, best_th_cfp_fp, auc_cfp_fp,
                                                                      eer_cfp_fp))

    log_str = '''| q = {:.2f}, m = {:.2f} | LFW    | AgeDB30 | CFP - FP |
            |------------------------|--------|---------|----------|
            | Accuracy               | {:.4f} | {:.4f}  | {:.4f}   |
            | EER                    | {:.4f} | {:.4f}  | {:.4f}   |
            | AUC                    | {:.4f} | {:.4f}  | {:.4f}   |
            | Threshold              | {:.4f} | {:.4f}  | {:.4f}   |
            |                        | mAP    | CMC-1   |          |
            | Y.T.F                  | {:.4f} | {:.4f}  |          |
            | F.S                    | {:.4f} | {:.4f}  |          | '''.format(0, 0,
                                                                                acc_lfw, acc_agedb30, acc_cfp_fp,
                                                                                eer_lfw, eer_agedb30, eer_cfp_fp,
                                                                                auc_lfw, auc_agedb30, auc_cfp_fp,
                                                                                best_th_lfw, best_th_agedb30,
                                                                                best_th_cfp_fp,
                                                                                mAp_ytf, rr_ytf[0],
                                                                                mAp_fs, rr_fs[0])

    print(log_str)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
