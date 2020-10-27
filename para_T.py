from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
if tf.__version__.startswith('1'):# important if you want to run with tf1.x,
    print('[*] enable eager execution')
    tf.compat.v1.enable_eager_execution()
import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from modules.models import ArcFaceModel,IoMFaceModelFromArFace,IoMFaceModelFromArFaceMLossHead,IoMFaceModelFromArFace2,IoMFaceModelFromArFace3,IoMFaceModelFromArFace_T,IoMFaceModelFromArFace_T1
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
from losses.euclidan_distance_loss import triplet_loss, triplet_loss_omoindrot
from losses.metric_learning_loss import arcface_pair_loss,ms_loss,bin_LUT_loss,code_balance_loss
from losses.sampling_matters import margin_loss,triplet_loss_with_sampling
import modules.dataset_triplet as dataset_triplet
from modules.evaluations import val_LFW

import matplotlib.pyplot as plt
import numpy as np
import collections

flags.DEFINE_string('cfg_path', './configs/iom_res50_twostage_triplet_online.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_enum('mode', 'eager_tf', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')

# modules.utils.set_memory_growth()


def main(_):
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

    arcmodel = ArcFaceModel(size=cfg['input_size'],
                         embd_shape=cfg['embd_shape'],
                         backbone_type=cfg['backbone_type'],
                         head_type='ArcHead',
                         training=False, # here equal false, just get the model without acrHead, to load the model trained by arcface
                         cfg=cfg)
    if cfg['train_dataset']:
        logging.info("load dataset from "+cfg['train_dataset'])
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']
        train_dataset = dataset_triplet.load_online_pair_wise_dataset(cfg['train_dataset'],ext = cfg['img_ext'],dataset_ext = cfg['dataset_ext'],samples_per_class = cfg['samples_per_class'],classes_per_batch = cfg['classes_per_batch'],is_ccrop = False)
    else:
        logging.info("load fake dataset.")
        steps_per_epoch = 1

    learning_rate = tf.constant(cfg['base_lr'])
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # loss_fn = SoftmaxLoss() #############################################
    loss_fn_quanti = triplet_loss.compute_quanti_loss
    m = cfg['m']
    q = cfg['q']

    if cfg['backbone_type'] == 'ResNet50':
        arc_ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_res50/')
    elif cfg['backbone_type'] == 'InceptionResNetV2':
        arc_ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_InceptionResNetV2/')
    elif cfg['backbone_type'] == 'lresnet100e_ir':
        arc_ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_lresnet100e_ir/')
    elif cfg['backbone_type'] == 'Xception':
        arc_ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_Xception/')
    elif cfg['backbone_type'] == 'VGG19':
        arc_ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_vgg19/')
    elif cfg['backbone_type'] == 'Insight_ResNet100' or cfg['backbone_type'] == 'Insight_ResNet50':
        arc_ckpt_path = None # here we don't have any check point file for this pre_build model, as it is loaded with weights
    else:
        arc_ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_res50/')

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if (not ckpt_path) & (arc_ckpt_path is not None):
        print("[*] load ckpt from {}".format(arc_ckpt_path))
        arcmodel.load_weights(arc_ckpt_path)
        # epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    for T in [1,5,10,100,500,1000]:
        cfg['T'] = T
        model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                       arcmodel=arcmodel, training=True,
                                       permKey=permKey, cfg=cfg)

        acc_lfw, best_th_lfw, auc_lfw, eer_lfw, embeddings_lfw = val_LFW(model, cfg)
        print(
            "    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_lfw, best_th_lfw, auc_lfw, eer_lfw))
        # here we would like to plot the code distribution
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
        plt.savefig(
            'plots/histogram_{}_m{}_q{}_t{}.svg'.format(cfg['sub_name'], cfg['m'], cfg['q'],
                                                        cfg['T']), format='svg')
        plt.close('all')
        print("[*] training done!")



if __name__ == '__main__':
    app.run(main)
