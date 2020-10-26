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
from modules.evaluations import get_val_data, perform_val, perform_val_yts

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

    if cfg['loss_fun'].startswith('margin_loss'):
        model = IoMFaceModelFromArFaceMLossHead(size=cfg['input_size'],
                                                arcmodel=arcmodel, training=True,
                                                permKey=permKey, cfg=cfg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)# seems can only use adam???
        # optimizer = tf.train.MomentumOptimizer(  learning_rate,   momentum=0.9, )
    else:
        # here I add the extra IoM layer and head
        if cfg['hidden_layer_remark'] == '1':
            model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                           arcmodel=arcmodel, training=True,
                                           permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == '2':
            model = IoMFaceModelFromArFace2(size=cfg['input_size'],
                                            arcmodel=arcmodel, training=True,
                                            permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == '3':
            model = IoMFaceModelFromArFace3(size=cfg['input_size'],
                                            arcmodel=arcmodel, training=True,
                                            permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == 'T':
            model = IoMFaceModelFromArFace_T(size=cfg['input_size'],
                                            arcmodel=arcmodel, training=True,
                                            permKey=permKey, cfg=cfg)
        elif cfg['hidden_layer_remark'] == 'T1':# one layer
            model = IoMFaceModelFromArFace_T1(size=cfg['input_size'],
                                            arcmodel=arcmodel, training=True,
                                            permKey=permKey, cfg=cfg)
        else:
            model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                           arcmodel=arcmodel, training=True,
                                           permKey=permKey, cfg=cfg)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9, nesterov=True)# can use adam sgd

    for layer in model.layers:
        if layer.name == 'arcface_model':
            layer.trainable = False
    # ########可训练层
    # model.layers[0].trainable  = True
    for x in model.trainable_weights:
        print("trainable:",x.name)
    print('\n')
    model.summary(line_length=80)

    print("[*] training from scratch.")
    epochs, steps = 1, 1
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
        is_flip = 0
        print('[*] is_flip : {}'.format(is_flip))
        if isLUT == 0 and measure == 'Jaccard':
            isLUT = q

        print("[*] Loading LFW, AgeDB30 and CFP-FP...")
        lfw, agedb_30, cfp_fp, lfw_issame, agedb_30_issame, cfp_fp_issame = \
            get_val_data(cfg['test_dataset'])

        print("[*] Perform Evaluation on LFW...")
        acc_lfw, best_th_lfw, auc_lfw, eer_lfw, embeddings_lfw = perform_val(
            cfg['embd_shape'], cfg['eval_batch_size'], model, lfw, lfw_issame,
            is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=isLUT, measure=measure,is_flip=is_flip)
        print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_lfw, best_th_lfw, auc_lfw, eer_lfw))
        # with open('embeddings/' + cfg['sub_name'] + '_embeddings_lfw.csv', 'w', newline='') as file:
        #     writer = csv.writer(file, escapechar='/', quoting=csv.QUOTE_NONE)
        #     writer.writerows(embeddings_lfw)
        print("[*] Perform Evaluation on AgeDB30...")
        acc_agedb30, best_th_agedb30, auc_agedb30, eer_agedb30, embeddings_agedb30 = perform_val(
            cfg['embd_shape'], cfg['eval_batch_size'], model, agedb_30,
            agedb_30_issame, is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=isLUT, measure=measure,is_flip=is_flip)
        print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_agedb30, best_th_agedb30, auc_agedb30,
                                                                          eer_agedb30))

        print("[*] Perform Evaluation on CFP-FP...")
        acc_cfp_fp, best_th_cfp_fp, auc_cfp_fp, eer_cfp_fp, embeddings_cfp_fp = perform_val(
            cfg['embd_shape'], cfg['eval_batch_size'], model, cfp_fp, cfp_fp_issame,
            is_ccrop=cfg['is_ccrop'], cfg=cfg, isLUT=isLUT, measure=measure,is_flip=is_flip)
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
        # with open('./logs/' + cfg['sub_name'] + "_Output.md", "a") as text_file:
        #     text_file.write(log_str)
        print(log_str)
        log_str2 = '''| q = {:.2f}, m = {:.2f},LUT={},dist={} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n '''.format(
            q, m, isLUT, measure, mAp_ytf, mAp_fs, rr_ytf[0], rr_fs[0], eer_lfw, eer_agedb30, eer_cfp_fp, acc_lfw,
            acc_agedb30, acc_cfp_fp, auc_lfw, auc_agedb30, auc_cfp_fp)
        print(log_str2)


    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        if tf.__version__.startswith('1'):  # important is you want to run with tf1.x,
            summary_writer = tf.contrib.summary.create_file_writer(
                './logs/' + cfg['sub_name'])
        else:
            summary_writer = tf.summary.create_file_writer(
                './logs/' + cfg['sub_name'])
        train_dataset = iter(train_dataset)

        while epochs <= cfg['epochs']:
            if steps % 5 == 0:
                start = time.time()
            if steps % 5000 == 0: #reshuffle and generate every epoch steps % steps_per_epoch == 0
                print('[*] reload DS.')
                train_dataset = dataset_triplet.load_online_pair_wise_dataset(cfg['train_dataset'], ext=cfg['img_ext'],
                                                                              dataset_ext=cfg['dataset_ext'],
                                                                              samples_per_class=cfg[
                                                                                  'samples_per_class'],
                                                                              classes_per_batch=cfg[
                                                                                  'classes_per_batch'], is_ccrop=False)
                train_dataset = iter(train_dataset)
            inputs, labels = next(train_dataset) #print(inputs[0][1][:])  labels[2][:]
            with tf.GradientTape() as tape:
                logist = model((inputs, labels), training=True)
                reg_loss = tf.cast(tf.reduce_sum(model.losses),tf.double)
                quanti_loss = 0.0
                if cfg['quanti']:
                    quanti_loss = tf.cast(loss_fn_quanti(logist),tf.float64)
                code_balance_loss_cal = 0.0
                # for metric learning, we have 1. batch_hard_triplet 2. batch_all_triplet_loss 3. batch_all_arc_triplet_loss
                if cfg['loss_fun'] == 'batch_hard_triplet':
                    pred_loss = triplet_loss_omoindrot.batch_hard_triplet_loss(labels, logist,margin=cfg['triplet_margin'])
                elif cfg['loss_fun'] == 'batch_all_triplet_loss':
                    pred_loss = triplet_loss_omoindrot.batch_all_triplet_loss(labels, logist,margin=cfg['triplet_margin'], scala=100)
                elif cfg['loss_fun'] == 'ms_loss':
                    pred_loss = ms_loss.ms_loss(labels, logist,ms_mining=True)
                elif cfg['loss_fun'] == 'margin_loss':
                    pred_loss = tf.constant(0.0,tf.float64)
                elif cfg['loss_fun'] == 'margin_loss_batch_all_triplet':
                    pred_loss = triplet_loss_omoindrot.batch_all_triplet_loss(labels, logist,margin=cfg['triplet_margin'], scala=100)
                elif cfg['loss_fun'] == 'margin_loss_batch_hard_triplet':
                    pred_loss = triplet_loss_omoindrot.batch_hard_triplet_loss(labels, logist,margin=cfg['triplet_margin'], scala=100)
                elif cfg['loss_fun'] == 'batch_all_arc_triplet_loss':
                    pred_loss, _ ,_= arcface_pair_loss.batch_all_triplet_arcloss(labels, logist, arc_margin=cfg['triplet_margin'], scala=32)
                elif cfg['loss_fun'] == 'batch_hard_arc_triplet_loss':
                    pred_loss = arcface_pair_loss.batch_hard_triplet_arcloss(labels, logist, steps,summary_writer,arc_margin=cfg['triplet_margin'], scala=32)
                elif cfg['loss_fun'] == 'semihard_triplet_loss':
                    pred_loss = triplet_loss.semihard_triplet_loss(labels, logist, margin=cfg['triplet_margin'])
                elif cfg['loss_fun'] == 'triplet_sampling':
                    beta_0 = 1.2
                elif cfg['loss_fun'] == 'only_bin_loss':
                    pred_loss = tf.constant(0.0, tf.float64)
                    reg_loss = tf.constant(0.0, tf.float64)
                    quanti_loss = tf.constant(0.0, tf.float64)
                if cfg['bin_lut_loss']=='tanh':
                    bin_loss = bin_LUT_loss.binary_loss_LUT(labels, logist,q) * 0.001
                elif cfg['bin_lut_loss']=='sigmoid':
                    bin_loss = bin_LUT_loss.binary_loss_LUT_sigmoid(labels, logist,q) * 0.001
                else:
                    bin_loss = tf.constant(0.0,tf.float64)
                if 'code_balance_loss' in cfg:
                    # code_balance_loss_cal_real = code_balance_loss.binary_balance_loss_q(logist, steps, summary_writer, q=cfg['q'])
                    code_balance_loss_cal_real = code_balance_loss.binary_balance_loss_merge(logist, steps, summary_writer, q=cfg['q'])
                    if cfg['code_balance_loss'] :
                        code_balance_loss_cal = code_balance_loss_cal_real

                total_loss = pred_loss + reg_loss * 0.5 + quanti_loss* 0.5 + bin_loss + code_balance_loss_cal

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if steps % 5 == 0:
                end = time.time()
                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}, time per step={:.2f}s, remaining time 4 this epoch={:.2f}min"
                print(verb_str.format(epochs, cfg['epochs'],
                                      steps % steps_per_epoch,
                                      steps_per_epoch,
                                      total_loss.numpy(),
                                      learning_rate.numpy(),end - start,(steps_per_epoch -(steps % steps_per_epoch)) * (end - start) /60.0))


            if steps ==2000:
                evl(0, 'Hamming')
                evl(int(math.log2(q)), 'Hamming')
                break

            steps += 1
            epochs = steps // steps_per_epoch + 1
    else:
        print("[*] only support eager_tf!")



    print("[*] training done!")


if __name__ == '__main__':
    app.run(main)
