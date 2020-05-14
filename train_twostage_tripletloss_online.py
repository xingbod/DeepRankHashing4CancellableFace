from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf
import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from modules.models import ArcFaceModel,IoMFaceModelFromArFace,IoMFaceModelFromArFaceMLossHead
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf
from losses.euclidan_distance_loss import triplet_loss, triplet_loss_omoindrot
from losses.metric_learning_loss import arcface_pair_loss,ms_loss,bin_LUT_loss
from losses.sampling_matters import margin_loss,triplet_loss_with_sampling
import modules.dataset_triplet as dataset_triplet
from modules.evaluations import val_LFW
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

    arc_ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_res50/')
    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if (not ckpt_path) & (arc_ckpt_path is not None):
        print("[*] load ckpt from {}".format(arc_ckpt_path))
        arcmodel.load_weights(arc_ckpt_path)
        # epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)

    if cfg['loss_fun'] == 'margin_loss':
        model = IoMFaceModelFromArFaceMLossHead(size=cfg['input_size'],
                                                arcmodel=arcmodel, training=True,
                                                permKey=permKey, cfg=cfg)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)# seems can only use adam???
        # optimizer = tf.train.MomentumOptimizer(  learning_rate,   momentum=0.9, )
    else:
        # here I add the extra IoM layer and head
        model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                       arcmodel=arcmodel, training=True,
                                       permKey=permKey, cfg=cfg)

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.9, nesterov=True)# can use adam sgd

    for layer in model.layers:
        if layer.name == 'arcface_model':
            layer.trainable = False
    # 可训练层
    # model.layers[0].trainable  = True
    for x in model.trainable_weights:
        print("trainable:",x.name)
    print('\n')
    model.summary(line_length=80)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1


    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        summary_writer = tf.summary.create_file_writer(
            './logs/' + cfg['sub_name'])

        train_dataset = iter(train_dataset)

        while epochs <= cfg['epochs']:
            if steps % 5 == 0:
                start = time.time()
            if steps % steps_per_epoch == 0: #reshuffle and generate every epoch
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
                # for metric learning, we have 1. batch_hard_triplet 2. batch_all_triplet_loss 3. batch_all_arc_triplet_loss
                if cfg['loss_fun'] == 'batch_hard_triplet':
                    pred_loss = triplet_loss_omoindrot.batch_hard_triplet_loss(labels, logist,margin=cfg['triplet_margin'])
                elif cfg['loss_fun'] == 'batch_all_triplet_loss':
                    pred_loss = triplet_loss_omoindrot.batch_all_triplet_loss(labels, logist,margin=cfg['triplet_margin'], scala=100)
                elif cfg['loss_fun'] == 'ms_loss':
                    pred_loss = ms_loss.ms_loss(labels, logist,ms_mining=True)
                elif cfg['loss_fun'] == 'margin_loss':
                    logist = tf.cast(logist,tf.double)
                    pred_loss = triplet_loss_omoindrot.batch_all_triplet_loss(labels, logist,margin=cfg['triplet_margin'], scala=100)
                    # pred_loss = tf.constant(0.0,tf.float64)
                elif cfg['loss_fun'] == 'batch_all_arc_triplet_loss':
                    pred_loss, _ ,_= arcface_pair_loss.batch_all_triplet_arcloss(labels, logist, arc_margin=cfg['triplet_margin'], scala=32)
                elif cfg['loss_fun'] == 'semihard_triplet_loss':
                    pred_loss = triplet_loss.semihard_triplet_loss(labels, logist, margin=cfg['triplet_margin'])
                elif cfg['loss_fun'] == 'triplet_sampling':
                    beta_0 = 1.2
                quanti_loss = tf.cast(loss_fn_quanti(logist),tf.float64)
                bin_loss = bin_LUT_loss.binary_loss_LUT(labels, logist) * 0.001
                total_loss = pred_loss + reg_loss * 0.5 + quanti_loss * 0.5 + bin_loss

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

                with summary_writer.as_default():
                    tf.summary.scalar(
                        'loss/total loss', total_loss, step=steps)
                    tf.summary.scalar(
                        'loss/pred loss', pred_loss, step=steps)
                    tf.summary.scalar(
                        'loss/reg loss', reg_loss, step=steps)
                    tf.summary.scalar(
                        'loss/quanti loss', quanti_loss, step=steps)
                    tf.summary.scalar(
                        'loss/quanti bin_loss', bin_loss, step=steps)
                    tf.summary.scalar(
                        'learning rate', optimizer.lr, step=steps)

            if steps % cfg['save_steps'] == 0:
                print('[*] save ckpt file!')
                model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
                    cfg['sub_name'], epochs, steps % steps_per_epoch))
            if steps % steps_per_epoch == 0:
                # acc_lfw, best_th_lfw, auc_lfw, eer_lfw, embeddings_lfw = val_LFW(model,cfg)
                # print("    acc {:.4f}, th: {:.2f}, auc {:.4f}, EER {:.4f}".format(acc_lfw, best_th_lfw, auc_lfw, eer_lfw))
                # with summary_writer.as_default():
                #     tf.summary.scalar( 'metric/epoch_acc', acc_lfw, step=epochs)
                #     tf.summary.scalar('metric/epoch_eer', acc_lfw, step=epochs)
                model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
                    cfg['sub_name'], epochs, steps % steps_per_epoch))
            steps += 1
            epochs = steps // steps_per_epoch + 1
    else:
        print("[*] only support eager_tf!")
        model.compile(optimizer=optimizer, loss=None)
        mc_callback = ModelCheckpoint(
            'checkpoints/' + cfg['sub_name'] + '/e_{epoch}_b_{batch}.ckpt',
            save_freq=cfg['save_steps'] * cfg['batch_size'], verbose=1,
            save_weights_only=True)
        tb_callback = TensorBoard(log_dir='logs/'+ cfg['sub_name'],
                                  update_freq=cfg['batch_size'] * 5,
                                  profile_batch=0)
        tb_callback._total_batches_seen = steps
        tb_callback._samples_seen = steps * cfg['batch_size']
        callbacks = [mc_callback, tb_callback]

        def batch_generator(train_dataset):
            train_dataset = iter(train_dataset)
            while True:
                inputs, labels = next(train_dataset) #print(inputs[0][1][:])  labels[2][:]
                yield [inputs, labels]

        model.fit_generator(batch_generator(train_dataset),
                  epochs=cfg['epochs'],
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  initial_epoch=epochs - 1)


    print("[*] training done!")


if __name__ == '__main__':
    app.run(main)
