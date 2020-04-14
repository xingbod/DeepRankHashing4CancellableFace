from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from modules.models import IoMFaceModel, ArcFaceModel
from modules.losses import SoftmaxLoss
from modules.utils import set_memory_growth, load_yaml, get_ckpt_inf,generatePermKey
from losses.angular_margin_loss import arcface_loss,cosface_loss,sphereface_loss
from losses.euclidan_distance_loss import triplet_loss,triplet_loss_vanila,contrastive_loss,triplet_loss_omoindrot
import modules.dataset_triplet as dataset_triplet
import modules
from tensorflow import keras

flags.DEFINE_string('cfg_path', './configs/iom_res50_triplet.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_tf'],
                  'fit: model.fit, eager_tf: custom GradientTape')

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

    model = ArcFaceModel(size=cfg['input_size'],
                         embd_shape=cfg['embd_shape'],
                         backbone_type=cfg['backbone_type'],
                         head_type='ArcHead',
                         training=False, # here equal false, just get the model without acrHead, to load the model trained by arcface
                         permKey=permKey, cfg=cfg)

    if cfg['train_dataset']:
        logging.info("load ms1m dataset.")
        dataset_len = cfg['num_samples']
        steps_per_epoch = dataset_len // cfg['batch_size']
        train_dataset = dataset_triplet.load_tfrecord_dataset(
            cfg['train_dataset'], cfg['batch_size'], cfg['binary_img'],
            is_ccrop=cfg['is_ccrop'])
    else:
        logging.info("load fake dataset.")
        steps_per_epoch = 1

    learning_rate = tf.constant(cfg['base_lr'])
    # optimizer = tf.keras.optimizers.SGD(
    #     learning_rate=learning_rate, momentum=0.9, nesterov=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # loss_fn = SoftmaxLoss() #############################################
    # loss_fn = triplet_loss_vanila.triplet_loss_adapted_from_tf
    loss_fn = triplet_loss.compute_triplet_loss
    loss_fn_quanti = triplet_loss.compute_quanti_loss
    # loss_fn = triplet_loss.hardest_triplet_loss
    # loss_fn = triplet_loss_omoindrot.batch_all_triplet_loss
    # loss_fn = tfa.losses.TripletSemiHardLoss()
    ckpt_path = tf.train.latest_checkpoint('./checkpoints/arc_res50/')
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
        # epochs, steps = get_ckpt_inf(ckpt_path, steps_per_epoch)
    else:
        print("[*] training from scratch.")
        epochs, steps = 1, 1
    epochs, steps = 1, 1
    # here I add the extra IoM layer and head
    m = cfg['m']
    q = cfg['q']
    model = tf.keras.Sequential([
        model,
        tf.keras.layers.Dense(m * q,
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1, seed=None),name='IoMProjection'),
        modules.layers.MaxIndexLinearTraining(units=m * q, q=q)
    ])
    model.layers[0].trainable  = False
    # for layer in model.layers:
    #     print(layer.name)
    #     layer.trainable = False
    # 可训练层
    for x in model.trainable_weights:
        print("trainable:",x.name)
    print('\n')
    model.summary(line_length=80)
    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        summary_writer = tf.summary.create_file_writer(
            './logs/' + cfg['sub_name'])

        train_dataset = iter(train_dataset)

        while epochs <= cfg['epochs']:
            inputs, labels = next(train_dataset) #print(inputs[0][1][:])  labels[2][:]
            tmp = inputs[0]
            shape = tf.shape(tmp) # [  3  66 112 112   3]
            newinput = tf.reshape(tmp, [shape[0] * shape[1], shape[2], shape[3], shape[4]])
            newlabel = tf.reshape(labels, [shape[0] * shape[1]])

            newinput = (newinput,newlabel)
            # if triplet loss
            # if cfg['head_type'] == 'IoMHead':
            #     mask = triplet_loss._get_triplet_mask(newlabel)
            #     mask_tmp = tf.reshape(mask, [tf.size(mask).numpy(), 1])
            #     if len(mask_tmp[mask_tmp])<0.0001*cfg['batch_size']*cfg['batch_size']*cfg['batch_size']:
            #         continue

            with tf.GradientTape() as tape:
                logist = model(newinput, training=True)
                anchor_features = logist[:shape[1]]
                positive_features = logist[shape[1]:shape[1]*2]
                negative_features = logist[shape[1]*2:]
                if cfg['head_type'] == 'IoMHead':
                    reg_loss = tf.cast(tf.reduce_sum(model.losses),tf.double)
                else:
                    reg_loss = tf.reduce_sum(model.losses)
                # pred_loss = loss_fn(newlabel, logist)*50
                pred_loss = loss_fn(anchor_features, positive_features,negative_features)
                quanti_loss = loss_fn_quanti(logist)
                total_loss = pred_loss + reg_loss + quanti_loss*0.5

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if steps % 5 == 0:
                verb_str = "Epoch {}/{}: {}/{}, loss={:.2f}, lr={:.4f}"
                print(verb_str.format(epochs, cfg['epochs'],
                                      steps % steps_per_epoch,
                                      steps_per_epoch,
                                      total_loss.numpy(),
                                      learning_rate.numpy()))

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
                        'learning rate', optimizer.lr, step=steps)

            if steps % cfg['save_steps'] == 0:
                print('[*] save ckpt file!')
                model.save_weights('checkpoints/{}/e_{}_b_{}.ckpt'.format(
                    cfg['sub_name'], epochs, steps % steps_per_epoch))

            steps += 1
            epochs = steps // steps_per_epoch + 1
    else:
        model.compile(optimizer=optimizer, loss=loss_fn)


    print("[*] training done!")


if __name__ == '__main__':
    app.run(main)
