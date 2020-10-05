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
import tqdm

from modules.evaluations import get_val_data, perform_val, perform_val_yts
from modules.models import ArcFaceModel, IoMFaceModelFromArFace, IoMFaceModelFromArFaceMLossHead,IoMFaceModelFromArFace2,IoMFaceModelFromArFace3,IoMFaceModelFromArFace_T,IoMFaceModelFromArFace_T1
from modules.utils import set_memory_growth, load_yaml, l2_norm
from modules.embedding_util import load_data_from_dir,extractFeat,extractFeatAppend
from modules.LUT import genLUT


# modules.utils.set_memory_growth()
flags.DEFINE_string('cfg_path', './configs/iom_res50.yaml', 'config file path')
flags.DEFINE_string('ckpt_epoch', '', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')
flags.DEFINE_integer('isLUT', 0, 'isLUT length of the look up table entry, 0 mean not use')

root_path = '/media/Storage/facedata/ijbc/'

def get_groundtruth(dataset):
    "{frame_id: [template_id, x, y, w, h]"
    frame_map = {}
    # with open(dataset, 'r', encoding='utf-8') as csvreader:
    with open(dataset, 'r') as csvreader:

        all_data = csvreader.readlines()
        for line in all_data[1:]:
            data = line.strip().split(',')
            template_id, subject_id, frame_name = data[:3]

            x, y, w, h = data[4:]
            # if 'frames' in frame_name:
            if frame_name not in frame_map:
                frame_map[frame_name] = []
            frame_data = [x, y, w, h,subject_id]
            frame_map[frame_name] = frame_data

    return frame_map


def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    isLUT = FLAGS.isLUT

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
        # exit()
    model.summary(line_length=80)
    cfg['embd_shape'] = m * q

    ###########################
    metadata_path = root_path + 'protocols/ijbc_1N_probe_mixed.csv'
    path_to_frames = root_path + 'images/'
    frames_data = get_groundtruth(metadata_path)
    feats = []
    names = []
    for items in tqdm.tqdm(frames_data):
        print(items,'*************')
        frame_id, frame_data = items
        x, y, w, h, subject_id = frame_data
        try:
            draw = cv2.cvtColor(cv2.imread(path_to_frames + frame_id), cv2.COLOR_BGR2RGB)
            y = int(y)
            x = int(x)
            w = int(w)
            h = int(h)
            face = draw[y:y + h, x:x + w]
            img = cv2.resize(face, (112, 112))
            img = img.astype(np.float32) / 255.
            input_data = np.expand_dims(img, 0)
            feat = model(input_data)
            feats.append(feat[0].numpy())
            names.append(subject_id)
        except Exception as e:
            print(e)

    with open('embeddings_dl/' + cfg['backbone_type'] + '_IJBC_1N_probe_feat_dlIoM_' + str(cfg['m']) + 'x' + str(
            cfg['q']) + '.csv',
              'w') as f:
        write = csv.writer(f)
        write.writerows(feats)
    with open('embeddings_dl/' + cfg['backbone_type'] + '_IJBC_1N_probe_name_dl_' + str(cfg['m']) + 'x' + str(
            cfg['q']) + '.txt', 'w') as outfile:
        for i in names:
            outfile.write(i + "\n")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
