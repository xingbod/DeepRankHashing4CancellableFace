'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import numpy as np
import tensorflow as tf
from modules.utils import set_memory_growth, load_yaml, l2_norm
from modules.models import ArcFaceModel, IoMFaceModelFromArFace, IoMFaceModelFromArFaceMLossHead,IoMFaceModelFromArFace2,IoMFaceModelFromArFace3,IoMFaceModelFromArFace_T,IoMFaceModelFromArFace_T1
import tqdm
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)
# set_memory_growth()

cfg = load_yaml('./config_arc/arc_res50.yaml')  # cfg = load_yaml(FLAGS.cfg_path)
permKey = None
if cfg['head_type'] == 'IoMHead':  #
    # permKey = generatePermKey(cfg['embd_shape'])
    permKey = tf.eye(cfg['embd_shape'])  # for training, we don't permutate, won't influence the performance

arcmodel = ArcFaceModel(size=cfg['input_size'],
                        embd_shape=cfg['embd_shape'],
                        backbone_type=cfg['backbone_type'],
                        head_type='ArcHead',
                        training=False,
                        cfg=cfg)

ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
if ckpt_path is not None:
    print("[*] load ckpt from {}".format(ckpt_path))
    arcmodel.load_weights(ckpt_path)
else:
    print("[*] Cannot find ckpt from {}.".format(ckpt_path))
    exit()


def load_data_from_dir(save_path, BATCH_SIZE=128, img_ext='png'):
    def transform_test_images(img):
        img = tf.image.resize(img, (112, 112))
        img = img / 255
        return img

    def get_label_withname(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        #         wh = tf.strings.split(parts[-1], ".")[0]
        wh = tf.strings.split(parts[-1], ".")[0]
        return wh

    def process_path_withname(file_path):
        label = get_label_withname(file_path)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = transform_test_images(img)
        return img, label

    list_gallery_ds = tf.data.Dataset.list_files(save_path + '/*/*.' + img_ext, shuffle=False)
    labeled_gallery_ds = list_gallery_ds.map(lambda x: process_path_withname(x))
    dataset = labeled_gallery_ds.batch(BATCH_SIZE)
    return dataset


def extractFeat(dataset, model, feature_dim):
    final_feature = np.zeros(feature_dim)
    feats = []
    names = []
    n = 0
    for image_batch, label_batch in tqdm.tqdm(dataset):
        feature = model(image_batch)
        for i in range(feature.shape[0]):
            n = n + 1
            feats.append(feature[i].numpy())
            mylabel = str(label_batch[i].numpy().decode("utf-8") + "")
            #             print(mylabel)
            names.append(mylabel)

    return feats, names, n

# for q in [2, 4, 8, 16]:
#     m = cfg['m'] = 512
#     q = cfg['q'] = q
#     cfg['hidden_layer_remark'] = '1'
#     # here I add the extra IoM layer and head
#     if cfg['hidden_layer_remark'] == '1':
#         model = IoMFaceModelFromArFace(size=cfg['input_size'],
#                                        arcmodel=arcmodel, training=False,
#                                        permKey=permKey, cfg=cfg)
#     model.summary(line_length=80)
#     cfg['embd_shape'] = m * q
#
#     dataset = load_data_from_dir('./data/lfw_mtcnnpy_160', BATCH_SIZE=128)
#     feats, names, n = extractFeat(dataset, model, m)
#     with open('embeddings/' + cfg['backbone_type'] + '_lfw_feat_dIoM_' + str(cfg['m']) + 'x' + str(cfg['q']) + '.csv',
#               'w') as f:
#         # using csv.writer method from CSV package
#         write = csv.writer(f)
#         write.writerows(feats)

for q in [2, 4, 8, 16]:
    m = cfg['m'] = 512
    q = cfg['q'] = q
    cfg['hidden_layer_remark'] = '1'
    # here I add the extra IoM layer and head
    if cfg['hidden_layer_remark'] == '1':
        model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                       arcmodel=arcmodel, training=False,
                                       permKey=permKey, cfg=cfg)
    model.summary(line_length=80)
    cfg['embd_shape'] = m * q

    dataset = load_data_from_dir('/media/Storage/facedata/vgg_mtcnnpy_160', BATCH_SIZE=128)
    feats, names, n = extractFeat(dataset, model, m)
    with open('embeddings/' + cfg['backbone_type'] + '_VGG2_feat_dIoM_' + str(cfg['m']) + 'x' + str(cfg['q']) + '.csv',
              'w') as f:
        # using csv.writer method from CSV package
        print('embeddings/' + cfg['backbone_type'] + '_VGG2_feat_dIoM_' + str(cfg['m']) + 'x' + str(cfg['q']) + '.csv')
        write = csv.writer(f)
        write.writerows(feats)
    with open('embeddings/' + cfg['backbone_type'] + '_VGG2_name_' + str(cfg['m']) + 'x' + str(
            cfg['q']) + '.txt', 'w') as outfile:
        for i in names:
            outfile.write(i + "\n")