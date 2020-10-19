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
from modules.embedding_util import load_data_from_dir

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# modules.utils.set_memory_growth()
flags.DEFINE_string('cfg_path', './configs/iom_res50.yaml', 'config file path')
flags.DEFINE_string('ckpt_epoch', '', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')

def main(_argv):
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    # set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)  # cfg = load_yaml(FLAGS.cfg_path)
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

    arcmodel.summary(line_length=80)

    dataset = load_data_from_dir('./data/lfw_mtcnnpy_160', BATCH_SIZE=cfg['batch_size'], ds='LFW')
    feats, names, n = extractFeat(dataset, arcmodel, 512)
    with open('embeddings/' + cfg['backbone_type'] + '_lfw_feat.csv',
              'w') as f:
        print('embeddings/' + cfg['backbone_type'] + '_lfw_feat.csv')
        write = csv.writer(f)
        write.writerows(feats)

    '''

    For VGG2, we should select and pre-process the vgg dataset first, as the dataset is quite large, we would only select 50 imgs per person

    '''
    #
    # dataset = load_data_from_dir('/media/Storage/facedata/vgg_mtcnnpy_160_shuffled', BATCH_SIZE=cfg['batch_size'], img_ext='png',ds='VGG2')
    # feats, names, n = extractFeat(dataset, arcmodel, 512)
    # with open('embeddings/' + cfg['backbone_type'] + '_VGG2_feat.csv',
    #           'w') as f:
    #     # using csv.writer method from CSV package
    #     print('embeddings/' + cfg['backbone_type'] + '_VGG2_feat.csv')
    #     write = csv.writer(f)
    #     write.writerows(feats)
    # with open('embeddings/' + cfg['backbone_type'] + '_VGG2_name.txt', 'w') as outfile:
    #     for i in names:
    #         outfile.write(i + "\n")
    #
    # '''
    #
    # For IJBC
    #
    # '''
    #
    # dataset = load_data_from_dir('/media/Storage/facedata/ijbc_probe_mtcnnpy_160', BATCH_SIZE=cfg['batch_size'], img_ext='png',ds='IJBC')
    # feats, names, n = extractFeat(dataset, arcmodel, 512)
    # with open('embeddings/' + cfg['backbone_type'] + '_ijbc_feat.csv',
    #           'w') as f:
    #     # using csv.writer method from CSV package
    #     print('embeddings/' + cfg['backbone_type'] + '_ijbc_feat.csv')
    #     write = csv.writer(f)
    #     write.writerows(feats)
    # with open('embeddings/' + cfg['backbone_type'] + '_ijbc_name.txt', 'w') as outfile:
    #     for i in names:
    #         outfile.write(i + "\n")
    #
    # dataset = load_data_from_dir('/media/Storage/facedata/ijbc_g1_mtcnnpy_160', BATCH_SIZE=cfg['batch_size'], img_ext='png',ds='IJBC')
    # feats, names, n = extractFeat(dataset, arcmodel, 512)
    # with open('embeddings/' + cfg['backbone_type'] + '_ijbcg1_feat.csv',
    #           'w') as f:
    #     # using csv.writer method from CSV package
    #     print('embeddings/' + cfg['backbone_type'] + '_ijbcg1_feat.csv')
    #     write = csv.writer(f)
    #     write.writerows(feats)
    # with open('embeddings/' + cfg['backbone_type'] + '_ijbcg1_name.txt', 'w') as outfile:
    #     for i in names:
    #         outfile.write(i + "\n")
    #
    # dataset = load_data_from_dir('/media/Storage/facedata/ijbc_g2_mtcnnpy_160', BATCH_SIZE=cfg['batch_size'], img_ext='png',ds='IJBC')
    # feats, names, n = extractFeat(dataset, arcmodel, 512)
    # with open('embeddings/' + cfg['backbone_type'] + '_ijbcg2_feat.csv',
    #           'w') as f:
    #     # using csv.writer method from CSV package
    #     print('embeddings/' + cfg['backbone_type'] + '_ijbcg2_feat.csv')
    #     write = csv.writer(f)
    #     write.writerows(feats)
    # with open('embeddings/' + cfg['backbone_type'] + '_ijbcg2_name.txt', 'w') as outfile:
    #     for i in names:
    #         outfile.write(i + "\n")



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
