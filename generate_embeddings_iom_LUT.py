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
from modules.embedding_util import load_data_from_dir,extractFeat,extractFeatAppend
from modules.LUT import genLUT
from modules.models import build_or_load_IoMmodel

# modules.utils.set_memory_growth()
flags.DEFINE_string('cfg_path', './configs/iom_res50.yaml', 'config file path')
flags.DEFINE_string('ckpt_epoch', '', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_integer('isLUT', 0, 'isLUT length of the look up table entry, 0 mean not use, 3 means 3 bits')
flags.DEFINE_integer('embedding_lfw', 0, 'embedding_lfw, 0 mean not use')
flags.DEFINE_integer('embedding_vgg2', 0, 'embedding_vgg2, 0 mean not use')
flags.DEFINE_integer('embedding_ijbc', 0, 'embedding_ijbc, 0 mean not use')
flags.DEFINE_integer('randomIoM', 0, 'randomIoM, 0 mean not use')

# e.g. python generate_embeddings_iom_LUT.py --cfg_path ./configs/config_18/cfg18_inresv2_512x8.yaml --isLUT 0 --embedding_lfw 1 --embedding_vgg2 1 --randomIoM 0
def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    isLUT = FLAGS.isLUT
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()
    # cfg = load_yaml('./config_arc/arc_lres100ir.yaml')  #
    cfg = load_yaml(FLAGS.cfg_path)
    model = build_or_load_IoMmodel(cfg)
    m = cfg['m']
    q = cfg['q']
    cfg['embd_shape'] = m * q
    embedding_lfw = FLAGS.embedding_lfw
    embedding_vgg2 = FLAGS.embedding_vgg2
    embedding_ijbc = FLAGS.embedding_ijbc # No need, as we use specifical protocol.
    randomIoM = FLAGS.randomIoM
    LUT = None
    if isLUT:
        LUT = genLUT(q=16, bin_dim=isLUT, isPerm=False)

    if embedding_lfw:
        dataset = load_data_from_dir('./data/lfw_mtcnnpy_160', BATCH_SIZE=cfg['eval_batch_size'])
        feats, names, n = extractFeat(dataset, model, isLUT, LUT)
        with open(
                'embeddings_loss/' + cfg['backbone_type'] + '_lfw_feat_randomIoM_' +str(randomIoM) +'_LUT_'+str(isLUT)+'_'+ str(cfg['m']) + 'x' + str(
                    cfg['q']) + '.csv',
                'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerows(feats)
        with open('embeddings_loss/' + cfg['backbone_type'] + '_lfw_name_randomIoM_' +str(randomIoM) +'_LUT_'+str(isLUT)+'_' + str(cfg['m']) + 'x' + str(
                cfg['q']) + '.txt', 'w') as outfile:
            for i in names:
                outfile.write(i + "\n")

    ###########################
    if embedding_vgg2:
        dataset = load_data_from_dir('/media/Storage/facedata/vgg_mtcnnpy_160_shuffled',
                                     BATCH_SIZE=cfg['eval_batch_size'],
                                     img_ext='png', ds='VGG2')
        feats, names, n = extractFeat(dataset, model, isLUT, LUT)
        with open('embeddings_loss/' + cfg['backbone_type'] + '_VGG2_feat_randomIoM_' +str(randomIoM) +'_LUT_'+str(isLUT)+'_'+ str(cfg['m']) + 'x' + str(
                cfg['q']) + '.csv',
                  'w') as f:
            write = csv.writer(f)
            write.writerows(feats)
        with open('embeddings_loss/' + cfg['backbone_type'] + '_VGG2_name_randomIoM_' +str(randomIoM) +'_LUT_'+str(isLUT)+'_'+str(cfg['m']) + 'x' + str(
                cfg['q']) + '.txt', 'w') as outfile:
            for i in names:
                outfile.write(i + "\n")

    if embedding_ijbc:
        ##########################################
        print('[*] IJBC not support LUT!')
        ##########################################
        dataset = load_data_from_dir('/media/Storage/facedata/ijbc_mtcnn_160/images/img',
                                     BATCH_SIZE=cfg['eval_batch_size'],
                                     img_ext='png', ds='IJBC')
        feats1, names1, n = extractFeat(dataset, model, isLUT, LUT)
        dataset2 = load_data_from_dir('/media/Storage/facedata/ijbc_mtcnn_160/images/frames',
                                      BATCH_SIZE=cfg['eval_batch_size'],
                                      img_ext='png', ds='IJBC')
        feats, names, n = extractFeatAppend(dataset2, model, feats1, names1, isLUT, LUT)

        with open('embeddings_loss/' + cfg['backbone_type'] + '_ijbc_feat_randomIoM_' +str(randomIoM) +'_LUT_'+str(isLUT)+str(cfg['m']) + 'x' + str(
                cfg['q']) + '.csv',
                  'w') as f:
            write = csv.writer(f)
            write.writerows(feats)
        with open('embeddings_loss/' + cfg['backbone_type'] + '_ijbc_name_randomIoM_' +str(randomIoM) +'_LUT_'+str(isLUT)+ str(cfg['m']) + 'x' + str(
                cfg['q']) + '.txt', 'w') as outfile:
            for i in names:
                outfile.write(i + "\n")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

