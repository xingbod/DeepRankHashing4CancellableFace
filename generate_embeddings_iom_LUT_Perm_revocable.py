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
from modules.embedding_util import load_data_from_dir,extractFeat
from modules.LUT import genLUT
from modules.utils import generatePermKey
# modules.utils.set_memory_growth()
flags.DEFINE_string('cfg_path', './configs/iom_res50.yaml', 'config file path')
flags.DEFINE_string('ckpt_epoch', '', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_integer('isLUT', 0, 'isLUT length of the look up table entry, 0 mean not use')

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
    permKey = None
    if cfg['head_type'] == 'IoMHead':  #
        permKey = generatePermKey(cfg['embd_shape'])
        # permKey = tf.eye(cfg['embd_shape'])  # for training, we don't permutate, won't influence the performance

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


    for cnt in range(20):
        m = cfg['m'] = 512
        q = cfg['q'] = 8
        LUT = genLUT(q=3, bin_dim=isLUT, isPerm=True)
        cfg['hidden_layer_remark'] = '1'
        # here I add the extra IoM layer and head
        if cfg['hidden_layer_remark'] == '1':
            model = IoMFaceModelFromArFace(size=cfg['input_size'],
                                           arcmodel=arcmodel, training=False,
                                           permKey=permKey, cfg=cfg)
        model.summary(line_length=80)
        cfg['embd_shape'] = m * q
        ##########################################
        dataset = load_data_from_dir('./data/lfw_mtcnnpy_160', BATCH_SIZE=cfg['eval_batch_size'], ds='LFW')
        feats, names, n = extractFeat(dataset, model, isLUT, LUT)
        with open(
                'embeddings_0831/' + cfg['backbone_type'] + '_lfw_feat_LUT_PERM_' + str(cfg['m']) + 'x' + str(
                    cfg['q']) +'_'+ str(cnt)+ '.csv',
                'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerows(feats)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

