'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
import tensorflow as tf
import os
import tqdm
import numpy as np
from modules.LUT import genLUT


def load_data_from_dir(save_path, BATCH_SIZE=128, img_ext='png', ds='LFW'):
    print("now loading " + ds)
    def transform_test_images(img):
        img = tf.image.resize(img, (112, 112))
        img = img / 255
        return img

    def get_label_withname(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        #         wh = tf.strings.split(parts[-1], ".")[0]
        if ds == 'LFW':
            wh = tf.strings.split(parts[-1], ".")[0]
        elif ds == 'VGG2' or ds == 'IJBC_CROP':
            wh = parts[-2]
        elif ds == 'IJBC':
            wh = tf.strings.split(parts[-1], "_")[0]
        else:
            wh = tf.strings.split(parts[-1], ".")[0]
        return wh

    def process_path_withname(file_path):
        label = get_label_withname(file_path)
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = transform_test_images(img)
        return img, label

    if ds == 'LFW':
        list_gallery_ds = tf.data.Dataset.list_files(save_path + '/*/*.' + img_ext, shuffle=False)
    elif ds == 'VGG2':
        list_gallery_ds = tf.data.Dataset.list_files(save_path + '/*/*.' + img_ext, shuffle=False)
    elif ds == 'IJBC':
        list_gallery_ds = tf.data.Dataset.list_files(save_path + '/*.' + img_ext, shuffle=False)
    else:
        list_gallery_ds = tf.data.Dataset.list_files(save_path + '/*/*.' + img_ext, shuffle=False)
    labeled_gallery_ds = list_gallery_ds.map(lambda x: process_path_withname(x))
    dataset = labeled_gallery_ds.batch(BATCH_SIZE)
    return dataset


def extractFeat(dataset, model,isLUT=0,LUT=None):
    feats = []
    names = []
    n = 0
    for image_batch, label_batch in tqdm.tqdm(dataset):
        # print("now is " + str(n))
        feature = model(image_batch)
        for i in range(feature.shape[0]):
            n = n + 1
            feats.append(feature[i].numpy())
            mylabel = str(label_batch[i].numpy().decode("utf-8") + "")
            #             print(mylabel)
            names.append(mylabel)
    print("total images "+ str(n))
    if isLUT:  # length of bin
        # here do the binary convert
        # # here convert the embedding to binary
        # LUT = genLUT(q=16, bin_dim=isLUT, isPerm=False)
        feats = tf.cast(feats, tf.int32)
        LUV = tf.gather(LUT, feats)
        feats = tf.reshape(LUV, (feats.shape[0], isLUT * feats.shape[1]))
        feats = feats.numpy()
    return feats, names, n

def extractFeatAppend(dataset, model,feats,names,isLUT=0,LUT=None):
    n = 0
    for image_batch, label_batch in tqdm.tqdm(dataset):
        # print("now is " + str(n))
        feature = model(image_batch)
        for i in range(feature.shape[0]):
            n = n + 1
            feats.append(feature[i].numpy())
            mylabel = str(label_batch[i].numpy().decode("utf-8") + "")
            #             print(mylabel)
            names.append(mylabel)
    print("total images "+ str(n))
    if isLUT:  # length of bin
        # here do the binary convert
        # # here convert the embedding to binary
        # LUT = genLUT(q=16, bin_dim=isLUT, isPerm=False)
        feats = tf.cast(feats, tf.int32)
        LUV = tf.gather(LUT, feats)
        feats = tf.reshape(LUV, (feats.shape[0], isLUT * feats.shape[1]))
        feats = feats.numpy()
    return feats, names, n