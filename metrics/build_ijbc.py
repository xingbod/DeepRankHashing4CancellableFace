'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from absl import app, flags, logging
import tensorflow as tf
import keras
import cv2
import numpy as np
import sys, pandas as pd
from PIL import Image
np.random.seed(123)  # for reproducibility
from mtcnn import MTCNN

mtcnn = MTCNN()
mtcnn.share_memory()


def to_image(arr):
    if type(arr).__module__ == 'PIL.Image':
        return arr
    if type(arr).__module__ == 'numpy':
        return Image.fromarray(arr)

def alignface(img1, ):
    img1 = Image.fromarray(img1)
    try:
        face1 = mtcnn.align_best(img1, limit=10, min_face_size=16, )
        face1 = np.asarray(face1)
        return face1, True

    except:
        logging.info(f'fail !! {img1}')
        face1 = to_image(img1).resize((112, 112), Image.BILINEAR)
        face1 = np.asarray(face1)
        return face1, False

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
            frame_data = [x, y, w, h]
            frame_map[frame_name] = frame_data

    return frame_map

def extract_facial_features_vggface_frames():

    path_to_frames = '/home/datasets/images/IJB/IJB-C/images/'
    metadata_path = '/home/datasets/images/IJB/IJB-C/protocols/ijbc_1N_probe_mixed.csv'
    save_path = '/home/datasets/images/IJB/IJB-C/images_cropped/'

    frames_data = get_groundtruth(metadata_path)

    for frame_id, frame_data in frames_data.items():
        print(frame_id)
        x, y, w, h = frame_data

        try:
            draw = cv2.imread(path_to_frames + frame_id)
        except Exception as e:
            print(e)
            continue

        y = int(y)
        x = int(x)
        w = int(w)
        h = int(h)

        face = draw[y:y + h, x:x + w]
        alignface,isSuccess = alignface(face)
        cv2.imwrite(save_path+frame_id, alignface)

    print("SUCCESS!!!!!")