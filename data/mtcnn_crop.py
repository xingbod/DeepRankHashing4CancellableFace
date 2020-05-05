#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2019 Iván de Paz Centeno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
from mtcnn import MTCNN
import glob
import os, sys
import pathlib
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts
#获取指定目录下的所有图片 /test_dataset/aligned_images_DB_YTF/aligned_images_DB/Aaron_Eckhart

detector = MTCNN()
imglist = glob.glob(r"./test_dataset/aligned_images_DB_YTF/aligned_images_DB/*/*/*.jpg")
save_dir = './test_dataset/aligned_images_DB_YTF/aligned_images_DB_112x112/'
# print("*******************",imglist)
for path in tqdm.tqdm(imglist):
    image = cv2.cvtColor(cv2.imread(path),
                         cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)
    if len(result) <=0:
        continue
    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']
    splits = splitall(path)##['.', 'aligned_images_DB_YTF', 'aligned_images_DB', 'Al_Leiter', '4', 'aligned_detect_4.103.jpg']

    s_path = os.path.join(save_dir,splits[-3],splits[-2],splits[-1])
    pathlib.Path(os.path.join(save_dir,splits[-3],splits[-2])).mkdir(parents=True, exist_ok=True)
    crop = image[bounding_box[1]: bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
    dim = (112, 112)
    # resize image
    resized = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(s_path, cv2.cvtColor(resized , cv2.COLOR_RGB2BGR))






# cv2.rectangle(image,
#               (bounding_box[0], bounding_box[1]),
#               (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
#               (0,155,255),
#               0)


#
# cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
# cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
#
# cv2.imwrite("ivan_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#
# print(result)
#
