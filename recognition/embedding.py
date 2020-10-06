'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
import argparse
import cv2
import numpy as np
import sys
from skimage import transform as trans

class Embedding:
  def __init__(self, model):
    image_size = (112,112)
    self.image_size = image_size
    self.model = model
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    src[:,0] += 8.0
    self.src = src

  def get(self, rimg, landmark):
    assert landmark.shape[0]==68 or landmark.shape[0]==5
    assert landmark.shape[1]==2
    if landmark.shape[0]==68:
      landmark5 = np.zeros( (5,2), dtype=np.float32 )
      landmark5[0] = (landmark[36]+landmark[39])/2
      landmark5[1] = (landmark[42]+landmark[45])/2
      landmark5[2] = landmark[30]
      landmark5[3] = landmark[48]
      landmark5[4] = landmark[54]
    else:
      landmark5 = landmark
    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, self.src)
    M = tform.params[0:2,:]
    img = cv2.warpAffine(rimg,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_flip = np.fliplr(img)
    # img = np.transpose(img, (2,0,1)) #3*112*112, RGB
    # img_flip = np.transpose(img_flip,(2,0,1))
    input_blob = np.zeros((2,  self.image_size[1], self.image_size[0],3),dtype=np.uint8)

    input_blob[0] = img
    input_blob[1] = img_flip

    input_data = input_blob.astype(np.float32) / 255.

    feat = self.model(input_data).numpy()
    feat = feat.reshape([-1, feat.shape[0] * feat.shape[1]])
    feat = feat.flatten()
    return feat

  def getCropImg(self, rimg, landmark):
    assert landmark.shape[0] == 68 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    if landmark.shape[0] == 68:
      landmark5 = np.zeros((5, 2), dtype=np.float32)
      landmark5[0] = (landmark[36] + landmark[39]) / 2
      landmark5[1] = (landmark[42] + landmark[45]) / 2
      landmark5[2] = landmark[30]
      landmark5[3] = landmark[48]
      landmark5[4] = landmark[54]
    else:
      landmark5 = landmark
    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, self.src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(rimg, M, (self.image_size[1], self.image_size[0]), borderValue=0.0)
    img = img.astype(np.float32) / 255.

    return img


  def getFeat(self, imgs):
    img_flips = []
    # for img in imgs:
    #   img_flip = np.fliplr(img)
    #   img_flips.append(img_flip)

    # input_data = imgs / 255.
    # input_data_flip = img_flips.astype(np.float32) / 255.
    input_data = np.expand_dims(imgs, 0)
    feat = self.model(input_data).numpy()
    # feat_flip = self.model(input_data_flip).numpy()

    # feat = feat.reshape([-1, feat.shape[0] * feat.shape[1]])
    # feat = feat.flatten()
    return feat