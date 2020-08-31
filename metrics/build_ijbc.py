'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
from absl import app, flags, logging
import cv2
import numpy as np
from PIL import Image
np.random.seed(123)  # for reproducibility
import concurrent.futures
import os
root_path = '/media/Storage/facedata/ijbc/'
# root_path = '/media/charles/Storage/CropAlignFace/data/IJB-C/'
path_to_frames = root_path + 'images/'
metadata_path = root_path + 'protocols/ijbc_1N_probe_mixed.csv'
# metadata_path = root_path + 'protocols/ijbc_1N_gallery_G1.csv'
# metadata_path = root_path + 'protocols/ijbc_1N_gallery_G2.csv'
save_path = root_path + 'images_cropped/'
nn =0
def to_image(arr):
    if type(arr).__module__ == 'PIL.Image':
        return arr
    if type(arr).__module__ == 'numpy':
        return Image.fromarray(arr)

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
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def process_crop(input):
    (frame_id, frame_data) = input
    print(frame_id)
    x, y, w, h,subject_id = frame_data
    try:
        draw = cv2.cvtColor(cv2.imread(path_to_frames + frame_id), cv2.COLOR_BGR2RGB)
        y = int(y)
        x = int(x)
        w = int(w)
        h = int(h)

        face = draw[y:y + h, x:x + w]
        create_dir(save_path + subject_id+'/')
        cv2.imwrite(save_path + subject_id+'/'+frame_id.split('/')[-2]+frame_id.split('/')[-1], face)
    except Exception as e:
        print(e)




def process_ijbc_frames():

    frames_data = get_groundtruth(metadata_path)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        frames_data = get_groundtruth(metadata_path)
        executor.map(process_crop, frames_data.items())


    print("SUCCESS!!!!!")

def main(_):

    process_ijbc_frames()
    # metadata_path = root_path + 'protocols/ijbc_1N_gallery_G1.csv'
    # process_ijbc_frames(path_to_frames,metadata_path,save_path)
    # metadata_path = root_path + 'protocols/ijbc_1N_gallery_G2.csv'
    # process_ijbc_frames(path_to_frames,metadata_path,save_path)


if __name__ == '__main__':
    app.run(main)
