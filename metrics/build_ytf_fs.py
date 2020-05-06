import os
import pathlib
import random
from shutil import copy, rmtree, copytree, copy2
import tensorflow as tf
# DATASET_NAME = 'facescrub'
DATASET_NAME = 'youtubeface'

if DATASET_NAME == 'youtubeface':
    TRAIN_SET_PATH = '/your/file/dir/to/youtubeface_train_set'
    TEST_SET_PATH = '/your/file/dir/to/youtubeface_test_set'
    NB_CLASSES = 1595
else:
    TRAIN_SET_PATH = '/your/file/dir/to/facescrub_train_set'
    TEST_SET_PATH = '/your/file/dir/to/facescrub_test_set'
    NB_CLASSES = 530

WEIGHTS_SAVE_PATH = 'you/file/dir/to/weights'
WEIGHTS_FILE_NAME = 'best/weights/dir/to/load'

NB_EPOCHS = 200
HASH_NUM = 48
SPLIT_NUM = 4
TOP_K = 50

IMAGE_WIDTH = 112
IMAGE_HEIGHT = 112


def check_path_valid(path):
    return path if path.endswith('/') else path + '/'


def check_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


'''

SPLIT_WEIGHTS_INTRA_ID = (
        40,5, 0)  # train cv val vs test for each identity,  40 5 for Y.T.F jpg | 120 5 for F.S. png
        
'''
def splitDS(ds_path, save_dir, SPLIT_WEIGHTS=[40,5], ds='ytf'):
    dir_list = [dI for dI in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, dI))]
    for dir_name in dir_list:
        data_dir = pathlib.Path(os.path.join(ds_path, dir_name))
        if ds == 'ytf':
            pic_list = list(data_dir.glob('*/*.jpg'))
        else:
            pic_list = list(data_dir.glob('*.png'))
        random.shuffle(pic_list)
        test_file_list = pic_list[0:SPLIT_WEIGHTS[1]]  # random.sample(pic_list, k=SPLIT_WEIGHTS[1])
        if (len(pic_list)<=SPLIT_WEIGHTS[1] ):
            print(data_dir, " has not enough samples, skiped!",len(pic_list))
            continue
        if (len(pic_list) -SPLIT_WEIGHTS[1] < SPLIT_WEIGHTS[0]):
            print(data_dir, " has not enough samples!",len(pic_list))
        train_file_list = random.sample(pic_list[SPLIT_WEIGHTS[1]:], k=min(SPLIT_WEIGHTS[0],len(pic_list)-SPLIT_WEIGHTS[1]))
        for images in train_file_list:
            dst_dir = os.path.join(save_dir + '/train_gallery', dir_name)
            check_folder(dst_dir)
            copy(images, dst_dir)
        for images in test_file_list:
            dst_dir = os.path.join(save_dir + '/test', dir_name)
            check_folder(dst_dir)
            copy(images, dst_dir)

if __name__ == '__main__':
    #ds_path = './data/test_dataset/facescrub_images_112x112/112x112'
    # ds_path = './data/test_dataset/facescrub_mtcnn_160/160x160'
    #save_path = './data/test_dataset/facescrub_images_112x112/'
    #splitDS(ds_path, save_path, SPLIT_WEIGHTS=[120, 5], ds='fs')

    ds_path = './data/test_dataset/aligned_images_DB_YTF/160x160/'
    save_path = './data/test_dataset/aligned_images_DB_YTF/'
    splitDS(ds_path, save_path, SPLIT_WEIGHTS=[40, 5], ds='ytf')

    # splitDS(ds_path, save_path, SPLIT_WEIGHTS=[120, 5], ds='fs')
    # dataset = load_data_split(save_path,BATCH_SIZE=16,img_ext='png')
    # for img, label in dataset:
    #     print(label)
    #
