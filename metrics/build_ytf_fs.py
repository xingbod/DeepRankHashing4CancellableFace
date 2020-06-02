import os
import pathlib
import random
from shutil import copy, rmtree, copytree, copy2
import tqdm


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
    if os.path.exists(save_dir + '/train_gallery'):
        rmtree(save_dir + '/train_gallery')
        rmtree(save_dir + '/test')

    dir_list = [dI for dI in os.listdir(ds_path) if os.path.isdir(os.path.join(ds_path, dI))]
    cnt = 0;
    for dir_name in tqdm.tqdm(dir_list):
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
            dst_dir = os.path.join(save_dir + '/train_gallery', "%05d" % cnt)
            check_folder(dst_dir)
            copy(images, dst_dir)
        for images in test_file_list:
            dst_dir = os.path.join(save_dir + '/test', "%05d" % cnt)
            check_folder(dst_dir)
            copy(images, dst_dir)
        cnt = cnt+1

if __name__ == '__main__':
    ds_path = './data/test_dataset/facescrub_images_112x112/112x112'
    # ds_path = './data/test_dataset/facescrub_mtcnn_160/160x160'
    save_path = './data/test_dataset/facescrub_images_112x112'
    splitDS(ds_path, save_path, SPLIT_WEIGHTS=[120, 5], ds='fs')

    ds_path = './data/test_dataset/aligned_images_DB_YTF/160x160/'
    save_path = './data/test_dataset/aligned_images_DB_YTF'
    splitDS(ds_path, save_path, SPLIT_WEIGHTS=[40, 5], ds='ytf')

    # splitDS(ds_path, save_path, SPLIT_WEIGHTS=[120, 5], ds='fs')
    # dataset = load_data_split(save_path,BATCH_SIZE=16,img_ext='png')
    # for img, label in dataset:
    #     print(label)
    #
