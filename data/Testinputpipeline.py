import os
import random
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../')
from data.utils import *
import data.config as config
import matplotlib as plt

# In[ ]:


def main():
    image_size=(112,112)
    #创建graph和model存放目录

    #获取图片地址和类别
    dataset=get_dataset(config.data_dir)
    #划分训练验证集
    if config.validation_set_split_ratio>0.0:
        train_set,val_set=split_dataset(dataset,config.validation_set_split_ratio,config.min_nrof_val_images_per_class)
    else:
        train_set,val_set=dataset,[]
    #训练集的种类数量
    nrof_classes=len(train_set)
    # 获取所有图像位置和相应类别
    image_list, label_list = get_image_paths_and_labels(train_set)
    assert len(image_list) > 0, '训练集不能为空'

    nrof_preprocess_threads = 4
    # 输入队列
    input_queue = tf.queue.FIFOQueue(
        capacity=2000000,
        dtypes=[tf.string, tf.int32],
        shapes=[(1,), (1,)],
        shared_name=None, name=None
    )

    # 获取图像，label的batch形式
    image_batch, label_batch = create_input_pipeline(input_queue,
                                                     image_size,
                                                     nrof_preprocess_threads)
    image_batch = tf.identity(image_batch, 'image_batch')
    image_batch = tf.identity(image_batch, 'input')
    label_batch = tf.identity(label_batch, 'label_batch')

    train_dataset = iter(image_batch)
    for i in range(1):
        imgs, labels = next(train_dataset)
        print(imgs)
        print(labels)
        # fig, axs = plt.subplots(1, 3)
        # axs[0].imshow(imgs[0].numpy().astype('uint8')[0])
        # axs[1].imshow(imgs[1].numpy().astype('uint8')[0])
        # axs[2].imshow(imgs[2].numpy().astype('uint8')[0])
        # plt.show()


# In[2]:


if __name__ == '__main__':
    main()