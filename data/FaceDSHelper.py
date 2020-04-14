import tensorflow as tf
import matplotlib as plt
import tqdm
import os
import re
import glob
class FaceDSHelper:
    INFO = "INFO:"
    is_augment = False
    is_normlize = False
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.in_hw = [112, 112]


    def build_train_datapipe(self, image_ann_list, batch_size: int,
                             is_augment: bool, is_normlize: bool,
                             is_training: bool) -> tf.data.Dataset:
        print('data augment is ', str(is_augment))
        img_shape = list(self.in_hw) + [3]
        fnames: tf.RaggedTensor = tf.ragged.constant(image_ann_list['fnames'], tf.string)
        labels: tf.RaggedTensor = tf.ragged.constant(image_ann_list['lables'], tf.int64)
        nclass = len(image_ann_list['lables'])
        print("******************8")
        print(fnames)
        print(labels)
        del image_ann_list

        def parser(self, fname: tf.Tensor, label: tf.Tensor):
            fname = tf.reshape(fname, (-1,))
            label = tf.reshape(label, (-1,))[:3]
            fname = tf.add(tf.add(self.dataset_root, '/'), fname)
            raw_imgs0: tf.Tensor = tf.image.decode_jpeg(tf.io.read_file(fname[0]), 3)
            raw_imgs1: tf.Tensor = tf.image.decode_jpeg(tf.io.read_file(fname[1]), 3)
            raw_imgs2: tf.Tensor = tf.image.decode_jpeg(tf.io.read_file(fname[2]), 3)
            # imgs do same augment  imgs do same augment ~
            if self.is_augment:
                raw_imgs0, _ = self.augment_img(raw_imgs0, None)
                raw_imgs1, _ = self.augment_img(raw_imgs1, None)
                raw_imgs2, _ = self.augment_img(raw_imgs2, None)  # normlize image
            if self.is_normlize:
                imgs0: tf.Tensor = self.normlize_img(raw_imgs0)
                imgs1: tf.Tensor = self.normlize_img(raw_imgs1)
                imgs2: tf.Tensor = self.normlize_img(raw_imgs2)
            else:
                imgs0 = tf.cast(raw_imgs0, tf.float32)
                imgs1 = tf.cast(raw_imgs1, tf.float32)
                imgs2 = tf.cast(raw_imgs2, tf.float32)
                imgs0.set_shape(img_shape)
                imgs1.set_shape(img_shape)
                imgs2.set_shape(img_shape)
                label.set_shape([3])  # Note y_true shape will be [batch,3]
            return (imgs0, imgs1, imgs2), (label)

        ds_trans = lambda fname, label: tf.data.Dataset.from_tensor_slices((fname, label)).shuffle(100).repeat()
        ds = tf.data.Dataset.from_tensor_slices((fnames, labels)).interleave(ds_trans, cycle_length=nclass,
                                                                             block_length=2).batch(2, True).shuffle(batch_size * 400).batch(2, True).map(parser, -1).batch(batch_size, True).prefetch(-1)

        return ds

h = FaceDSHelper('./split/')
dataset_path = './split/'
image_ann_list = dict()

sample_id_name = []
sample_fname = []
for id_name in tqdm.tqdm(os.listdir(dataset_path)):
    img_paths = glob.glob(os.path.join(dataset_path, id_name, '*.jpg'))
    sample_id_name.append(int(id_name))
    sample_fname_tmp = []
    for img_path in img_paths:
        sample_fname_tmp.append(img_path)
    sample_fname.append(sample_fname_tmp)

image_ann_list['fnames'] = sample_fname
image_ann_list['lables'] = sample_id_name
dataset = h.build_train_datapipe(image_ann_list, batch_size=8,
                             is_augment=False, is_normlize=False,
                             is_training=True)
iters = iter(dataset)
imgs, labels = next(iters)
print(imgs)

for i in range(20):
    imgs, labels = next(iters)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(imgs[0].numpy().astype('uint8')[0])
    axs[1].imshow(imgs[1].numpy().astype('uint8')[0])
    axs[2].imshow(imgs[2].numpy().astype('uint8')[0])
    plt.show()
