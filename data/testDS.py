# import tensorflow as tf
# from absl import flags
# import os
# import matplotlib as plt
# FLAGS = flags.FLAGS
# import pathlib
# import tqdm
#
# # h = FcaeRecHelper('data/ms1m_img_ann.npy', [112, 112], 128, use_softmax=False)
#
# mypath = './triplet/'
# train_list =  [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
#
#
# def input_fn():
#     batch_size = 8
#
#     shuffle = True
#     mypath = './triplet/'
#     train_list = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
#     # dirs = list(map(lambda x: os.path.join(x, 'train-*' if self.is_training else 'validation-*')), self.dirs)
#
#     def prefetch_dataset(filename):
#       dataset = tf.data.TFRecordDataset(
#           filename, buffer_size=100)
#       return dataset
#
#     datasets = []
#     for glob in tqdm.tqdm(train_list):
#       dataset = tf.data.Dataset.list_files(glob)
#       dataset = dataset.interleave(
#             prefetch_dataset,
#             cycle_length=2) # if order is important
#       dataset = dataset.shuffle(batch_size, None, True).repeat().prefetch(batch_size)
#       datasets.append(dataset)
#
#     def gen(x):
#       return tf.data.Dataset.range(x,x+1).repeat(2)
#
#     choice = tf.data.Dataset.range(len(datasets)).repeat().flat_map(gen)
#
#     dataset = tf.data.experimental.choose_from_datasets(datasets, choice)
#         # .map( # apply function to each element of the dataset in parallel
#         # self.dataset_parser, num_parallel_calls=-1)
#
#     dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(8)
#
#     return dataset
#
# dataset = input_fn()
# iters = iter(dataset)
# imgs, labels = next(iters)
# print(imgs)

#
# in_hw = [160, 160]
# # print(train_list)
# print(len(train_list))
# img_shape = list(in_hw) + [3]
#
# is_augment = False
# is_normlize = False
#
# # feature = {'image/source_id': _int64_feature(source_id),
# #                'image/filename': _bytes_feature(filename),
# #                'image/encoded': _bytes_feature(img_str)}
#
# def parser(stream: bytes):
#     examples: dict = tf.io.parse_single_example(
#         stream,
#         {'image/encoded': tf.io.FixedLenFeature([], tf.string),
#          'image/source_id': tf.io.FixedLenFeature([], tf.int64)})
#     return tf.image.decode_jpeg(examples['image/encoded'], 3), examples['image/source_id']
#
# def pair_parser(raw_imgs, labels):
#     # imgs do same augment ~
#     if is_augment:
#         1
#         # raw_imgs, _ = augment_img(raw_imgs, None)
#     # normlize image
#     if is_normlize:
#         1
#         # imgs: tf.Tensor = normlize_img(raw_imgs)
#     else:
#         imgs = tf.cast(raw_imgs, tf.float32)
#
#     imgs.set_shape([4] + img_shape)
#     labels.set_shape([4, ])
#     # Note y_true shape will be [batch,3]
#     return (imgs[0], imgs[1], imgs[2]), (labels[:3])
#
# batch_size = 1
# ds = tf.data.Dataset.from_tensor_slices(train_list).interleave(lambda x: tf.data.TFRecordDataset(x)
#                   .shuffle(100)
#                   .repeat(), cycle_length=-1,
#                   block_length=2,
#                   num_parallel_calls=-1).map(parser, -1).batch(4, True).map(pair_parser, -1).batch(8, True)
#
# iters = iter(ds)
# imgs, labels = next(iters)
# print(imgs)
# for i in range(20):
#     imgs, labels = next(iters)
#     fig, axs = plt.subplots(1, 3)
#     axs[0].imshow(imgs[0].numpy().astype('uint8')[0])
#     axs[1].imshow(imgs[1].numpy().astype('uint8')[0])
#     axs[2].imshow(imgs[2].numpy().astype('uint8')[0])
#     plt.show()
#













# def dataset_parser(stream: bytes):
#     examples: dict = tf.io.parse_single_example(
#         stream,
#         {'img': tf.io.FixedLenFeature([], tf.string),
#          'label': tf.io.FixedLenFeature([], tf.int64)})
#     return tf.image.decode_jpeg(examples['img'], 3), examples['label']
# def pair_parser(raw_imgs, labels,is_augment=False,is_normlize = False):
#     # imgs do same augment ~
#     if is_augment:
#         # raw_imgs, _ = h.augment_img(raw_imgs, None)
#         a=1
#     # normlize image
#     if is_normlize:
#         a=1
#         # imgs: tf.Tensor = h.normlize_img(raw_imgs)
#     else:
#         imgs = tf.cast(raw_imgs, tf.float32)
#
#     imgs.set_shape([4] + [ 160 ,160 ,3])
#     labels.set_shape([4, ])
#     # Note y_true shape will be [batch,3]
#     return (imgs[0], imgs[1], imgs[2]), (labels[:3])
# def getfilelist(dirs):
#   Filelist = []
#   for home, dirs, files in os.walk(dirs):
#     for filename in files:
# # 文件名列表，包含完整路径
#       Filelist.append(os.path.join(home, filename))
# # # 文件名列表，只包含文件名
# # Filelist.append( filename)
#   return Filelist
#
# def input_fn( data_dir,batch_size=1,is_training = True ):
#     # batch_size = params['batch_size']
#     dirs = getfilelist(data_dir)
#     print("******************************************")
#     print(dirs)
#
#     dataset = (tf.data.Dataset.from_tensor_slices(dirs).interleave(lambda x: tf.data.TFRecordDataset(x)
#                                  .shuffle(100)
#                                  .repeat(), cycle_length=-1,
#                                  block_length=2,
#                                  num_parallel_calls=-1)
#                .map(dataset_parser, -1)
#                .batch(4, True)
#                .map(pair_parser,  -1)
#                .batch(batch_size, True))  # if order is important
#     return dataset
#
# ds = input_fn("val")
# iters = iter(ds)
# for i in range(20):
#     imgs, labels = next(iters)
#     fig, axs = plt.subplots(1, 3)
#     axs[0].imshow(imgs[0].numpy().astype('uint8')[0])
#     axs[1].imshow(imgs[1].numpy().astype('uint8')[0])
#     axs[2].imshow(imgs[2].numpy().astype('uint8')[0])
#     plt.show()