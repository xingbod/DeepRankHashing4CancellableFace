import tensorflow as tf
import os


def _transform_images(is_ccrop=False):
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (128, 128))
        x_train = tf.image.random_crop(x_train, (112, 112, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train
    return transform_images


def _transform_targets(y_train):
    return y_train

def _parse_tfrecord(binary_img=False, is_ccrop=False):
    def parse_tfrecord(tfrecord):
        if binary_img:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/filename': tf.io.FixedLenFeature([], tf.string),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        else:
            features = {'image/source_id1': tf.io.FixedLenFeature([], tf.int64),
               'image/source_id2': tf.io.FixedLenFeature([], tf.int64),
               'image/source_id3': tf.io.FixedLenFeature([], tf.int64),
               'image/img_path1': tf.io.FixedLenFeature([], tf.string),
               'image/img_path2': tf.io.FixedLenFeature([], tf.string),
               'image/img_path3': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            # print("****************************************************")
            # print(x['image/img_path1'])
            # print(x['image/img_path2'])
            # print(x['image/img_path3'])
            image_encoded1 = tf.io.read_file(x['image/img_path1'])
            image_encoded2 = tf.io.read_file(x['image/img_path2'])
            image_encoded3 = tf.io.read_file(x['image/img_path3'])
            x_train1 = tf.image.decode_jpeg(image_encoded1, channels=3)
            x_train2 = tf.image.decode_jpeg(image_encoded2, channels=3)
            x_train3 = tf.image.decode_jpeg(image_encoded3, channels=3)
        y_train = (tf.cast(x['image/source_id1'], tf.float32),tf.cast(x['image/source_id2'], tf.float32),tf.cast(x['image/source_id3'], tf.float32))

        x_train1 = _transform_images(is_ccrop=is_ccrop)(x_train1)
        x_train2 = _transform_images(is_ccrop=is_ccrop)(x_train2)
        x_train3 = _transform_images(is_ccrop=is_ccrop)(x_train3)
        x_train = (x_train1,x_train2,x_train3)

        y_train = _transform_targets(y_train)
        return (x_train, y_train), y_train
    return parse_tfrecord



def load_tfrecord_dataset(tfrecord_name, batch_size,
                          binary_img=False, shuffle=False, buffer_size=10240,
                          is_ccrop=False):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(binary_img=binary_img, is_ccrop=is_ccrop),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(int(batch_size/3))
    dataset = dataset.prefetch(
        # buffer_size=buffer_size)
    buffer_size = tf.data.experimental.AUTOTUNE)
    return dataset

########################################################################################################################
def ListFiles(basedir,ext):
    list_ds = tf.data.Dataset.list_files(basedir+"/*."+ext).repeat()
    return list_ds
def pair_parser(imgs,totalsamples, dataset='VGG2'):
    images = imgs
    labels = []
    if dataset == 'VGG2':
        for i in range(totalsamples):
            labels.append(tf.strings.to_number( tf.strings.substr(tf.strings.split(imgs[None,i], os.path.sep)[0, -2], pos=1, len=6), out_type=tf.dtypes.int32))
    else:
        for i in range(totalsamples):
            labels.append(tf.strings.to_number( tf.strings.split(imgs[None,i],os.path.sep)[0,-2], out_type=tf.dtypes.int32))
    return images,labels

def preprocess_image(image,totalsamples,is_ccrop=False):
    images = []
    for i in range(totalsamples):
        img = tf.image.decode_jpeg(image[i], channels=3)
        img = _transform_images(is_ccrop=is_ccrop)(img)
        images.append(img)
    return images

def load_and_preprocess_image(path,labels,totalsamples,is_ccrop=False):
    image = []
    for i in range(totalsamples):
        image.append(tf.io.read_file(path[i]))
    return preprocess_image(image,totalsamples,is_ccrop=False),labels

def load_online_pair_wise_dataset(dbdir,ext = 'jpg',dataset_ext = 'ms',samples_per_class = 3,classes_per_batch = 4,is_ccrop = False, buffer_size=10240):
    # dataset_ext = 'ms'  # VGG2
    # block_length how many samples per class
    # samples_per_class = 3
    # how many classes for each batch
    # classes_per_batch = 4
    allsubdir = [os.path.join(dbdir, o) for o in os.listdir(dbdir)
                 if os.path.isdir(os.path.join(dbdir, o))]
    path_ds = tf.data.Dataset.from_tensor_slices(allsubdir).repeat()
    ds = path_ds.shuffle(buffer_size).interleave(lambda x: ListFiles(x, ext), cycle_length=4000,# every time, cycle_length classes are taken out to do the maping MS 85742
                                           block_length=samples_per_class,
                                           num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(classes_per_batch * samples_per_class, True).map(
        lambda x: pair_parser(x, classes_per_batch * samples_per_class, dataset=dataset_ext), -1).map(
        lambda path, labels: load_and_preprocess_image(path, labels, classes_per_batch * samples_per_class,is_ccrop=is_ccrop),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    return ds