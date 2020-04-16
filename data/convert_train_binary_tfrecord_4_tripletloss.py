from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf


flags.DEFINE_string('dataset_path', '../Dataset/ms1m_align_112/imgs',
                    'path to dataset')
flags.DEFINE_string('output_path', './data/ms1m_triplet.tfrecord',
                    'path to ouput tfrecord')


def pair_parser(imgs):
    # Note y_true shape will be [batch,3]
    return (imgs[0], imgs[1], imgs[2]), ([1, 1, 2])


def processOneDir4(basedir):
    list_ds = tf.data.Dataset.list_files(basedir + "/*.png").shuffle(100).repeat()
    return list_ds


def generateTriplet(imgs, label, dataset='VGG2'):
    if dataset == 'VGG2':
        labels = [int(tf.strings.split(imgs[0], os.path.sep)[0, -2][1:]),
                  int(tf.strings.split(imgs[1], os.path.sep)[0, -2][1:]),
                  int(tf.strings.split(imgs[2], os.path.sep)[0, -2][1:])]
        print("[*] ", labels)
    else:
        labels = [int(tf.strings.split(imgs[0], os.path.sep)[0, -2]), int(tf.strings.split(imgs[1], os.path.sep)[0, -2]),
              int(tf.strings.split(imgs[2], os.path.sep)[0, -2])]
    return (imgs), (labels)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str,source_id, img_path):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id1': _int64_feature(source_id[0]),
               'image/source_id2': _int64_feature(source_id[1]),
               'image/source_id3': _int64_feature(source_id[2]),
               'image/img_path1': _bytes_feature(img_path[0]),
               'image/img_path2': _bytes_feature(img_path[1]),
               'image/img_path3': _bytes_feature(img_path[2]),
               'image/encoded1': _bytes_feature(img_str[0]),
               'image/encoded2': _bytes_feature(img_str[1]),
               'image/encoded3': _bytes_feature(img_str[2])}

    return tf.train.Example(features=tf.train.Features(feature=feature))

def main(_):
    dataset_path = FLAGS.dataset_path

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    samples = []
    logging.info('Reading data list...')
    logging.info('Writing tfrecord file...')
    allsubdir = [os.path.join(dataset_path, o) for o in os.listdir(dataset_path)
                 if os.path.isdir(os.path.join(dataset_path, o))]
    path_ds = tf.data.Dataset.from_tensor_slices(allsubdir)
    ds = path_ds.interleave(lambda x: processOneDir4(x), cycle_length=301, #85742 301
                            block_length=2,
                            num_parallel_calls=4).batch(4, True).map(pair_parser, -1).batch(1, True).map(
        generateTriplet, -1)
    iters = iter(ds)
    cnt=0
    with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
        while cnt<1000000:
            imgs, label = next(iters)
            if imgs[0] != imgs[1]:
                print(cnt)
                print(imgs[0].numpy())
                print(imgs[1].numpy())
                print(imgs[2].numpy())
                # print(imgs[1])
                # print(imgs[2])
                img_str = [open(imgs[0].numpy()[0], 'rb').read(), open(imgs[1].numpy()[0], 'rb').read(), open(imgs[2].numpy()[0], 'rb').read()]
                cnt = cnt + 1
                tf_example = make_example(img_str = img_str,
                                          source_id=label,
                                      img_path=[imgs[0].numpy()[0], imgs[1].numpy()[0], imgs[2].numpy()[0]])
                writer.write(tf_example.SerializeToString())

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
