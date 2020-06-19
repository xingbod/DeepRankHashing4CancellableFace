from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf
import time
import gc

flags.DEFINE_string('dataset_path', 'G:/facedataset/vggface2_train/vgg_mtcnnpy_160/',
                    'path to dataset')
flags.DEFINE_string('output_path', 'vgg16_binary_triplet.tfrecord',
                    'path to ouput tfrecord')

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def pair_parser(imgs):
    # Note y_true shape will be [batch,3]
    return (imgs[0], imgs[1], imgs[2]), ([1, 1, 2])


def processOneDir4(basedir):
    list_ds = tf.data.Dataset.list_files(basedir + "/*.png").shuffle(100).repeat()
    return list_ds

def generateTriplet(imgs, label, dataset='VGG2'):
    if dataset == 'VGG2':
        tmp1=tf.strings.substr(tf.strings.split(imgs[0], os.path.sep)[0, -2], pos=1, len=6)
        tmp2=tf.strings.substr(tf.strings.split(imgs[1], os.path.sep)[0, -2], pos=1, len=6)
        tmp3=tf.strings.substr(tf.strings.split(imgs[2], os.path.sep)[0, -2], pos=1, len=6)
        labels = [int(tmp1),int(tmp2),int(tmp3)]
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
    ds = path_ds.interleave(lambda x: processOneDir4(x), cycle_length=256, #85742 301
                            block_length=2,
                            num_parallel_calls=-1).batch(4, True).map(pair_parser, -1).batch(1, True).map(
        generateTriplet, -1)
    iters = iter(ds)
    total = 1000000
    with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
        cnt = 0
        while cnt<total:
            if cnt % 5 == 0:
                start = time.time()
            imgs, label = next(iters)
            if imgs[0] != imgs[1]:
                # print(cnt)
                # print(imgs[0].numpy())
                # print(imgs[1].numpy())
                # print(imgs[2].numpy())
                # print(imgs[1])
                # print(imgs[2])
                img_str = [open(imgs[0].numpy()[0], 'rb').read(), open(imgs[1].numpy()[0], 'rb').read(), open(imgs[2].numpy()[0], 'rb').read()]
                cnt = cnt + 1
                tf_example = make_example(img_str=img_str,
                                          source_id=label,
                                      img_path=[imgs[0].numpy()[0], imgs[1].numpy()[0], imgs[2].numpy()[0]])
                writer.write(tf_example.SerializeToString())
                del img_str
            if cnt % 5 == 0:
                end = time.time()
                verb_str = "now={:.2f}, time per step={:.2f}s, remaining time={:.2f}min"
                print(verb_str.format(cnt, end - start,
                                      (total-cnt) * (end - start) / 60.0))
                gc.collect()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
