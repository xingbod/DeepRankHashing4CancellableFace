


h = FcaeRecHelper('data/ms1m_img_ann.npy', [112, 112], 128, use_softmax=False)
len(h.train_list)
img_shape = list(h.in_hw) + [3]

is_augment = True
is_normlize = False

def parser(stream: bytes):
    # parser tfrecords
    examples: dict = tf.io.parse_single_example(
        stream,
        {'img': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)})
    return tf.image.decode_jpeg(examples['img'], 3), examples['label']

def pair_parser(raw_imgs, labels):
    # imgs do same augment ~
    if is_augment:
        raw_imgs, _ = h.augment_img(raw_imgs, None)
    # normlize image
    if is_normlize:
        imgs: tf.Tensor = h.normlize_img(raw_imgs)
    else:
        imgs = tf.cast(raw_imgs, tf.float32)

    imgs.set_shape([4] + img_shape)
    labels.set_shape([4, ])
    # Note y_true shape will be [batch,3]
    return (imgs[0], imgs[1], imgs[2]), (labels[:3])

batch_size = 1
# h.train_list : ['a.tfrecords','b.tfrecords','c.tfrecords',...]
ds = (tf.data.Dataset.from_tensor_slices(h.train_list)
        .interleave(lambda x: tf.data.TFRecordDataset(x)
                    .shuffle(100)
                    .repeat(), cycle_length=-1,
                    # block_length = 2 is important
                    block_length=2,
                    num_parallel_calls=-1)
        .map(parser, -1)
        .batch(4, True)
        .map(pair_parser, -1)
        .batch(batch_size, True))

iters = iter(ds)
for i in range(20):
    imgs, labels = next(iters)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(imgs[0].numpy().astype('uint8')[0])
    axs[1].imshow(imgs[1].numpy().astype('uint8')[0])
    axs[2].imshow(imgs[2].numpy().astype('uint8')[0])
    plt.show()