import tensorflow as tf
import sys
PAIRWISE_DISTANCES = 'pairwise_distances'
POSITIVE_DISTANCES = 'positive_distances'
NEGATIVE_DISTANCES = 'negative_distances'
POSITIVE_LOSS = 'positive_loss'
NEGATIVE_LOSS = 'negative_loss'

def pairwise_distance(tensor_0):
    g = tf.reduce_sum(tf.square(tensor_0), axis=1, keepdims=True)
    G = tf.matmul(tensor_0, tf.transpose(tensor_0))
    d = g - 2. * G + tf.transpose(g)
    d = tf.cast(tf.sqrt(tf.maximum(d, 0.05)),tf.float64)
    return d


def build_positive_mask(labels):
    mask = tf.equal(labels[:, None], labels[None, :])
    return mask

'''
section 4 in https://arxiv.org/pdf/1706.07567.pdf
Sampling Matters in Deep Embedding Learning
'''
def distance_to_weight(d, n):
    # unlogged:
    # a = d**(n-2)
    # b = (1. - 1/4.*d**2)**((n-3)/2)
    # q = (a*b)**(-1)
    n = tf.cast(n, tf.double)
    a_inv = (2.0 - n) * tf.cast(tf.math.log(d), tf.float64)
    b_inv = -((n - 3) / 2) * tf.cast(tf.math.log(1.0 - 0.25 * (d ** 2.0)), tf.float64)
    log_q = a_inv + b_inv
    max_per_row = tf.reduce_max(
        tf.where(tf.math.is_inf(log_q), tf.zeros_like(log_q), log_q),
        axis=1,
        keepdims=True)

    return log_q - max_per_row


def split_distances_into_positive_and_negative(distances, positive_mask):
    # for summaries
    pos_inds = tf.where(positive_mask)
    neg_inds = tf.where(tf.logical_not(positive_mask))
    pos_ds = tf.gather_nd(distances, pos_inds)*tf.constant(1)
    neg_ds = tf.gather_nd(distances, neg_inds)*tf.constant(1)
    tf.identity(distances, name=PAIRWISE_DISTANCES)
    tf.identity(pos_ds, name=POSITIVE_DISTANCES)
    tf.identity(neg_ds, name=NEGATIVE_DISTANCES)
    tf.summary.scalar(name=NEGATIVE_DISTANCES, tensor=tf.reduce_mean(neg_ds))
    tf.summary.scalar(name=POSITIVE_DISTANCES, tensor=tf.reduce_mean(pos_ds))


def calc_weights(embedding, labels, cutoff):
    n = tf.shape(embedding)[1]
    positive_mask = build_positive_mask(labels)
    pairwise_distances = pairwise_distance(embedding)
    pairwise_distances_for_sample = tf.maximum(pairwise_distances, cutoff)
    log_weights = distance_to_weight(pairwise_distances_for_sample, n)
    return log_weights, positive_mask, pairwise_distances


def zero_non_contributing_examples(log_weights, distances, positive_mask, margins):
    # We only sample negative examples. These will only generate loss
    # if they are within a margin. If not we want to set their probability
    # to zero -> -\infty in log
    negative_inf_array = tf.math.log(tf.zeros_like(log_weights))
    log_weights = tf.where(positive_mask, negative_inf_array, log_weights)
    margins = tf.cast(margins,tf.double)
    log_weights = tf.cast(log_weights,tf.double)
    log_weights = tf.where(
        (distances - margins[:, None]) < 0,
        log_weights,
        negative_inf_array)
    return log_weights


def get_positive_pairs(positive_mask, labels):
    pairs_with_same_label = tf.where(positive_mask)
    not_paired_with_self = tf.not_equal(pairs_with_same_label[:, 0], pairs_with_same_label[:, 1])
    anchor_pos_indices = tf.where(not_paired_with_self)[:, 0]
    anchor_pos_indices = tf.gather(pairs_with_same_label, anchor_pos_indices)
    anchors, positives = anchor_pos_indices[:, 0], anchor_pos_indices[:, 1]
    anchor_labels = tf.gather(labels, anchors)
    return anchors, positives, anchor_labels


def get_negative_pairs(log_weights, count, labels):
    num_examples = tf.shape(count)[0]
    to_sample = count - 1
    most_examples_to_sample = tf.reduce_max(to_sample)
    negative_indices = tf.random.categorical(
        log_weights,
        most_examples_to_sample)

    # we only keep the number of samples we need to sample
    # we also drop indices, which come from rows with all
    # infinite weights
    max_counts = tf.tile(tf.range(0, limit=most_examples_to_sample)[None, :], (num_examples, 1))
    below_count = tf.math.less(max_counts, to_sample[:, None])
    inf_check = tf.math.logical_not(tf.reduce_all(tf.math.is_inf(log_weights), axis=1, keepdims=True))
    to_take = tf.math.logical_and(inf_check, below_count)

    indices_to_take = tf.where(to_take)
    negative_indices = tf.gather_nd(negative_indices, indices_to_take)*tf.constant(1,tf.int64)# here minus 1 to correct the indices
    anchor_indices = indices_to_take[:, 0]
    anchor_labels = tf.gather(labels, anchor_indices)

    return anchor_indices, negative_indices, anchor_labels


def margin_loss(embedding, labels, beta, params,cfg,steps,summary_writer):
    labels = tf.cast(labels, tf.int32)
    # print("labels,",labels)
    beta = tf.gather_nd(beta, labels[:, None])*tf.constant(1.0)
    alpha = params['alpha']
    nu = params['nu']
    cutoff = params['cutoff']
    add_summary = params['add_summary']
    margin = beta + alpha

    log_weights, positive_mask, pairwise_distances = calc_weights(embedding, labels, cutoff)
    log_weights = zero_non_contributing_examples(
        log_weights,
        pairwise_distances,
        positive_mask,
        margin)
    tf.identity(pairwise_distances, PAIRWISE_DISTANCES)

    # work out how many examples of a given label there
    # are and create tensor with that number for each example
    # eg. labels = [0, 0, 0, 1, 1, 2]
    # then count = [3, 3, 3, 2, 2, 1]
    _, info_idx, counts = tf.unique_with_counts(labels)
    count = tf.gather(counts, info_idx)

    # sample indices
    positive_anchor_indices, positive_indices, _ = get_positive_pairs(
        positive_mask,
        labels)
    negative_anchor_indices, negative_indices, _ = get_negative_pairs(
        log_weights,
        count,
        labels)

    # tf.print(
    #     positive_anchor_indices,
    #     [labels],
    #     "summarize=1000",
    #     "message='labels'", output_stream=sys.stdout, sep=',')
    # tf.print(
    #     positive_anchor_indices,
    #     negative_anchor_indices,
    #     "summarize=1000",
    #     "message='neg_a'", output_stream=sys.stdout, sep=',')
    # tf.print(
    #     positive_anchor_indices,
    #     negative_indices,
    #     "summarize=1000",
    #     "message='neg_i'", output_stream=sys.stdout, sep=',')
    # tf.print(
    #     positive_anchor_indices,
    #     [log_weights[0]],
    #     "summarize=1000",
    #     "message='weights'", output_stream=sys.stdout, sep=',')

    # positive loss
    ap = tf.stack([positive_anchor_indices, positive_indices], axis=1)
    d_ap = tf.gather_nd(pairwise_distances, ap)*tf.constant(1.0,tf.float64)
    beta_ap = tf.gather(beta, positive_anchor_indices)
    beta_ap = tf.cast(beta_ap,tf.double)
    d_ap = tf.cast(d_ap,tf.double)
    poss_loss = tf.maximum(alpha + d_ap - beta_ap, 0)
    poss_contribute_pairs = tf.math.count_nonzero(poss_loss, dtype=tf.double)
    poss_loss = tf.reduce_sum(poss_loss)

    # negative
    an = tf.stack([negative_anchor_indices, negative_indices], axis=1)
    # print('*******negative_anchor_indices',negative_anchor_indices)
    # print('*******negative_indices',negative_indices)
    # print('*******an',an)
    # print('*******pairwise_distances',pairwise_distances)
    d_an = tf.gather_nd(pairwise_distances, an)*tf.constant(1.0,tf.float64)
    beta_an = tf.gather(beta, negative_anchor_indices)
    beta_an = tf.cast(beta_an,tf.double)
    d_an = tf.cast(d_an,tf.double)
    neg_loss = tf.maximum(alpha + beta_an - d_an, 0)
    neg_contribute_pairs = tf.math.count_nonzero(neg_loss, dtype=tf.double)
    neg_loss = tf.reduce_sum(neg_loss)

    beta_loss = tf.maximum((tf.reduce_sum(beta_ap) + tf.reduce_sum(beta_an)) * nu, 0.)
    pairs = tf.maximum(poss_contribute_pairs + neg_contribute_pairs, 1.)
    loss = (poss_loss + neg_loss + beta_loss)/pairs
    loss = tf.cast(loss,tf.float32)

    with summary_writer.as_default():
        if add_summary:
            pos_inds = tf.where(positive_mask)
            neg_inds = tf.where(tf.math.logical_not(positive_mask))
            pos_ds = tf.gather_nd(pairwise_distances, pos_inds) * tf.constant(1, tf.float64)
            neg_ds = tf.gather_nd(pairwise_distances, neg_inds) * tf.constant(1, tf.float64)
            tf.summary.scalar('margin/' + NEGATIVE_LOSS, neg_loss / pairs,step=steps)
            tf.summary.scalar('margin/' + POSITIVE_LOSS, poss_loss / pairs, step=steps)
            tf.summary.histogram('margin/' + POSITIVE_DISTANCES, pos_ds, step=steps)
            tf.summary.histogram('margin/' + NEGATIVE_DISTANCES, neg_ds, step=steps)
            tf.summary.scalar('margin/' + 'beta', tf.reduce_mean(beta), step=steps)


    return loss

class MarginLossLayer(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self,num_classes= 85742, params=None,cfg=None, **kwargs):
        super(MarginLossLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.params = params
        self.cfg = cfg
        self.steps=1
        if self.params is None:
            self.params = {
                'alpha': cfg['alpha'],#50 margin of seperate
                'nu': 0.,
                'cutoff': 0.5,
                'add_summary': True,
                'beta_0': cfg['beta_0']# weight
            }
        print("params:",self.params)
        super(MarginLossLayer, self).__init__(**kwargs)
        self.summary_writer = tf.summary.create_file_writer(
        './logs/' + cfg['sub_name']+"/margin/")

    def build(self, input_shape):
        self.betas = self.add_weight(name='beta_margins',
                                 shape=(self.num_classes),
                                 initializer=tf.keras.initializers.Constant(self.params['beta_0'] ),
                                 trainable=True)


    def call(self, embedding, labels):
        loss = margin_loss(embedding,labels, self.betas, self.params,self.cfg,self.steps,self.summary_writer)
        # print('loss',loss)
        self.steps = self.steps+1
        self.add_loss(loss)
        return embedding
