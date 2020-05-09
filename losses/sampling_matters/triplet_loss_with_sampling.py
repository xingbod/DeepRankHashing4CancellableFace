"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf

PAIRWISE_DISTANCES = 'pairwise_distances'
POSITIVE_DISTANCES = 'positive_distances'
NEGATIVE_DISTANCES = 'negative_distances'
POSITIVE_LOSS = 'positive_loss'
NEGATIVE_LOSS = 'negative_loss'

def pairwise_distance(tensor_0):
    g = tf.reduce_sum(tf.square(tensor_0), axis=1, keepdims=True)
    G = tf.matmul(tensor_0, tf.transpose(tensor_0))
    d = g - 2. * G + tf.transpose(g)
    d = tf.sqrt(tf.maximum(d, 0.05))
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
    a_inv = (2.0 - n) * tf.math.log(d)
    b_inv = -((n - 3) / 2) * tf.math.log(1.0 - 0.25 * (d ** 2.0))
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
    pos_ds = tf.gather_nd(distances, pos_inds)
    neg_ds = tf.gather_nd(distances, neg_inds)
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
    negative_indices = tf.gather_nd(negative_indices, indices_to_take)
    anchor_indices = indices_to_take[:, 0]
    anchor_labels = tf.gather(labels, anchor_indices)

    return anchor_indices, negative_indices, anchor_labels



def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0),tf.double)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def batch_all_triplet_loss(labels, embeddings, margin=1.0, scala=100, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
    assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask,tf.double)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16),tf.double)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss)*scala / (num_positive_triplets + 1e-16)

    # return triplet_loss, fraction_positive_triplets
    # print(fraction_positive_triplets)
    return triplet_loss


def batch_triplet_sampling_loss(labels, embeddings,beta, params,cfg,steps,margin=1.0,squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """

    beta = tf.gather_nd(beta, labels[:, None])
    alpha = params['alpha']
    cutoff = params['cutoff']
    add_summary = params['add_summary']

    log_weights, positive_mask, pairwise_distances = calc_weights(embeddings, labels, cutoff)
    log_weights = zero_non_contributing_examples(
        log_weights,
        pairwise_distances,
        positive_mask,
        beta + alpha)
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

    # positive loss
    ap = tf.stack([positive_anchor_indices, positive_indices], axis=1)
    d_ap = tf.gather_nd(pairwise_distances, ap)
    d_ap = tf.cast(d_ap, tf.double)

    # negative
    an = tf.stack([negative_anchor_indices, negative_indices], axis=1)
    d_an = tf.gather_nd(pairwise_distances, an)
    d_an = tf.cast(d_an, tf.double)

    triplet_loss = tf.maximum(d_ap - d_an + margin, 0.0)

    # Get final mean triplet loss
    triplet_loss = tf.reduce_mean(triplet_loss)

    return triplet_loss, positive_mask,pairwise_distances