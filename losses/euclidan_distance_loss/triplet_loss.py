import tensorflow as tf


__all__ = [
    "hardest_triplet_loss", "semihard_triplet_loss"
]

"""
FaceNet: A Unified Embedding for Face Recognition and Clustering (CVPR "15)
Florian Schroff, Dmitry Kalenichenko, James Philbin
https://arxiv.org/abs/1503.03832

Implementation inspired by blogpost:
https://omoindrot.github.io/triplet-loss

triplet loss:
L = max(0, ||f(x_a) - f(x_p)||^2 - ||f(x_a) - f(x_n)||^2 + margin)
a = anchor
p = positive (label[a]==label[p])
n = negative (label[a]!=label[n])

easy triplets:
||f(x_a) - f(x_p)||^2 + margin < ||f(x_a) - f(x_n)||^2
positive is closer to anchor than negative by atleast margin
||f(x_a) - f(x_p)||^2 - ||f(x_a) - f(x_n)||^2 + margin = negative value

hard triplets:
||f(x_a) - f(x_n)||^2 < ||f(x_a) - f(x_p)||^2
negative is closer to anchor than positive
||f(x_a) - f(x_p)||^2 - ||f(x_a) - f(x_n)||^2 + margin = postive value

semi-hard triplets:
||f(x_a) - f(x_p)||^2 < ||f(x_a) - f(x_n)||^2 < ||f(x_a) - f(x_p)||^2 + margin
postivie is closer to anchor than negative but negative is within margin
||f(x_a) - f(x_p)||^2 - ||f(x_a) - f(x_n)||^2 + margin = postive value
"""


def pairwise_distances(embeddings):
    # get pariwise-distance matrix
    # p_dist_mat[i, j] = ||f(x_i) - f(x_j)||^2
    #                  = ||f(x_i)||^2  - 2 <f(x_i), f(x_j)> + ||f(x_j)||^2
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    p_dist_mat = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    p_dist_mat = tf.maximum(0.0, p_dist_mat)
    return p_dist_mat


def anchor_positive_mask(labels):
    diag = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    zero_diag = tf.logical_not(diag)
    labels_eq = tf.equal(labels, tf.transpose(labels))
    mask = tf.logical_and(labels_eq, zero_diag)
    return mask


def anchor_negative_mask(labels):
    labels_eq = tf.equal(labels, tf.transpose(labels))
    mask = tf.logical_not(labels_eq)
    return mask


def hardest_triplet_loss(labels, embeddings, margin=1.0):
    # get pairwise distances of embeddings
    pairwise_dist = pairwise_distances(embeddings)

    # get anchors to positives distances
    mask_ap = anchor_positive_mask(labels)
    mask_ap = tf.cast(mask_ap, tf.float32)

    # get hardest positive for each anchor
    ap_dist = tf.multiply(mask_ap, pairwise_dist)
    hardest_ap_dist = tf.reduce_max(ap_dist, axis=1, keepdims=True)

    # get anchors to negatives distances
    mask_an = anchor_negative_mask(labels)
    mask_an = tf.cast(mask_an, tf.float32)

    # add maximum distance to each positive (so we can get mininum negative distance)
    # get hardest negative for each anchor
    max_an_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    an_dist = pairwise_dist + max_an_dist * (1.0 - mask_an)
    hardest_an_dist = tf.reduce_min(an_dist, axis=1, keepdims=True)

    # calculate triplet loss using hardest positive and negatives of each anchor
    triplet_loss = tf.maximum(0.0, hardest_ap_dist - hardest_an_dist + margin)
    triplet_loss = tf.reduce_mean(triplet_loss)
    return triplet_loss


def triplet_mask(labels):
    # get distinct indicies (i!=j, i!=k and j!=k)
    diag = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    zero_diag = tf.logical_not(diag)
    ij_not_eq = tf.expand_dims(zero_diag, 2)
    ik_not_eq = tf.expand_dims(zero_diag, 1)
    jk_not_eq = tf.expand_dims(zero_diag, 0)
    distinct_idxes = tf.logical_and(
        tf.logical_and(ij_not_eq, ik_not_eq), jk_not_eq)

    # get valid label indicies (label[i]==label[j] and label[i]!=label[k])
    labels = tf.expand_dims(labels, 1)
    label_eq = tf.equal(labels, tf.transpose(labels))
    ij_eq = tf.expand_dims(label_eq, 2)
    ik_eq = tf.expand_dims(label_eq, 1)
    ik_not_eq = tf.logical_not(ik_eq)
    valid_label_idxes = tf.logical_and(ij_eq, ik_not_eq)

    # combine distinct indicies and valid label indicies
    mask = tf.logical_and(distinct_idxes, valid_label_idxes)
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

def semihard_triplet_loss(labels, embeddings, margin=1.0):
    # get pairwise distances of embeddings
    # print("*******************")
    # print(labels)
    # print(embeddings)
    pairwise_dist = pairwise_distances(embeddings)

    # build 3d anchor/positive/negative distance tensor
    ap_dist = tf.expand_dims(pairwise_dist, 2)
    an_dist = tf.expand_dims(pairwise_dist, 1)
    triplet_loss = ap_dist - an_dist + margin

    # remove invalid triplets
    # mask = triplet_mask(labels)
    mask = _get_triplet_mask(labels)

    mask = tf.cast(mask, tf.float32)
    triplet_loss = tf.multiply(triplet_loss, tf.cast(mask,tf.double))

    # remove easy triplets (negative losses)
    triplet_loss = tf.maximum(0.0, triplet_loss)

    # get number of non-zero triplets and normalize
    pos_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.double)
    num_pos_triplets = tf.reduce_sum(pos_triplets)
    triplet_loss = tf.cast(tf.reduce_sum(triplet_loss), tf.double) / (num_pos_triplets + 1e-16)
    return triplet_loss
