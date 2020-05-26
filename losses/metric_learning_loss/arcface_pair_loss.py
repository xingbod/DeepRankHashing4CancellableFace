
"""
ArcFace: Additive Angular Margin Loss for Deep Face Recognition (CVPR '19)
Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou
https://arxiv.org/abs/1801.07698

Implementation inspired by repo:
https://github.com/4uiiurz1/keras-arcface

additive angular margin loss:
L = -log(e^(s(cos(theta_{y_i, i}) + m)) / (e^(s(cos(theta_{y_i, i}) + m) + sum(e^(s(cos(theta_{j, i})))))
W = W* / ||W*||
x = x* / ||x*||
cos(theta_{j, i}) = W_j.T * x_i
"""

import tensorflow as tf
import tensorflow.keras.backend as K
PAIRWISE_DISTANCES = 'pairwise_distances'
POSITIVE_DISTANCES = 'positive_distances'
NEGATIVE_DISTANCES = 'negative_distances'
POSITIVE_LOSS = 'positive_loss'
NEGATIVE_LOSS = 'negative_loss'


def cosine_distance(a, b=None):
    """Compute element-wise cosine distance between `a` and `b`.
    Parameters
    ----------
    a : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    b : tf.Tensor
        A matrix of shape NxL with N row-vectors of dimensionality L.
    Returns
    -------
    tf.Tensor
        A matrix of shape NxM where element (i, j) contains the cosine distance
        between elements `a[i]` and `b[j]`.
    """
    a_normed = tf.nn.l2_normalize(a, axis=1)
    b_normed = a_normed if b is None else tf.nn.l2_normalize(b, axis=1)
    return (
        tf.constant(1.0, tf.double) -
        tf.matmul(a_normed, tf.transpose(b_normed)))

def _pairwise_angle(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """

    # get logits from multiplying embeddings (batch_size, embedding_size)
    logits = tf.matmul(embeddings, tf.transpose(embeddings))
    # norm of the vector
    embeddings_norm1 = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1))
    embeddings_norm2 = embeddings_norm1
    logits = tf.math.divide(logits, embeddings_norm1 * embeddings_norm2)
    theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
    return theta


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


def batch_all_triplet_arcloss(labels, embeddings, arc_margin=0,scala=20):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss 0 5.4 -1 are better options for arctriplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_angle = _pairwise_angle(embeddings)
    # print(pairwise_angle)
    # shape (batch_size, batch_size, 1)
    anchor_positive_angle = tf.expand_dims(pairwise_angle, 2)
    assert anchor_positive_angle.shape[2] == 1, "{}".format(anchor_positive_angle.shape)
    # shape (batch_size, 1, batch_size)
    anchor_negative_angle = tf.expand_dims(pairwise_angle, 1)
    assert anchor_negative_angle.shape[1] == 1, "{}".format(anchor_negative_angle.shape)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    # L_{i j}=s_{i j}\left(\frac{\exp (\cos (\theta+m))}{\exp (\cos (\theta+m))+\exp (\sin (\theta))}\right)+(1\left.-s_{i j}\right)\left(\frac{\exp (\sin (\theta+m))}{\exp (\sin (\theta+m))+\exp (\cos (\theta))}\right)

    # triplet_arcloss = tf.math.exp(tf.math.cos(anchor_positive_angle+arc_margin)) / (tf.math.exp(tf.math.cos(anchor_positive_angle+arc_margin))+tf.math.exp(tf.math.sin(anchor_positive_angle))) +  \
    # tf.math.exp(tf.math.sin(anchor_negative_angle+arc_margin)) / \
    # (tf.math.exp(tf.math.sin(anchor_negative_angle+arc_margin))+tf.math.exp(tf.math.cos(anchor_negative_angle)))
    # tmp = anchor_positive_angle
    # anchor_positive_angle = anchor_negative_angle
    # anchor_negative_angle = tmp

    triplet_arcloss_positive = tf.math.exp(tf.math.cos(anchor_negative_angle + arc_margin)) / (
            tf.math.exp(tf.math.cos(anchor_negative_angle + arc_margin)) + tf.math.exp(
        tf.math.sin(anchor_negative_angle)))

    triplet_arcloss_negetive = tf.math.exp(tf.math.sin(anchor_positive_angle + arc_margin)) / \
                      (tf.math.exp(tf.math.sin(anchor_positive_angle + arc_margin)) + tf.math.exp(
                          tf.math.cos(anchor_positive_angle)))

    triplet_arcloss = triplet_arcloss_positive + triplet_arcloss_negetive
    # Put to zero the invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask, tf.double)
    triplet_arcloss = tf.cast(triplet_arcloss, tf.double)
    triplet_arcloss = tf.multiply(mask, triplet_arcloss)

    # Remove negative losses (i.e. the easy triplets)
    triplet_arcloss = tf.maximum(triplet_arcloss, 0.0)

    # Count number of positive triplets (where triplet_arcloss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_arcloss, 1e-16), tf.double)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_arcloss = tf.reduce_sum(triplet_arcloss) * scala / (num_positive_triplets + 1e-16)

    # return triplet_loss, fraction_positive_triplets
    # print(fraction_positive_triplets)
    return triplet_arcloss,tf.reduce_mean(triplet_arcloss_positive) ,tf.reduce_mean(triplet_arcloss_negetive)


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
def batch_hard_triplet_arcloss(labels, embeddings,steps,summary_writer,arc_margin=0,scala=20):
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
    # Get the pairwise distance matrix
    pairwise_angle = _pairwise_angle(embeddings)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    # mask_anchor_positive = tf.to_float(mask_anchor_positive)
    mask_anchor_positive = tf.cast(mask_anchor_positive,tf.float64)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_angle)

    # shape (batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = tf.cast(mask_anchor_negative,tf.float64)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = tf.reduce_max(pairwise_angle, axis=1, keepdims=True)
    anchor_negative_dist = pairwise_angle + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)


    triplet_arcloss_positive = tf.math.exp(tf.math.cos(hardest_negative_dist + arc_margin)) / (
            tf.math.exp(tf.math.cos(hardest_negative_dist + arc_margin)) + tf.math.exp(
        tf.math.sin(hardest_negative_dist)))

    triplet_arcloss_negetive = tf.math.exp(tf.math.sin(hardest_positive_dist + arc_margin)) / \
                      (tf.math.exp(tf.math.sin(hardest_positive_dist + arc_margin)) + tf.math.exp(
                          tf.math.cos(hardest_positive_dist)))
    triplet_arcloss = triplet_arcloss_positive + triplet_arcloss_negetive

    triplet_arcloss = tf.reduce_sum(triplet_arcloss) * scala

    # Get final mean triplet loss
    with summary_writer.as_default():

        tf.summary.scalar('margin/hardest_positive_dist', tf.reduce_mean(hardest_positive_dist), step=steps)
        tf.summary.scalar('margin/hardest_negative_dist', tf.reduce_mean(hardest_negative_dist), step=steps)
        tf.summary.scalar('margin/' + POSITIVE_LOSS, tf.reduce_sum(triplet_arcloss_positive), step=steps)
        tf.summary.scalar('margin/' + NEGATIVE_LOSS, tf.reduce_sum(triplet_arcloss_negetive), step=steps)
        tf.summary.histogram('margin/' + POSITIVE_DISTANCES, anchor_positive_dist, step=steps)
        tf.summary.histogram('margin/' + NEGATIVE_DISTANCES, mask_anchor_negative, step=steps)

    return triplet_arcloss

if __name__ == '__main__':
    embeddings = [[6.0, 4, 7, 5, 4, 6, 4, 5, 0, 2],
                  [1, 4, 0, 5, 1, 4, 5, 5, 5, 2],
                  [6, 0, 2, 0, 2, 3, 5, 3, 4, 4],
                  [6, 0, 0, 0, 3, 6, 5, 1, 4, 3]]
    print(_pairwise_angle(embeddings))