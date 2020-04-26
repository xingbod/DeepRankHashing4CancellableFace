
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
    # logits = tf.matmul(embeddings, tf.transpose(embeddings))
    logits = cosine_distance(embeddings, embeddings)
    # print(logits)
    # logits = logits / (tf.reduce_sum(embeddings)*tf.reduce_sum(embeddings))
    # clip logits to prevent zero division when backward
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


def batch_all_triplet_arcloss(labels, embeddings, arc_margin=1.0,scala=20):
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
    print(pairwise_angle)
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
