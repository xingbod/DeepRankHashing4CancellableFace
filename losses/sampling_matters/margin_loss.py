import tensorflow as tf

PAIRWISE_DISTANCES = 'pairwise_distances'
POSITIVE_DISTANCES = 'positive_distances'
NEGATIVE_DISTANCES = 'negative_distances'
POSITIVE_LOSS = 'positive_loss'
NEGATIVE_LOSS = 'negative_loss'


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

def pairwise_distance(tensor_0):
    g = tf.reduce_sum(tf.square(tensor_0), axis=1, keepdims=True)
    G = tf.matmul(tensor_0, tf.transpose(tensor_0))
    d = g - 2. * G + tf.transpose(g)
    d = tf.sqrt(tf.maximum(d, 0.05))
    return d


def build_positive_mask(labels):
    mask = tf.equal(labels[:, None], labels[None, :])
    return mask


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


def margin_loss(labels, embedding, beta, params):
    beta = tf.gather_nd(beta, labels[:, None])
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

    # positive_anchor_indices = tf.Print(
    #     positive_anchor_indices,
    #     [labels],
    #     summarize=1000,
    #     message='labels')
    # positive_anchor_indices = tf.Print(
    #     positive_anchor_indices,
    #     [negative_anchor_indices],
    #     summarize=1000,
    #     message='neg_a')
    # positive_anchor_indices = tf.Print(
    #     positive_anchor_indices,
    #     [negative_indices],
    #     summarize=1000,
    #     message='neg_i')
    # positive_anchor_indices = tf.Print(
    #     positive_anchor_indices,
    #     [log_weights[0]],
    #     summarize=1000,
    #     message='weights')

    # positive loss
    ap = tf.stack([positive_anchor_indices, positive_indices], axis=1)
    d_ap = tf.gather_nd(pairwise_distances, ap)
    beta_ap = tf.gather(beta, positive_anchor_indices)
    beta_ap = tf.cast(beta_ap,tf.double)
    d_ap = tf.cast(d_ap,tf.double)
    poss_loss = tf.maximum(alpha + d_ap - beta_ap, 0)
    poss_contribute_pairs = tf.math.count_nonzero(poss_loss, dtype=tf.double)
    poss_loss = tf.reduce_sum(poss_loss)

    # negative
    an = tf.stack([negative_anchor_indices, negative_indices], axis=1)
    d_an = tf.gather_nd(pairwise_distances, an)
    beta_an = tf.gather(beta, negative_anchor_indices)
    beta_an = tf.cast(beta_an,tf.double)
    d_an = tf.cast(d_an,tf.double)
    neg_loss = tf.maximum(alpha + beta_an - d_an, 0)
    neg_contribute_pairs = tf.math.count_nonzero(neg_loss, dtype=tf.double)
    neg_loss = tf.reduce_sum(neg_loss)

    beta_loss = tf.maximum((tf.reduce_sum(beta_ap) + tf.reduce_sum(beta_an)) * nu, 0.)
    pairs = tf.maximum(poss_contribute_pairs + neg_contribute_pairs, 1.)
    loss = (poss_loss + neg_loss + beta_loss)/pairs

    if add_summary:
        pos_inds = tf.where(positive_mask)
        neg_inds = tf.where(tf.math.logical_not(positive_mask))
        pos_ds = tf.gather_nd(pairwise_distances, pos_inds)
        neg_ds = tf.gather_nd(pairwise_distances, neg_inds)
        tf.summary.scalar('margin/' + NEGATIVE_LOSS, neg_loss/pairs)
        tf.summary.scalar('margin/' + POSITIVE_LOSS, poss_loss/pairs)
        tf.summary.histogram('margin/' + POSITIVE_DISTANCES, pos_ds)
        tf.summary.histogram('margin/' + NEGATIVE_DISTANCES, neg_ds)
        tf.summary.scalar('margin/' + 'beta', tf.reduce_mean(beta))

    return loss