'''

Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning
@inproceedings{wang2019multi,
  title={Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning},
  author={Wang, Xun and Han, Xintong and Huang, Weilin and Dong, Dengke and Scott, Matthew R},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={5022--5030},
  year={2019}
}

$L_{M S}=\frac{1}{m} \sum_{i=1}^{m}\left\{\frac{1}{\alpha} \log \left[1+\sum_{k \in P_{i}} e^{-\alpha\left(S_{i k}-\lambda\right)}\right]+\frac{1}{\beta} \log \left[1+\sum_{k \in N_{i}} e^{\beta\left(S_{i k}-\lambda\right)}\right]\right\}$

'''
import os
import sys
import numpy as np
import tensorflow as tf

def ms_loss(labels, embeddings, alpha=2.0, beta=50.0, lamb=1.0, eps=0.1, ms_mining=False,scala=100):
    '''
    ref: http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf
    official codes: https://github.com/MalongTech/research-ms-loss
    '''
    # make sure emebedding should be l2-normalized
    # embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    labels = tf.reshape(labels, [-1, 1])

    batch_size = embeddings.get_shape().as_list()[0]

    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)

    mask_pos = tf.cast(adjacency, dtype=tf.double) - tf.eye(batch_size, dtype=tf.double)
    mask_neg = tf.cast(adjacency_not, dtype=tf.double)

    sim_mat = tf.matmul(embeddings, embeddings, transpose_a=False, transpose_b=True)
    # below two lines process norm
    embeddings_norm1 = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1))
    sim_mat = tf.math.divide(sim_mat, embeddings_norm1 * embeddings_norm1)

    sim_mat = tf.maximum(sim_mat, 0.0)

    pos_mat = tf.multiply(sim_mat, mask_pos)
    neg_mat = tf.multiply(sim_mat, mask_neg)

    if ms_mining:
        max_val = tf.reduce_max(neg_mat, axis=1, keepdims=True)
        tmp_max_val = tf.reduce_max(pos_mat, axis=1, keepdims=True)
        min_val = tf.reduce_min(tf.multiply(sim_mat - tmp_max_val, mask_pos), axis=1, keepdims=True) + tmp_max_val

        max_val = tf.tile(max_val, [1, batch_size])
        min_val = tf.tile(min_val, [1, batch_size])

        mask_pos = tf.where(pos_mat < max_val + eps, mask_pos, tf.zeros_like(mask_pos))
        mask_neg = tf.where(neg_mat > min_val - eps, mask_neg, tf.zeros_like(mask_neg))

    pos_exp = tf.exp(-alpha * (pos_mat - lamb))
    pos_exp = tf.where(mask_pos > 0.0, pos_exp, tf.zeros_like(pos_exp))

    neg_exp = tf.exp(beta * (neg_mat - lamb))
    neg_exp = tf.where(mask_neg > 0.0, neg_exp, tf.zeros_like(neg_exp))

    pos_term = tf.math.log(1.0 + tf.reduce_sum(pos_exp, axis=1)) / alpha
    neg_term = tf.math.log(1.0 + tf.reduce_sum(neg_exp, axis=1)) / beta

    loss = tf.reduce_mean(pos_term + neg_term) * scala

    return loss