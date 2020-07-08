'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''
"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf


def binary_balance_loss(embeddings,steps,summary_writer,q=2,scala=100):
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
    final_loss = tf.reduce_sum(tf.math.abs(tf.reduce_mean(embeddings,1)-(q-1)/2.0))
    if steps% 5 ==0:
        with summary_writer.as_default():
            tf.summary.histogram('code_balance/', embeddings, step=steps)
    return final_loss

def binary_balance_loss_q(embeddings,steps,summary_writer,q=2,scala=100):
    values = tf.cast(embeddings,tf.int32)
    frequency = tf.math.bincount(values,minlength=q,maxlength=q)
    prab = frequency / tf.reduce_sum(frequency)
    final_loss = tf.reduce_sum(tf.abs(prab-1/q)) * scala
    # Get final mean triplet loss
    if steps% 5 ==0:
        with summary_writer.as_default():
            tf.summary.histogram('code_balance/', values, step=steps)
    return final_loss
'''
Regularizing Neural Networks by Penalizing Confident Output Distributions 
Gabriel Pereyra, George Tucker, Jan Chorowski, Lukasz Kaiser, Geoffrey Hinton

H\left(p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})\right)=-\sum_{i} p_{\theta}\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}\right) \log \left(p_{\theta}\left(\boldsymbol{y}_{i} \mid \boldsymbol{x}\right)\right)

\mathcal{L}(\theta)=-\sum \log p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})-\beta H\left(p_{\theta}(\boldsymbol{y} \mid \boldsymbol{x})\right)

H(p)= - sum(p*log(p))
L = -sum(log(p)) - beta H(p) 


'''
def binary_balance_loss_merge(embeddings,steps,summary_writer,q=2,scala=100):
    embeddings = tf.math.round(embeddings)
    values = tf.cast(embeddings, tf.int32)
    final_loss_mean = tf.reduce_sum(tf.math.abs(tf.reduce_mean(embeddings- (q - 1) / 2.0, 1) ))

    frequency = tf.math.bincount(values, minlength=q, maxlength=q)
    prab = frequency / tf.reduce_sum(frequency)
    # final_loss_hist = -tf.math.log(1-tf.reduce_sum(tf.abs(prab - 1 / q))) * scala

    # second way?
    prab = prab + 1e-8
    H = -tf.reduce_sum(prab * tf.math.log(prab))
    final_loss_entropy = -tf.reduce_sum(tf.math.log(prab))- H


    if steps % 5 == 0:
        with summary_writer.as_default():
            tf.summary.scalar('loss/code balance loss_mean/', final_loss_mean, step=steps)
            tf.summary.scalar('loss/code balance loss_histo/', final_loss_entropy, step=steps)
            tf.summary.histogram('code_balance/', values, step=steps)

    final_loss = final_loss_entropy + final_loss_mean
    return final_loss


def binary_balance_loss_entropy(embeddings,steps,summary_writer,q=2,scala=100):
    embeddings = tf.math.round(embeddings)
    values = tf.cast(embeddings, tf.int32)

    frequency = tf.math.bincount(values, minlength=q, maxlength=q)
    prab = frequency / tf.reduce_sum(frequency)
    # final_loss_hist = -tf.math.log(1-tf.reduce_sum(tf.abs(prab - 1 / q))) * scala

    # second way?
    prab = prab + 1e-8
    H = -tf.reduce_sum(prab * tf.math.log(prab))
    final_loss_entropy = -tf.reduce_sum(tf.math.log(prab))- H


    if steps % 5 == 0:
        with summary_writer.as_default():
            tf.summary.scalar('loss/code balance loss_histo/', final_loss_entropy, step=steps)
            tf.summary.histogram('code_balance/', values, step=steps)

    final_loss = final_loss_entropy  * 200
    return final_loss

if __name__ == '__main__':
    embeddings = [[6.0, 4, 7, 5, 4, 6, 4, 5, 0, 2],
                  [1, 4, 0, 5, 1, 4, 5, 5, 5, 2],
                  [6, 0, 2, 0, 2, 3, 5, 3, 4, 4],
                  [6, 0, 0, 0, 3, 6, 5, 1, 4, 3]]
    print(binary_balance_loss_merge(embeddings,q=6))