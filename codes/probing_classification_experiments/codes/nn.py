from __future__ import unicode_literals
import spacy
from spacy.attrs import ORTH, LIKE_URL
import tensorflow as tf
import numpy as np
#from utils import normalize_features, features_vectors

def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def create_cell(dim, n_layers, dropout):
    cell = tf.nn.rnn_cell.GRUCell(dim, reuse=tf.AUTO_REUSE )
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    if n_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layers)
    return cell

def retrive_var(scopes):
    var = []
    for scope in scopes:
        var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)
    return var

def linear(inp, dim_out, scope, reuse=False):
    dim_in = inp.get_shape().as_list()[-1]
    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()
        # Gets an existing variable with these parameters or create a new one.
        W = tf.get_variable('W', [dim_in, dim_out])
        b = tf.get_variable('b', [dim_out])
    return tf.matmul(inp, W) + b

def combine(x, y, scope, reuse=False):
    dim_x = x.get_shape().as_list()[-1]
    dim_y = y.get_shape().as_list()[-1]

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W = tf.get_variable('W', [dim_x+dim_y, dim_x])
        b = tf.get_variable('b', [dim_x])

    h = tf.matmul(tf.concat([x, y], 1), W) + b
    return leaky_relu(h)

def feed_forward(inp, scope, reuse=False):
    dim = inp.get_shape().as_list()[-1]

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W1 = tf.get_variable('W1', [dim, dim])
        b1 = tf.get_variable('b1', [dim])
        W2 = tf.get_variable('W2', [dim, 1])
        b2 = tf.get_variable('b2', [1])
    h1 = leaky_relu(tf.matmul(inp, W1) + b1)
    logits = tf.matmul(h1, W2) + b2

    return tf.reshape(logits, [-1])
def feed_forward_aligned(inp, scope, reuse=False):
    dim = inp.get_shape().as_list()[-1]
    inp = tf.reshape(inp, [-1, dim])

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W1 = tf.get_variable('W1', [dim, dim])
        b1 = tf.get_variable('b1', [dim])
        W2 = tf.get_variable('W2', [dim, 1])
        b2 = tf.get_variable('b2', [1])
    h1 = leaky_relu(tf.matmul(inp, W1) + b1)
    logits = tf.matmul(h1, W2) + b2

    return tf.reshape(logits, [-1])
def feed_forward_aligned_classifier(inp, scope, reuse=False):
    dim = inp.get_shape().as_list()[-1]
    #dim = inp.shape().as_list([-1])
    print('22222',inp)
    print('33333',dim)
    inp = tf.reshape(inp, [-1, dim])
    print(tf.contrib.framework.get_name_scope())
    with tf.variable_scope(scope) as vs:
        print('4444444',vs, reuse)
        try: 
            W1 = tf.get_variable('W1_c', [dim, dim])
            b1 = tf.get_variable('b1_c', [dim])
            W2 = tf.get_variable('W2_c', [dim, 1])
            b2 = tf.get_variable('b2_c', [1])   
        except ValueError:     
            vs.reuse_variables()
            W1 = tf.get_variable('W1_c', [dim, dim])
            b1 = tf.get_variable('b1_c', [dim])
            W2 = tf.get_variable('W2_c', [dim, 1])
            b2 = tf.get_variable('b2_c', [1])
        '''
        if reuse:
            print(8888888888888888, 'reuse:',reuse)
            vs.reuse_variables()

        W1 = tf.get_variable('W1_c', [dim, dim])
        b1 = tf.get_variable('b1_c', [dim])
        W2 = tf.get_variable('W2_c', [dim, 1])
        b2 = tf.get_variable('b2_c', [1])
        '''
    h1 = leaky_relu(tf.matmul(inp, W1) + b1)
    logits = tf.matmul(h1, W2) + b2

    return tf.reshape(logits, [-1])



def gumbel_softmax(logits, gamma, eps=1e-20):
    #tf.shape : This operation returns a 1-D integer tensor representing the shape of input.
    U = tf.random_uniform(tf.shape(logits))
    G = -tf.log(-tf.log(U + eps) + eps)
    return tf.nn.softmax((logits + G) / gamma)
#soft_h_ori, soft_logits_ori = rnn_decode(self.h_ori, go, max_len,
    #cell_g, soft_func, scope='generator')
def softsample_word(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        prob = gumbel_softmax(logits, gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func

def softmax_word(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        prob = tf.nn.softmax(logits / gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func

def argmax_word(dropout, proj_W, proj_b, embedding):

    def loop_func(output):
        #print('loop_funcoooooooooooooo', output.shape)
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        #print('logitsssssssssss', logits.shape)
        word = tf.argmax(logits, axis=1)
        inp = tf.nn.embedding_lookup(embedding, word)
        #return word, inp, logits
        return inp, logits

    return loop_func
def rnn_decode(h, inp, length, cell, loop_func, scope):
    h_seq, logits_seq = [], []
    inp_shape = inp.shape
    with tf.variable_scope(scope):
        tf.get_variable_scope().reuse_variables()
        for t in range(length):
            h_seq.append(tf.expand_dims(h, 1))
            output, h = cell(inp, h)
            inp, logits = loop_func(output)
            inp.set_shape(inp_shape)
            logits_seq.append(tf.expand_dims(logits, 1))

    return tf.concat(h_seq, 1), tf.concat(logits_seq, 1)

# hard_h_ori, self.hard_logits_ori = rnn_decode(self.h_ori, go, max_len, cell_g, hard_func, scope='generator')



def cnn(inp, filter_sizes, n_filters, dropout, scope, reuse=False):
    '''
    :param inp:
    :param filter_sizes: default 1,2,3,4,5
    :param n_filters: default 128
    :param dropout:
    :param scope:
    :param reuse:
    :return:
    '''
    dim = inp.get_shape().as_list()[-1]
    inp = tf.expand_dims(inp, -1)

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        outputs = []
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size):
                W = tf.get_variable('W', [size, dim, 1, n_filters])
                b = tf.get_variable('b', [n_filters])
                conv = tf.nn.conv2d(inp, W,
                    strides=[1, 1, 1, 1], padding='VALID')

                #conv = tf.nn.conv2d(inp, W,
                    #strides=[1, 1, 1, 1], padding='SAME')
                h = leaky_relu(conv + b)
                # max pooling over time
                pooled = tf.reduce_max(h, reduction_indices=1)
                pooled = tf.reshape(pooled, [-1, n_filters])
                outputs.append(pooled)
        outputs = tf.concat(outputs, 1)
        outputs = tf.nn.dropout(outputs, dropout)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [n_filters*len(filter_sizes), 1])
            b = tf.get_variable('b', [1])
            logits = tf.reshape(tf.matmul(outputs, W) + b, [-1])
    return logits

def discriminator(x_real, x_fake, ones, zeros,
    filter_sizes, n_filters, dropout, scope,
    wgan=False, eta=10):
    d_real = cnn(x_real, filter_sizes, n_filters, dropout, scope)
    d_fake = cnn(x_fake, filter_sizes, n_filters, dropout, scope, reuse=True)

    if wgan:
        eps = tf.random_uniform([], 0.0, 1.0)
        mix = eps * x_real + (1-eps) * x_fake
        d_mix = cnn(mix, filter_sizes, n_filters, dropout, scope, reuse=True)
        grad = tf.gradients(d_mix, mix)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        loss = d_fake-d_real + eta*tf.square(grad_norm-1)
        return tf.reduce_mean(loss), -tf.reduce_mean(loss)

    else:
        # ba tavajoh be formule cross entropy dar gan ma hamishe bara true ha one va bara false ha zero migirim
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_real)) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_fake))
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_fake))
        return loss_d, loss_g



def classifier_cnn(x_0, x_1, ones, zeros,
    filter_sizes, n_filters, dropout, scope,
    wgan=False, eta=10):
    '''
    x_0: x with origional style 0
    x_fake: x with origional style 1
    '''
    d_0 = cnn(x_0, filter_sizes, n_filters, dropout, scope)
    d_1 = cnn(x_1, filter_sizes, n_filters, dropout, scope, reuse=True)

    if wgan:
        eps = tf.random_uniform([], 0.0, 1.0)
        mix = eps * x_real + (1-eps) * x_fake
        d_mix = cnn(mix, filter_sizes, n_filters, dropout, scope, reuse=True)
        grad = tf.gradients(d_mix, mix)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        loss = d_fake-d_real + eta*tf.square(grad_norm-1)
        return tf.reduce_mean(loss), -tf.reduce_mean(loss)

    else:
        # ba tavajoh be formule cross entropy dar gan ma hamishe bara true ha one va bara false ha zero migirim
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_0)) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=output, logits=d_1))

        return loss_d

def classifier(x_0, x_1, ones, zeros,
    filter_sizes, n_filters, dropout, scope,
    wgan=False, eta=10):
    '''
    x_0: x with origional style 0
    x_fake: x with origional style 1
    '''
    print('xxxxxxxxxxxxxxxxxxxxxxxxxx000000000000000000',x_0)
    d_0 = feed_forward_aligned_classifier(x_0,  scope)
    d_1 = feed_forward_aligned_classifier(x_1, scope, reuse=True)

    if wgan:
        eps = tf.random_uniform([], 0.0, 1.0)
        mix = eps * x_real + (1-eps) * x_fake
        d_mix = cnn(mix, filter_sizes, n_filters, dropout, scope, reuse=True)
        grad = tf.gradients(d_mix, mix)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        loss = d_fake-d_real + eta*tf.square(grad_norm-1)
        return tf.reduce_mean(loss), -tf.reduce_mean(loss)

    else:
        # ba tavajoh be formule cross entropy dar gan ma hamishe bara true ha one va bara false ha zero migirim
        loss_d0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_0))
        loss_d1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_1))

        return loss_d0 +loss_d1

