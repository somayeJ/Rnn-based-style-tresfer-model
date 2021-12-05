from __future__ import unicode_literals
import spacy
from spacy.attrs import ORTH, LIKE_URL
import tensorflow as tf
import numpy as np
from utils_tf_1 import normalize_features, features_vectors
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

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


def gumbel_softmax(logits, gamma, eps=1e-20):
    #tf.shape : This operation returns a 1-D integer tensor representing the shape of input.
    U = tf.random_uniform(tf.shape(logits))
    G = -tf.log(-tf.log(U + eps) + eps)
    return tf.nn.softmax((logits + G) / gamma)
#soft_h_ori, soft_logits_ori = rnn_decode(self.h_ori, go, max_len,
    #cell_g, soft_func, scope='generator')
def softsample_word(z,dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        output = tf.concat([output, z],1)
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

def argmax_word(z,dropout, proj_W, proj_b, embedding):

    def loop_func(output):
        #print('loop_funcoooooooooooooo', output.shape)
        output = tf.nn.dropout(output, dropout)
        output = tf.concat([output], 1)
        output = tf.concat([output, z],1)
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

def rnn_decode_modified(h, inp, length, cell, loop_func, extra_features,vocab ,args, scope):
    h_seq, logits_seq = [], []
    with tf.variable_scope(scope):
        #print('****************************', h, inp.shape)
        #tf.get_variable_scope().reuse_variables()
        inps = []
        for t in range(length):
            h_seq.append(tf.expand_dims(h, 1))
            #print('t_before',t,'****************************', h.shape, inp.shape)

            output, h = cell(inp, h)
            #print('inp,output', inp, output)
            # h.shape= 700, inp.shape= 106, output, 700
            #print('t_after', t, '****************************', h.shape, inp.shape,'out_put', output.shape, output)
            #word, inp0, logits = loop_func(output)
            inp0, logits = loop_func(output)
            logits_seq.append(tf.expand_dims(logits, 1))
            # inja tedadi az feat ha ro ke mishe vared kard vared mikonim baghie ro -2 mizarim
            # natoonestam key arg-max word ro ke yek tensor hast tabdil konam be formi ke mikham ke as id2word kalame ro peyda konam
            if extra_features:
                #print('666666222222222200000000000')
                '''
                inp_word = tf.gather(vocab.id2word, word)
                print('word_idword_idword_id', tf.gather(vocab.id2word, word), inp0.shape, inp_word.shape)
                inps.append(inp_word)  # bayad kalame ro peyda konim pass bedim be features_vectors
                print('inpppppppppp', inps)

                
                nlp = spacy.load('en_core_web_lg')
                doc_feat_list0 , doc_feat_list1, data_tokens0, data_tokens1,word_features_d_0, word_features_d_1= features_vectors([' '.join(inps)], [], nlp)
                normalized_df, normalized_feat_numeric_0 = normalize_features(word_features_d_0, vocab.word2id)
                no_feats = normalized_feat_numeric_0.iloc[1, :-1].shape[0]
                print('normalized_feat_numeric_0', no_feats)
                if inp_word in ['<pad>']:
                    token_extra_features = [[-2] * no_feats]
                else:
                    token_features_of_seq = normalized_feat_numeric_0.loc[normalized_feat_numeric_0.sent_id == sent_no_0,
                                           :].iloc[:, :-1].values.tolist()
                    token_extra_features= token_features_of_seq[-1]
                    print('((((((((((((((token_features_of_seq)))))))))))', token_features_of_seq)
                
                print('snormalized_feat', token_extra_features)
                print('numericcccccccccccccccc',normalized_feat_numeric_0 )
                print('join(inps)',[' '.join(inps)])
                '''
                no_feats = args.dim_token_feat
                token_extra_features = [[-2] * no_feats]* args.batch_size*2

                token_extra_features = tf.cast(token_extra_features, tf.float32)

                inp = tf.concat([inp0,token_extra_features ],-1)
                #inp = tf.concat([inp0,tf.zeros([inp0.shape[0], dim_token_size ])], -1)
            else:
                inp, logits = loop_func(output)
                #word_id, inp, logits = loop_func(output)
            #print('t_after_loop', t, '****************************', h.shape, inp.shape)
            # inp.shape= 100

    return tf.concat(h_seq, 1), tf.concat(logits_seq, 1)

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

def cnn_h0(inp, filter_sizes, n_filters, dropout, scope, reuse=False):
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
                    strides=[1, 1, 1, 1], padding='SAME')

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

def discriminator_h0(x_real, x_fake, ones, zeros,
    filter_sizes, n_filters, dropout, scope,
    wgan=False, eta=10):
    d_real = cnn_h0(x_real, filter_sizes, n_filters, dropout, scope)
    d_fake = cnn_h0(x_fake, filter_sizes, n_filters, dropout, scope, reuse=True)

    if wgan:
        eps = tf.random_uniform([], 0.0, 1.0)
        mix = eps * x_real + (1-eps) * x_fake
        d_mix = cnn(mix, filter_sizes, n_filters, dropout, scope, reuse=True)
        grad = tf.gradients(d_mix, mix)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        loss = d_fake-d_real + eta*tf.square(grad_norm-1)
        return tf.reduce_mean(loss), -tf.reduce_mean(loss)

    else:
        print(d_real, ones)
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_real)) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_fake))
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_fake))
        return loss_d, loss_g

def discriminator_aligned(x_real, x_fake, ones, zeros,
    filter_sizes, n_filters, dropout, scope,
    wgan=False, eta=10):
    d_real = feed_forward_aligned(x_real,  scope)
    d_fake = feed_forward_aligned(x_fake, scope, reuse=True)

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

'''
def discriminator_aligned(x_real, x_fake, ones, zeros,
    filter_sizes, n_filters, dropout, scope,
    wgan=False, eta=10):
    d_real = feed_forward(x_real,  scope)
    d_fake = feed_forward(x_fake, scope, reuse=True)

    if wgan:
        eps = tf.random_uniform([], 0.0, 1.0)
        mix = eps * x_real + (1-eps) * x_fake
        d_mix = cnn(mix, filter_sizes, n_filters, dropout, scope, reuse=True)
        grad = tf.gradients(d_mix, mix)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        loss = d_fake-d_real + eta*tf.square(grad_norm-1)
        return tf.reduce_mean(loss), -tf.reduce_mean(loss)

    else:
        # ba tavajoh be formule cross entropy dar gan ma hamishe bara true haone va bara false ha zero migirim
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_real)) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_fake))

        return loss_d, loss_d
def discriminator_aligned_2(x_real, x_fake, ones, zeros,
    filter_sizes, n_filters, dropout, scope,
    wgan=False, eta=10):
    #d_real = cnn(x_real, filter_sizes, n_filters, dropout, scope)
    #d_fake = cnn(x_fake, filter_sizes, n_filters, dropout, scope, reuse=True)
    d_real = feed_forward(x_real, scope )
    d_fake = feed_forward(x_fake, scope, reuse=True )
    #d_fake = cnn(x_fake, filter_sizes, n_filters, dropout, scope)

    if wgan:
        eps = tf.random_uniform([], 0.0, 1.0)
        mix = eps * x_real + (1-eps) * x_fake
        d_mix = cnn(mix, filter_sizes, n_filters, dropout, scope, reuse=True)
        grad = tf.gradients(d_mix, mix)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        loss = d_fake-d_real + eta*tf.square(grad_norm-1)
        return tf.reduce_mean(loss), -tf.reduce_mean(loss)

    else:
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_real)) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_fake))
        return loss_d, loss_d

def discriminator_aligned3(x_real, x_fake, ones, zeros,
    filter_sizes, n_filters, dropout, scope,
    wgan=False, eta=10):
    #d_real = cnn(x_real, filter_sizes, n_filters, dropout, scope)
    #d_fake = cnn(x_fake, filter_sizes, n_filters, dropout, scope, reuse=True)
    d_real = feed_forward(x_real, scope )
    d_fake = feed_forward(x_fake, scope, reuse=True )
    #d_fake = cnn(x_fake, filter_sizes, n_filters, dropout, scope)

    if wgan:
        eps = tf.random_uniform([], 0.0, 1.0)
        mix = eps * x_real + (1-eps) * x_fake
        d_mix = cnn(mix, filter_sizes, n_filters, dropout, scope, reuse=True)
        grad = tf.gradients(d_mix, mix)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        loss = d_fake-d_real + eta*tf.square(grad_norm-1)
        return tf.reduce_mean(loss), -tf.reduce_mean(loss)

    else:
        loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_real)) + \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_fake))
        loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_fake))+ \
                 tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_real))
        return loss_d, loss_g
'''