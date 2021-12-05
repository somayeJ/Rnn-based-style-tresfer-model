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
        print('/gen/w',W)
        
    return tf.matmul(inp, W) + b

def linear_2(inp, dim_out, scope, reuse=False):
    dim_in = inp.get_shape().as_list()[-1]
    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()
        # Gets an existing variable with these parameters or create a new one.
        W2 = tf.get_variable('W2', [dim_in, dim_out])
        b2 = tf.get_variable('b2', [dim_out])
    return tf.matmul(inp, W2) + b2

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

def softsample_word0(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output, context_vector,t):
        output0 = tf.nn.dropout(output, dropout)
        #output0=tf.expand_dims(output00,1)
        print('output0.shape',output0.shape) #(?,700)
        print('context_vector.shape',context_vector.shape)#(?, 20, 500)
        print(' context_vector[:,t,:].shape', context_vector[:,t,:].shape)# (?, 500)
        print('tf.expand_dims(output0,1)',tf.expand_dims(output0,1))# 
        # Tensor("generator_4/ExpandDims:0", shape=(?, 1, 700), dtype=float32)
        output = tf.concat([ output0, context_vector[:,t,:]], axis=1)
        # output_size = [batch_size,1,dim_h+(250*2)]
        logits = tf.matmul(output, proj_W) + proj_b
        prob = gumbel_softmax(logits, gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func

def softmax_word0(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        prob = tf.nn.softmax(logits / gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func

def argmax_word0(dropout, proj_W, proj_b, embedding):

    def loop_func(output,context_vector,t):
        output0 = tf.nn.dropout(output, dropout)
        #output0=tf.expand_dims(output00,1)
        output = tf.concat([ output0, context_vector[:,t,:]], axis=1)
        # output_size = [batch_size,1,dim_h+(250*2)]
        logits = tf.matmul(output, proj_W) + proj_b
        word = tf.argmax(logits, axis=1)
        inp = tf.nn.embedding_lookup(embedding, word)
        #return word, inp, logits
        return inp, logits
    return loop_func

# in the code softsample_word and argmax_word are used
#soft_h_tsf, soft_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
            #cell_g, soft_func, scope='generator')   
def rnn_decode0(context_vector,h, inp, length, cell, loop_func, scope):
    # context_vector.shape = [batch_size,max_len,  2*250]
    h_seq, logits_seq = [], []
    inp_shape = inp.shape
    with tf.variable_scope(scope):
        tf.get_variable_scope().reuse_variables()
        for t in range(length):
            
            '''
            h0= tf.concat([h,context_vector[:,t,:]],axis=1)
            h_seq.append(tf.expand_dims(h0, 1))# we do not want to append the h0 (h_ori or h_tsf) to the h seqs
            ''' 
            h_seq.append(tf.expand_dims(h, 1))

            output, h = cell(inp, h)
            inp, logits = loop_func(output,context_vector,t)
            inp.set_shape(inp_shape)
            logits_seq.append(tf.expand_dims(logits, 1))

    return tf.concat(h_seq, 1), tf.concat(logits_seq, 1)

def attention(dim_h,batch_size,W_score1,b_score1,W_score2,h,weights_input,batch_len_input,src_context):
    #src_contexts.shape: [batchsize,1, batch_len_input, 2*250]

    # expand tgt 
    expanded_tgt = tf.tile(tf.reshape(h, [ batch_size, 1,1, dim_h]), [ 1,1, batch_len_input, 1 ])
    # expanded_tgt.shape: [batch_size, 1,batch_len_input, dim_h]
    
    # score_input: concat
    #score_input = tf.concat([expanded_tgt,src_context], axis=2)
    # score_input.shape: [batch_size, batch_len_input, 2*250+dim_h]
    score_input = tf.concat([
            tf.reshape(expanded_tgt, [ batch_size, 1*batch_len_input, dim_h ]),
            tf.reshape(src_context, [ batch_size, batch_len_input*1, 2*250 ])], axis=2)
    # score_input.shape: [batch_size,1* batch_len_input, 2*250+dim_h]

    # pass through two feed forward attention layer
    scores_h = tf.tanh(tf.matmul(tf.reshape(score_input, [ batch_size*1*batch_len_input, dim_h+2*250 ]), W_score1) + b_score1)
    scores   = tf.reshape(tf.matmul(scores_h, W_score2), [ batch_size*1* batch_len_input ])
    #score_h.shape: [batch_size* batch_len_input, att_hidden_size]
    #scores.shape [ batch_size*  self.batch_len ])
    
    # src_mask:
    src_mask = tf.tile(tf.reshape(weights_input, [ batch_size, 1, batch_len_input ]), [ 1, 1, 1 ])
    #src_mask.shape =  [batch_size,1, batch_len_input]
    scores *= tf.reshape(src_mask,[-1])
    scores = tf.reshape(scores,[ batch_size, 1, batch_len_input])
    att = tf.nn.softmax(scores)
    expanded_att = tf.tile(tf.reshape(att, [ batch_size,1, batch_len_input, 1]), [1, 1,  1, 2*250 ])
    # expanded_att.shape: [batch_size, batch_len_input, 2*250]

    # context_vector
    context_vector = tf.reduce_sum(src_context*expanded_att, axis=2)
    # context_vector.shape: (?, 1,  500)
    print(src_context.shape)# (?, 1, ?, 500)
    print(expanded_att.shape)# (?, 1, ?, 500)
    print(context_vector.shape)# (?, 1,  500)
    return context_vector

def softsample_word(dropout, proj_W, proj_b, embedding, gamma):
    def loop_func(batch_size,output, context_vector):
        output0= tf.nn.dropout(output, dropout)
        #output0=tf.expand_dims(output00,1)
        print('output00.shape',output0.shape) #(?,700)
        print('context_vector.shape',context_vector.shape)#(?, 20, 500)
        print(' context_vector[:,t,:].shape', context_vector[:,:].shape)# (?, 500)
        print('output0',output0.shape)# 
        # Tensor("generator_4/ExpandDims:0", shape=(?, 1, 700), dtype=float32)
        print(output0.shape)
        print(context_vector.shape)
        print(context_vector[:,0,:].shape)
        print(context_vector[:,:].shape)
        output = tf.concat([ output0, tf.reshape(context_vector, [ batch_size,   2*250])], axis=1)
 
        
        # output_size = [batch_size,1,dim_h+(250*2)]
        logits = tf.matmul(output, proj_W) + proj_b
        prob = gumbel_softmax(logits, gamma)
        inp = tf.matmul(prob, embedding)
        print(inp.shape)
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

    def loop_func(batch_size,output,context_vector):
        output0= tf.nn.dropout(output, dropout)
        #output0=tf.expand_dims(output00,1)
        output = tf.concat([ output0, tf.reshape(context_vector, [ batch_size,   2*250])], axis=1)
        # output_size = [batch_size,1,dim_h+(250*2)]
        logits = tf.matmul(output, proj_W) + proj_b
        word = tf.argmax(logits, axis=1)
        print(word.shape)
        inp = tf.nn.embedding_lookup(embedding, word)
        print(inp.shape)
        #return word, inp, logits
        return inp, logits
    return loop_func

# in the code softsample_word and argmax_word are used
#soft_h_tsf, soft_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
            #cell_g, soft_func, scope='generator')   
def rnn_decode(dim_h,batch_size,W_score1,b_score1,W_score2,src_context,h, inp, weights_input,batch_len_input,length, cell, loop_func, scope):
    # context_vector.shape = [batch_size,max_len,  2*250]
    h_seq, logits_seq = [], []
    inp_shape = inp.shape
    print(inp_shape)

    with tf.variable_scope(scope):
        tf.get_variable_scope().reuse_variables()
        for t in range(length):
            print('ttttttttttt',batch_size,t,loop_func)
            '''
            h0= tf.concat([h,context_vector[:,t,:]],axis=1)
            h_seq.append(tf.expand_dims(h0, 1))# we do not want to append the h0 (h_ori or h_tsf) to the h seqs
            ''' 
            h_seq.append(tf.expand_dims(h, 1))
            print('******',h.shape,t)
            output, h = cell(inp, h)
            context_vector = attention(dim_h,batch_size,W_score1,b_score1,W_score2,output,weights_input,batch_len_input,src_context)
            inp, logits = loop_func(batch_size,output,context_vector)
            print(inp.shape)
            
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
                #shape (1, 700, 1, 128) and found shape (1, 1200, 1, 128).
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
    print('x_real.shape',x_real.shape)
    print('x_fake.shape',x_fake.shape)
    
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

