# this code is to create the embeddings as the last state of the encoder, and takes train, dev and test data and with two styles saves them into files with their numpy format
# it can be run in the terminal by running a code like the following:
# python ...z_create.py   --load_model true --model ../../tmp/.../model  --vocab ../../tmp/.../yelp.vocab  --output ../probing_classification/output_emb/adv_MD/sentiment.train  --train ../../data/yelp/sentiment.train                                    
# --train or --dev or -- test shows the directory of the files we want to produce their embedding  representation, --output shows where want to store the files with embedding representations
# --model and --vocab: shows the address of the directory in which  model and the vocab file is stored from which we want to compute the emb_rep of the sequences 
# for different models, we need to replace the model part of that model in this file which is  from line 183 up and keep the rest
import os
import sys
import time
import ipdb
import random
import cPickle as pickle
#import _pickle  as pickle
import numpy as np
import tensorflow as tf
from vocab import Vocabulary, build_vocab
from accumulator import Accumulator
from options import load_arguments
from file_io import load_sent, write_sent
from utils_tf_1 import *
from nn_tf_v_1 import *
import beam_search_tf_1, greedy_decoding_tf_1
os.environ["CUDA_VISIBLE_DEVICES"]="0"
tfd = tf.contrib.distributions

class Model(object):
    def __init__(self, args, vocab):
        # y: style
        dim_y = args.dim_y # 200
        # z = dim_h- dim_y
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        max_len = args.max_seq_length
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        n_filters = args.n_filters
        beta1, beta2 = 0.5, 0.999
        grad_clip = 30.0
        landa = 0.005
        # we fill in these parts in feed_dictionary dictionary
        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.rho = tf.placeholder(tf.float32,
            name='rho')
        self.gamma = tf.placeholder(tf.float32,
            name='gamma')

        self.batch_len = tf.placeholder(tf.int32,
            name='batch_len')
        self.batch_size = tf.placeholder(tf.int32,
            name='batch_size')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None],    #size * len
            name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='dec_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None],
            name='weights')
        self.labels = tf.placeholder(tf.float32, [None],
            name='labels')
        self.len = tf.placeholder(tf.int32,None,
            name= 'max_seq_len_batch')

        labels = tf.reshape(self.labels, [-1, 1])
        # vocab.embedding: it is random initialization of embedding matrix
        self.embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))
        '''
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])
        '''

        with tf.variable_scope('projection0'):
            proj_W0 = tf.get_variable('W', [dim_h +dim_z, vocab.size])
            proj_b0 = tf.get_variable('b', [vocab.size])
        print('*'*10)
        print(proj_W0)
        with tf.variable_scope('projection1'):
            proj_W1 = tf.get_variable('W', [dim_h+dim_z, vocab.size])
            proj_b1 = tf.get_variable('b', [vocab.size])
        print('*'*10)
        print(proj_W1)
        enc_inputs = tf.nn.embedding_lookup(self.embedding, self.enc_inputs)
        dec_inputs = tf.nn.embedding_lookup(self.embedding, self.dec_inputs)

        #####   auto-encoder   #####
        # init_state of encoder is style (y)
        # linear(labels, dim_y): tensor with dim labels.shape(0) * dim_y, 
        # labels.shape = batch_size.shape , since labels is the list of styles of each input
        init_state = tf.concat([linear(labels, dim_y, scope='encoder'),
            tf.zeros([self.batch_size, dim_z])], 1) # Tensor("concat:0", shape=(?, 700), dtype=float32) 700= dim_h = dim y + dim_z
        print(init_state)
        # create GRU cell
        cell_e = create_cell(dim_h, n_layers, self.dropout) # dim_h = dim_y +dim_z =700
        _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
            initial_state=init_state, scope='encoder') # z = last hidden layer of the encoder, z.shape: (batch_size, dim_h)
        # for z, we consider, just the z part and reduce the style part
        z = z[:, dim_y:]
        #print(self.z)  # print(z): Tensor("strided_slice:0", shape=(?, 500), dtype=float32)
        # cell_e = create_cell(dim_z, n_layers, self.dropout)
        # _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
        #    dtype=tf.float32, scope='encoder')

        half = self.batch_size / 2
        z0 = z[:half]
        z1 = z[half:]

        def make_prior(code_size):
            loc = tf.zeros(code_size)
            scale = tf.ones(code_size)
            return tfd.MultivariateNormalDiag(loc, scale)

        loc0 = tf.layers.dense(z0,dim_z)
        print('mean_pos',loc0.shape)
        scale0 = tf.layers.dense(z0, dim_z, tf.nn.softplus)

        posterior0 = tfd.MultivariateNormalDiag(loc0, scale0)
        prior0 = make_prior(dim_z)
        self.z0 = posterior0.sample()

        loc1 = tf.layers.dense(z1,dim_z)
        scale1 = tf.layers.dense(z1, dim_z, tf.nn.softplus)
        posterior1 = tfd.MultivariateNormalDiag(loc1, scale1)
        prior1 = make_prior(dim_z)
        self.z1 = posterior1.sample()

        self.h_ori0 = tf.concat([linear(labels[:half], dim_y,
            scope='generator0'), self.z0], 1) #rx0
        self.h_ori1 = tf.concat([linear(labels[half:], dim_y,
            scope='generator1'), self.z1], 1) #rx1

        self.h_tsf0 = tf.concat([linear(1-labels[:half], dim_y,
            scope='generator1', reuse=True), self.z0], 1) #tsf0
        self.h_tsf1 = tf.concat([linear(1-labels[half:], dim_y,
            scope='generator0', reuse=True), self.z1], 1) #tsf1

        cell_g0 = create_cell(dim_h, n_layers, self.dropout)
        # sequence of hidden layers of decoder
        g_outputs0, _ = tf.nn.dynamic_rnn(cell_g0, dec_inputs[:half],
            initial_state=self.h_ori0, scope='generator0') #rx0

        cell_g1 = create_cell(dim_h, n_layers, self.dropout)
        # sequence of hidden layers of decoder
        g_outputs1, _ = tf.nn.dynamic_rnn(cell_g1, dec_inputs[half:],
            initial_state=self.h_ori1, scope='generator1') #rx1
        print(g_outputs0) # Tensor("generator_2/transpose_1:0", shape=(?, ?, 700), dtype=float32) # shape= batch_size* max_len + 1* dim_h
        # attach h0 in the front, we concat h0= (y;z)  to the beginning of g_outputs
        # tf.expand_dims: Inserts a dimension of 1 into a tensor's shape  # Tensor("ExpandDims:0", shape=(?, 1, 700), dtype=float32)
        # self.h_ori.shape= batch_size * dim_y + dim_z # Tensor("concat_1:0", shape=(?, 700), dtype=float32)

        teach_h0 = tf.concat([tf.expand_dims(self.h_ori0, 1), g_outputs0], 1)
        teach_h1 = tf.concat([tf.expand_dims(self.h_ori1, 1), g_outputs1], 1)

        # after expand_dims, self.h_ori.shape = batch_size * 1 *  dim_y + dim_z ,  # Tensor("ExpandDims:0", shape=(?, 1, 700), dtype=float32)
        # dim g_outputs: batch_size * max_len + 1 * dim_h
        print(teach_h0) # Tensor("concat_3:0", shape=(?, ?, 700), dtype=float32) # after concat, teach_h.shape = batch_size * 1 + dim_g_outputs(max_len + 1) * dim_y + dim_z
        g_outputs0 = tf.nn.dropout(g_outputs0, self.dropout)
        g_outputs0 = tf.concat([g_outputs0, tf.tile(tf.reshape(self.z0, [self.batch_size/2, 1, dim_z]), [1, self.len, 1])],  axis=2)  # in reshaping the product of the dimensions (SIZE) must remain the same as that of the original shape

        g_outputs0 = tf.reshape(g_outputs0, [-1, dim_h+dim_z])
        g_logits0 = tf.matmul(g_outputs0, proj_W0) + proj_b0

        g_outputs1 = tf.nn.dropout(g_outputs1, self.dropout)
        g_outputs1 = tf.concat([g_outputs1, tf.tile(tf.reshape(self.z1, [self.batch_size/2, 1, dim_z]), [1, self.len, 1])],  axis=2)  # in reshaping the product of the dimensions (SIZE) must remain the same as that of the original shape

        g_outputs1 = tf.reshape(g_outputs1, [-1, dim_h+dim_z])
        g_logits1 = tf.matmul(g_outputs1, proj_W1) + proj_b1

        # tf.reshape(t, [-1]): flattens tensor t
        # Computes sparse softmax cross entropy between logits and labels
        loss_rec0 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets[:half], [-1]), logits=g_logits0)
        loss_rec0 *= tf.reshape(self.weights[:half], [-1])
        loss_rec0 = tf.reduce_sum(loss_rec0) / tf.to_float(self.batch_size)
        divergence0 = tfd.kl_divergence(posterior0, prior0)
        divergence0 =  tf.reduce_mean(divergence0) / tf.to_float(self.batch_size)
        self.elbo0 = loss_rec0 + (landa * divergence0)

        loss_rec1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets[half:], [-1]), logits=g_logits1)
        loss_rec1 *= tf.reshape(self.weights[half:], [-1])
        loss_rec1 = tf.reduce_sum(loss_rec1) / tf.to_float(self.batch_size)
        divergence1 = tfd.kl_divergence(posterior1, prior1)
        # for VAE it is good to have tf.reduce_mean & not tf.reduce_sum
        divergence1 =  tf.reduce_mean(divergence1) / tf.to_float(self.batch_size)
        self.elbo1 = loss_rec1 + (landa * divergence1)
        # uptil here, AE, initialised, encoder with y, and decoder with y + z

        self.elbo = self.elbo0 + self.elbo1
        #####   feed-previous decoding   #####
        # two approahes are taken here for feeding (two kinds of inputs)
        go = dec_inputs[:,0,:]
        # soft_func: takes the outputs (all states of decoder) as input and returns inp = tf.matmul(prob, embedding) & also logits of the outputs
        soft_func0 = softsample_word(self.z0,self.dropout, proj_W0, proj_b0, self.embedding,
            self.gamma)
        soft_func1 = softsample_word(self.z1,self.dropout, proj_W1, proj_b1, self.embedding,
            self.gamma)
        # hard_func: takes the outputs (all states of decoder) as input and returns(inp = embedding vector baray har kalame toolidi dar har time step) & also (logits of the outputs of the decoder)
        hard_func0 = argmax_word(self.z0,self.dropout, proj_W0, proj_b0, self.embedding)
        hard_func1 = argmax_word(self.z1,self.dropout, proj_W1, proj_b1, self.embedding)
        # rnn_decode returns sequence of outputs (states?), with the first member as h-ori or h-tsf and sequence of logits of outputs
        soft_h_ori0, soft_logits_ori0 = rnn_decode(self.h_ori0, go[:half], max_len,
            cell_g0, soft_func0, scope='generator0') # soft_rx0
        # x_fake
        soft_h_tsf1, soft_logits_tsf1 = rnn_decode(self.h_tsf1, go[half:], max_len,
            cell_g0, soft_func0, scope='generator0') # soft_tsf1
        hard_h_ori0, self.hard_logits_ori0 = rnn_decode(self.h_ori0, go[:half], max_len,
            cell_g0, hard_func0, scope='generator0') # hard_rx0
        hard_h_tsf1, self.hard_logits_tsf1 = rnn_decode(self.h_tsf1, go[half:], max_len,
            cell_g0, hard_func0, scope='generator0') # hard_tsf1

        soft_h_ori1, soft_logits_ori1 = rnn_decode(self.h_ori1, go[half:], max_len,
            cell_g1, soft_func1, scope='generator1') # soft_rx1
        # x_fake
        soft_h_tsf0, soft_logits_tsf0 = rnn_decode(self.h_tsf0, go[:half], max_len,
            cell_g1, soft_func1, scope='generator1') # soft_tsf0
        hard_h_ori1, self.hard_logits_ori1 = rnn_decode(self.h_ori1, go[half:], max_len,
            cell_g1, hard_func1, scope='generator1') # hard_rx1
        hard_h_tsf0, self.hard_logits_tsf0 = rnn_decode(self.h_tsf0, go[:half], max_len,
            cell_g1, hard_func1, scope='generator1') # hard_tsf0

        #####   discriminator   #####
        # a batch's first half consists of sentences of one style and second half of the other
        zeros, ones = self.labels[:half], self.labels[half:]
        soft_h_tsf0 = soft_h_tsf0[:, :1+self.batch_len, :]
        soft_h_tsf1 = soft_h_tsf1[:, :1+self.batch_len, :]
        # the size of the sequence of the outputs of decoder is the same in soft_h_tsf since in run_decode,
        # we produce to the no of max_len the size can be produced after the token end (they are not pads at the end)
        self.loss_d0, self.loss_g0 = discriminator(teach_h0, soft_h_tsf1,
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator0')
        self.loss_d1, self.loss_g1 = discriminator(teach_h1, soft_h_tsf0,
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator1')
        #####   optimizer   #####
        self.loss_adv = self.loss_g0 + self.loss_g1
        self.loss0 = self.elbo0 + self.rho * self.loss_g0
        self.loss1 = self.elbo1 + self.rho * self.loss_g1
        self.loss = self.loss0 + self.loss1

        theta_eg = retrive_var(['encoder', 'generator1', 'generator0',
            'embedding', 'projection0', 'projection1'])
        theta_eg0 = retrive_var(['encoder', 'generator0',
            'embedding', 'projection0'])
        theta_eg1 = retrive_var(['encoder', 'generator1',
            'embedding', 'projection1'])

        theta_d0 = retrive_var(['discriminator0'])
        theta_d1 = retrive_var(['discriminator1'])

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2)

        grad_rec0, _ = zip(*opt.compute_gradients(self.elbo0, theta_eg0))
        grad_rec1, _ = zip(*opt.compute_gradients(self.elbo1, theta_eg1))
        grad_adv0, _ = zip(*opt.compute_gradients(self.loss_g0, theta_eg0))
        grad_adv1, _ = zip(*opt.compute_gradients(self.loss_g1, theta_eg1))

        grad0, _ = zip(*opt.compute_gradients(self.loss0, theta_eg0))
        grad0, _ = tf.clip_by_global_norm(grad0, grad_clip)
        grad1, _ = zip(*opt.compute_gradients(self.loss1, theta_eg1))
        grad1, _ = tf.clip_by_global_norm(grad1, grad_clip)

        self.grad_rec_norm0 = tf.global_norm(grad_rec0)
        self.grad_rec_norm1 = tf.global_norm(grad_rec1)
        self.grad_adv_norm0 = tf.global_norm(grad_adv0)
        self.grad_adv_norm1 = tf.global_norm(grad_adv1)
        self.grad_norm0 = tf.global_norm(grad0)
        self.grad_norm1 = tf.global_norm(grad1)

        self.optimize_tot0 = opt.apply_gradients(zip(grad0, theta_eg0))
        self.optimize_tot1 = opt.apply_gradients(zip(grad1, theta_eg1))

        self.optimize_rec0 = opt.minimize(self.elbo0, var_list=theta_eg0)
        self.optimize_rec1 = opt.minimize(self.elbo1, var_list=theta_eg1)

        self.optimize_d0 = opt.minimize(self.loss_d0, var_list=theta_d0)
        self.optimize_d1 = opt.minimize(self.loss_d1, var_list=theta_d1)

        #self.train_writer = tf.summary.FileWriter('./log')
        #self.train_writer.add_graph(sess.graph)

        self.saver = tf.train.Saver()

def seq_embeddings(model, sess, args, vocab, data0,data1):
    '''
    :param model:
    :param sess:
    :param args:
    :param vocab:
    :param data0:
    :param data1:
    :return:  the embedding vectors lists (last state of encoder) corresponding to the sentences in the data
    '''
    batches, order0, order1 = get_batches(data0,data1,
        vocab.word2id, args.batch_size)
    
    embed_data0 = []
    embed_data1 = []
    for batch in batches:
        embed_data =[]
        print("batch['size'], len(batch['enc_inputs']),len(batch['enc_inputs'])/2",batch['size'], len(batch['enc_inputs']),len(batch['enc_inputs'])/2)
        half = len(batch['enc_inputs'])/2
        print("half,len(batches), len(batch['enc_inputs'])",half,len(batches), len(batch['enc_inputs']))
        print("batch['enc_inputs'],batch['enc_inputs'][0]",batch['enc_inputs'][:10],len(batch['enc_inputs'][0]))
        
        z0,z1=sess.run( [model.z0, model.z1],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.enc_inputs: batch['enc_inputs'],
                       model.dec_inputs: batch['dec_inputs'],
                       model.labels: batch['labels']})
        #print(len(z),len(z[0]),type(z), len(z[0][0])) # z_shape : 1*128*500
        print("len(z0), len(z0[0],len(z1[0])",len(z0), len(z0[0]),len(z1[0])) # 64*500
        embed_data0.extend(z0)
        embed_data1.extend(z1)
    print("len(embed_data1) after loop",len(embed_data1),len(embed_data0) )
    return embed_data0, embed_data1

def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    
    return model

if __name__ == '__main__':
    #tf.reset_default_graph() # in tu code asli nist
    args = load_arguments()

    #####   data preparation   #####
    if args.train:
        data0 = load_sent(args.train + '.0', args.max_train_size)
        data1 = load_sent(args.train + '.1', args.max_train_size)
        print '#sents of training file 0:', len(data0)
        print '#sents of training file 1:', len(data1)

        if not os.path.isfile(args.vocab):
            build_vocab(data0 + data1, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print 'vocabulary size:', vocab.size

    if args.dev:
        data0 = load_sent(args.dev + '.0')
        data1 = load_sent(args.dev + '.1')

    if args.test:
        data0 = load_sent(args.test + '.0')
        data1 = load_sent(args.test + '.1')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)


        # def seq_embeddings(model, decoder, sess, args, vocab, data, out_path):
        z_seqs0, z_seqs1 = seq_embeddings(model, sess, args, vocab,
                                data0, data1)
        np.save(args.output+'.0'+'.npy',z_seqs0)
        np.save(args.output+'.1'+'.npy',z_seqs1)

        vectors0=np.load(args.output+'.0'+'.npy',allow_pickle=False )
        vectors1=np.load(args.output+'.1'+'.npy',allow_pickle=False )

        print('length of written files 0 and their seq embedding size',len(vectors0), len(vectors0[0]))
        print('length of written files 1 and their seq embedding size',len(vectors1), len(vectors1[0]))



