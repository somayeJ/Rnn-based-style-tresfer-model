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
            name='max_seq_len_batch')

        labels = tf.reshape(self.labels, [-1, 1])
        # vocab.embedding: it is random initialization of embedding matrix
        self.embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))

        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h +dim_z , vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

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
        self.z = z[:, dim_y:]
        # print(z): Tensor("strided_slice:0", shape=(?, 500), dtype=float32)
        # cell_e = create_cell(dim_z, n_layers, self.dropout)
        # _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
        #    dtype=tf.float32, scope='encoder')

        self.h_ori = tf.concat([linear(labels, dim_y,
            scope='generator'), self.z], 1)
        #self.h_ori = tf.concat([h_ori, self.z],1)
        self.h_tsf = tf.concat([linear(1-labels, dim_y,
            scope='generator', reuse=True), self.z], 1)
        #self.h_tsf = tf.concat([h_tsf,self.z],1)

        cell_g = create_cell(dim_h, n_layers, self.dropout)
        # sequence of hidden layers of decoder
        g_outputs0, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs,
            initial_state=self.h_ori, scope='generator')
        teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs0], 1)


        # after expand_dims, self.h_ori.shape = batch_size * 1 *  dim_y + dim_z ,  # Tensor("ExpandDims:0", shape=(?, 1, 700), dtype=float32)
        # dim g_outputs: batch_size * max_len + 1 * dim_h
        print(teach_h) # Tensor("concat_3:0", shape=(?, ?, 700), dtype=float32) # after concat, teach_h.shape = batch_size * 1 + dim_g_outputs(max_len + 1) * dim_y + dim_z
        g_outputs0 = tf.nn.dropout(g_outputs0, self.dropout)
        g_outputs = tf.concat([g_outputs0, tf.tile(tf.reshape(self.z, [self.batch_size, 1, dim_z]), [1, self.len, 1])],  axis=2)  # in reshaping the product of the dimensions (SIZE) must remain the same as that of the original shape

        g_outputs = tf.reshape(g_outputs, [-1, dim_h+dim_z])
        g_logits = tf.matmul(g_outputs, proj_W) + proj_b
        #logits = tf.reshape(tf.matmul(tf.reshape(preoutput, [ batch_size*tgt_num_steps, tgt_state_size+src_state_size +image_feat_vect_compress_size]), W) + b, [ batch_size, tgt_num_steps, len(unique_vocab_hyp) ])


        # tf.reshape(t, [-1]): flattens tensor t
        # Computes sparse softmax cross entropy between logits and labels
        loss_rec = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=g_logits)
        loss_rec *= tf.reshape(self.weights, [-1])
        self.loss_rec = tf.reduce_sum(loss_rec) / tf.to_float(self.batch_size)
        # uptil here, AE, initialised, encoder with y, and decoder with y + z
        #####   feed-previous decoding   #####
        # two approahes are taken here for feeding (two kinds of inputs)
        go = dec_inputs[:,0,:]
        # soft_func: takes the outputs (all states of decoder) as input and returns inp = tf.matmul(prob, embedding) & also logits of the outputs
        soft_func = softsample_word(self.z,self.dropout, proj_W, proj_b, self.embedding,
            self.gamma)
        # hard_func: takes the outputs (all states of decoder) as input and returns(inp = embedding vector baray har kalame toolidi dar har time step) & also (logits of the outputs of the decoder)
        hard_func = argmax_word(self.z,self.dropout, proj_W, proj_b, self.embedding)
        # rnn_decode returns sequence of outputs (states?), with the first member as h-ori or h-tsf and sequence of logits of outputs
        soft_h_ori, soft_logits_ori = rnn_decode(self.h_ori, go, max_len,
            cell_g, soft_func, scope='generator')
        # x_fake
        soft_h_tsf, soft_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
            cell_g, soft_func, scope='generator')
        hard_h_ori, self.hard_logits_ori = rnn_decode(self.h_ori, go, max_len,
            cell_g, hard_func, scope='generator')
        hard_h_tsf, self.hard_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
            cell_g, hard_func, scope='generator')

        #####   discriminator   #####
        # a batch's first half consists of sentences of one style and second half of the other
        half = self.batch_size / 2
        zeros, ones = self.labels[:half], self.labels[half:]
        soft_h_tsf = soft_h_tsf[:, :1+self.batch_len, :]
        # the size of the sequence of the outputs of decoder is the same in soft_h_tsf since in run_decode,
        # we produce to the no of max_len the size can be produced after the token end (they are not pads at the end)
        self.loss_d0, loss_g0 = discriminator(teach_h[:half], soft_h_tsf[half:],
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator0')
        self.loss_d1, loss_g1 = discriminator(teach_h[half:], soft_h_tsf[:half],
            ones, zeros, filter_sizes, n_filters, self.dropout,
            scope='discriminator1')
        #####   optimizer   #####
        self.loss_adv = loss_g0 + loss_g1
        self.loss = self.loss_rec + self.rho * self.loss_adv

        theta_eg = retrive_var(['encoder', 'generator',
            'embedding', 'projection'])
        theta_d0 = retrive_var(['discriminator0'])
        theta_d1 = retrive_var(['discriminator1'])

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2)

        grad_rec, _ = zip(*opt.compute_gradients(self.loss_rec, theta_eg))
        grad_adv, _ = zip(*opt.compute_gradients(self.loss_adv, theta_eg))
        grad, _ = zip(*opt.compute_gradients(self.loss, theta_eg))
        grad, _ = tf.clip_by_global_norm(grad, grad_clip)

        self.grad_rec_norm = tf.global_norm(grad_rec)
        self.grad_adv_norm = tf.global_norm(grad_adv)
        self.grad_norm = tf.global_norm(grad)

        self.optimize_tot = opt.apply_gradients(zip(grad, theta_eg))
        self.optimize_rec = opt.minimize(self.loss_rec, var_list=theta_eg)
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
        vocab.word2id, args.batch_size, args)
    
    embed_data0 = []
    embed_data1 = []
    for batch in batches:
        embed_data =[]
        print("batch['size'], len(batch['enc_inputs']),len(batch['enc_inputs'])/2",batch['size'], len(batch['enc_inputs']),len(batch['enc_inputs'])/2)
        half = len(batch['enc_inputs'])/2
        print("half,len(batches), len(batch['enc_inputs'])",half,len(batches), len(batch['enc_inputs']))
        print("batch['enc_inputs'],batch['enc_inputs'][0]",batch['enc_inputs'][:10],len(batch['enc_inputs'][0]))
        
        z=sess.run( [model.z],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.enc_inputs: batch['enc_inputs'],
                       model.dec_inputs: batch['dec_inputs'],
                       model.labels: batch['labels']})
        print(len(z),len(z[0]),type(z), len(z[0][0])) # z_shape : 1*128*500
        print("len(z[0][:half], len(z[0][:half]",len(z[0][:half]),len(z[0][0])) # 64*500
        embed_data0.extend(z[0][:half])
        embed_data1.extend(z[0][half:])
        """
        print(len(embed_data0), len(embed_data1))
        input()
        print("len(embed_data)",embed_data,len(embed_data), len(embed_data[0]))
        input()
        embed_data0.extend([embed_data[0][:half]])
        embed_data1.extend([embed_data[0][half:]])
        print('len(embed_data[0][0])', len(embed_data[0][0]))
        print(11111111111111,len(embed_data0), len(embed_data1), len(embed_data1[0]), len(embed_data1[0][0]), len(embed_data) , len(embed_data[0]))

    print("len(embed_data1) after loop",len(embed_data1))
    embed_vectors_final0 = []
    embed_vectors_final1 = []
    for l in embed_data0:
        embed_vectors_final0.extend(l)
    print(len(embed_vectors_final0))
    matrix0 = np.array(embed_vectors_final0)
    for l in embed_data1:
        embed_vectors_final1.extend(l)
    print(len(embed_vectors_final1))
    matrix1 = np.array(embed_vectors_final1)
    """
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



