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

os.environ["CUDA_VISIBLE_DEVICES"]="1"
class Model(object):
    def __init__(self, args, vocab):
        # y: style
        dim_y = args.dim_y # 200
        att_hidden_size= 500
        # z = dim_h- dim_y
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        dim_proj = dim_h 
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
        self.batch_len_input = tf.placeholder(tf.int32,
            name='batch_len_input')
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
        self.weights_input = tf.placeholder(tf.float32, [None, None],
            name='weights_input')

        labels = tf.reshape(self.labels, [-1, 1])
        # vocab.embedding: it is random initialization of embedding matrix
        embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))

        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_proj, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        with tf.variable_scope('attention'):
            W_score1 = tf.get_variable('W_score1', [dim_h,att_hidden_size ], tf.float32, tf.random_normal_initializer())
            W_score2 = tf.get_variable('W_score2', [dim_h, att_hidden_size], tf.float32, tf.random_normal_initializer())
            

            #b_score1 = tf.get_variable('b_score1', [att_hidden_size], tf.float32, tf.zeros_initializer())
            #b_score2 = tf.get_variable('b_score2', [att_hidden_size], tf.float32, tf.zeros_initializer())

            W_score3 = tf.get_variable('W_score3', [att_hidden_size, 1])
            #b_score3 = tf.get_variable('b_score3', [1], tf.float32, tf.zeros_initializer())

        enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
        dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

        #####   auto-encoder   #####
        # init_state of encoder is style (y)
        # linear(labels, dim_y): tensor with dim labels.shape(0) * dim_y, 
        # labels.shape = batch_size.shape , since labels is the list of styles of each input
        init_state = tf.concat([linear(labels, 100, scope='encoder'),
            tf.zeros([self.batch_size, 250])], 1) # Tensor("concat:0", shape=(?, 700), dtype=float32) 700= dim_h = dim y + dim_z
        print('81',init_state)

        # create GRU cell
        cell_e_f = create_cell(350, n_layers, self.dropout) # dim_h = dim_y +dim_z =700
        cell_e_b = create_cell(350, n_layers, self.dropout) # dim_h = dim_y +dim_z =700
        print('89',cell_e_f)

        ((src_hs_fw, src_hs_bw), (src_h_fw, src_h_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_e_f, cell_e_b, enc_inputs, 
             initial_state_fw=init_state, initial_state_bw=init_state, scope='encoder')
        #self.src_contexts0 = tf.concat([src_hs_fw, src_hs_bw],axis=2)
        #print('self.src_contexts0',self.src_contexts0)
        self.src_contexts0 = tf.concat([src_hs_fw[:,:,100:], src_hs_bw[:,:,100:]],axis=2)
        # src_contexts= [batchsize, batch_len_input, 2*250]
        #self.z0 = tf.concat([src_h_fw, src_h_bw] , axis=1)
        self.z1 = tf.concat([src_h_fw[:,100:], src_h_bw[:,100:]] , axis=1)       
    
        #1: change init_state decoder
        init_state_g = tf.zeros([self.batch_size, dim_z])
        self.h_ori = tf.concat([linear(labels, dim_y,
            scope='generator'), init_state_g ], 1)
        self.h_tsf = tf.concat([linear(1-labels, dim_y,
            scope='generator', reuse=True), init_state_g ], 1)
        #self.z2 = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)# [batch_size,700]

        expanded_label_ori = tf.tile(tf.reshape(linear(labels, dim_y, scope='generator', reuse=True), [ self.batch_size, 1, dim_y]), [ 1, self.batch_len_input, 1 ])
        expanded_label_tsf = tf.tile(tf.reshape(linear(1-labels, dim_y, scope='generator', reuse=True), [ self.batch_size, 1, dim_y]), [ 1, self.batch_len_input, 1 ])
        self.src_contexts0_ori = tf.concat([expanded_label_ori, self.src_contexts0 ], 2)
        self.src_contexts0_tsf = tf.concat([expanded_label_tsf, self.src_contexts0 ], 2)

        self.z1_ori = tf.concat([linear(labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)# [batch_size,700]
        self.z1_tsf = tf.concat([linear(1-labels, dim_y, scope='generator', reuse=True), self.z1 ], 1)# [batch_size,700]

        
        cell_g = create_cell(dim_h, n_layers, self.dropout)

        go = dec_inputs[:,0,:]
        #go.shape= batch_size,100

        proj_func = teach_h_word(self.dropout, proj_W, proj_b,embedding,dec_inputs)
        soft_func = softsample_word(self.dropout, proj_W, proj_b, embedding,self.gamma)
        #tf.nn.softmax produces just the result of applying the softmax function to an input tensor. The softmax "squishes" the inputs so that sum(input) = 1: it's a way of normalizing. The shape of output of a softmax is the same as the input: it just normalizes the values. The outputs of softmax can be interpreted as probabilities.
        # hard_func: takes the outputs (all states of decoder) as input and returns(inp = embedding vector baray har kalame toolidi dar har time step) & also (logits of the outputs of the decoder)
        hard_func = argmax_word(self.dropout, proj_W, proj_b, embedding)

        length_output = dec_inputs[:,0,:].get_shape()
        #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        #print('cell_g',cell_g)

        #print('length_output[0].value',length_output[0].value)
        #print('self.dec_inputs.get_shape()',dec_inputs.get_shape())
        #print('self.src_contexts',self.src_contexts)#('self.src_contexts', <tf.Tensor 'concat_2:0' shape=(?, ?, 500) dtype=float32>) 
    
        soft_h_ori, soft_logits_ori ,self.context_vector_seq_sori= rnn_decode('',self.z1_ori,dim_h,self.batch_size,W_score1,W_score2,W_score3,
            self.src_contexts0_ori,self.h_ori, go, self.weights_input,self.batch_len_input,max_len,cell_g, soft_func, scope='generator')
        # x_fake
        soft_h_tsf, soft_logits_tsf,self.context_vector_seq_stsf = rnn_decode('',self.z1_tsf,dim_h,self.batch_size,W_score1,W_score2,W_score3,
            self.src_contexts0_tsf,self.h_tsf, go, self.weights_input,self.batch_len_input,max_len,  cell_g, soft_func, scope='generator')
        hard_h_ori, self.hard_logits_ori,self.context_vector_seq_hori = rnn_decode('',self.z1_ori,dim_h,self.batch_size,W_score1,W_score2,W_score3,
            self.src_contexts0_ori,self.h_ori, go, self.weights_input,self.batch_len_input,max_len,cell_g, hard_func, scope='generator')
        hard_h_tsf, self.hard_logits_tsf,self.context_vector_seq_htsf = rnn_decode('',self.z1_tsf,dim_h,self.batch_size,W_score1,W_score2,W_score3,
            self.src_contexts0_tsf,self.h_tsf, go,self.weights_input,self.batch_len_input, max_len, cell_g, hard_func, scope='generator')

        teach_h,g_logits,self.context_vector_seq_teach0 = rnn_decode('teacher_force',self.z1_ori,dim_h,self.batch_size,W_score1,W_score2,W_score3,
            self.src_contexts0_ori,self.h_ori,go, self.weights_input, self.batch_len_input,max_len, cell_g,proj_func,scope='generator')
        print('g_logits',g_logits)
        print('self.targets',self.targets)
        self.context_vector_seq_teach =tf.reduce_sum(self.context_vector_seq_teach0, axis=1)# batch_siz, 700, batch_len_input
        self.g_logits = tf.reshape(g_logits,[self.batch_size*21, 9357])
        loss_rec = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=self.g_logits)

        loss_rec *= tf.reshape(self.weights, [-1])
        self.loss_rec = tf.reduce_sum(loss_rec) / tf.to_float(self.batch_size)

        #####   discriminator   #####
        # a batch's first half consists of sentences of one style and second half of the other
        half = self.batch_size / 2
        zeros, ones = self.labels[:half], self.labels[half:]
        print('233',soft_h_tsf.shape)
        soft_h_tsf = soft_h_tsf[:, :1+self.batch_len, :]
        print('235',soft_h_tsf.shape)

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
            'embedding', 'projection', 'attention'])
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

        self.saver = tf.train.Saver()

def transfer(model, decoder, sess, args, vocab, data0, data1, out_path):
    batches, order0, order1 = get_batches(data0, data1,
        vocab.word2id, args.batch_size, args)

    data0_rec, data1_rec = [], []
    data0_tsf, data1_tsf = [], []
    losses = Accumulator(len(batches), ['loss', 'rec', 'adv', 'd0', 'd1'])
    for batch in batches:
        #decoder.rewrite(batch): hard logits ro tabdil be kalame mikone va az token ens be baed hazf mikone
        rec, tsf = decoder.rewrite(batch)
        # context_vector.shape: [batch_size,max_len,  2*250]
        half = batch['size'] / 2
        data0_rec += rec[:half]
        data1_rec += rec[half:]
        data0_tsf += tsf[:half]
        data1_tsf += tsf[half:]
        #losses.add: for in len (values= [loss, loss_rec, loss_adv, loss_d0, loss_d1]), value[i]/(batchsize) ro be har yek az moalefe ha ezafe mikonim
        loss, loss_rec, loss_adv, loss_d0, loss_d1 = sess.run([model.loss,
            model.loss_rec, model.loss_adv, model.loss_d0, model.loss_d1],
            feed_dict=feed_dictionary(model, batch, args.rho, args.gamma_min))
        losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1])
    n0, n1 = len(data0), len(data1)
    data0_rec = reorder(order0, data0_rec)[:n0]
    data1_rec = reorder(order1, data1_rec)[:n1]
    #reorder: changes the order of the list data0_tsf based on order0
    data0_tsf = reorder(order0, data0_tsf)[:n0]
    data1_tsf = reorder(order1, data1_tsf)[:n1]
    if out_path:
        write_sent(data0_rec, out_path+'.0'+'.rec')
        write_sent(data1_rec, out_path+'.1'+'.rec')
        write_sent(data0_tsf, out_path+'.0'+'.tsf')
        write_sent(data1_tsf, out_path+'.1'+'.tsf')
    return losses

def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print ('Loading model from', args.model)
        model.saver.restore(sess, args.model)
    else:
        print ('Creating model with fresh parameters.')
        sess.run(tf.global_variables_initializer())
    
    return model

if __name__ == '__main__':
    tf.reset_default_graph() # in tu code asli nist
    args = load_arguments()

    #####   data preparation   #####
    if args.train:
        if args.downsampling_train_SDS:
            train0_orig = load_sent(args.train + '.0', args.max_train_size)
            train1_orig = load_sent(args.train + '.1', args.max_train_size)
            if len(train0_orig)<len(train1_orig):
                train1=train1_orig[:len(train0_orig)]
                train0 = train0_orig[:]
            if len(train1_orig)<len(train0_orig):
                train0=train0_orig[:len(train1_orig)]
                train1 = train1_orig[:]
            print ('#sents of training file 0:', len(train0))
            print ('#sents of training file 1:', len(train1))
        else:
            train0 = load_sent(args.train + '.0', args.max_train_size)
            train1 = load_sent(args.train + '.1', args.max_train_size)
            print ('#sents of training file 0:', len(train0))
            print ('#sents of training file 1:', len(train1))


        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print ('vocabulary size:', vocab.size)

    if args.dev:
        if args.downsampling_train_SDS:
            dev0_orig = load_sent(args.dev + '.0')
            dev1_orig = load_sent(args.dev + '.1')
            if len(dev0_orig)<len(dev1_orig):
                dev1 = dev1_orig[:len(dev0_orig)]
                dev0 = dev0_orig[:]
            if len(dev1_orig)<len(dev0_orig):
                dev0 = dev0_orig[:len(dev1_orig)]
                dev1 = dev1_orig[:]
        else:
            dev0 = load_sent(args.dev + '.0')
            dev1 = load_sent(args.dev + '.1')

    if args.test:
        if args.downsampling_train_SDS:
            test0_orig = load_sent(args.test + '.0')
            test1_orig = load_sent(args.test + '.1')
            if len(test0_orig)<len(test1_orig):
                test1 = test1_orig[:len(test0_orig)]
                test0 = test0_orig[:]
            if len(test1_orig)<len(test0_orig):
                test0 = test0_orig[:test1_orig]
                test1 = test1[:]
        else:
            test0 = load_sent(args.test + '.0')
            test1 = load_sent(args.test + '.1')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)

        if args.beam > 1:
            decoder = beam_search_tf_1.Decoder(sess, args, vocab, model)
        else:
            decoder = greedy_decoding_tf_1.Decoder(sess, args, vocab, model)

        if args.train:
            batches, _, _ = get_batches(train0, train1, vocab.word2id,
                args.batch_size,args, noisy=True)
            random.shuffle(batches)
            start_time = time.time()
            step = 0
            losses = Accumulator(args.steps_per_checkpoint,
                ['loss', 'rec', 'adv', 'd0', 'd1'])
            best_dev = float('inf')
            learning_rate = args.learning_rate
            rho = args.rho
            gamma = args.gamma_init
            dropout = args.dropout_keep_prob
            #gradients = Accumulator(args.steps_per_checkpoint,
            #    ['|grad_rec|', '|grad_adv|', '|grad|'])

            for epoch in range(1, 1+args.max_epochs):
                print( '--------------------epoch %d--------------------' % epoch)
                print ('learning_rate:', learning_rate, '  gamma:', gamma)

                for batch in batches:
                    feed_dict = feed_dictionary(model, batch, rho, gamma,
                        dropout, learning_rate)
                    loss_d0, _ = sess.run([model.loss_d0, model.optimize_d0],
                        feed_dict=feed_dict)
                    loss_d1, _ = sess.run([model.loss_d1, model.optimize_d1],
                        feed_dict=feed_dict)
                    
                    # do not back-propagate from the discriminator
                    # when it is too poor
                    if loss_d0 < 1.2 and loss_d1 < 1.2:
                        optimize = model.optimize_tot
                    else:
                        optimize = model.optimize_rec

                    loss, loss_rec, loss_adv, _ = sess.run([model.loss,
                        model.loss_rec, model.loss_adv, optimize],
                        feed_dict=feed_dict)
                    losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1])

                    #grad_rec, grad_adv, grad = sess.run([model.grad_rec_norm,
                    #    model.grad_adv_norm, model.grad_norm],
                    #    feed_dict=feed_dict)
                    #    gradients.add([grad_rec, grad_adv, grad])

                    step += 1
                    if step % args.steps_per_checkpoint == 0:
                        losses.output('step %d, time %.0fs,'
                            % (step, time.time() - start_time))
                        losses.clear()

                        #gradients.output()
                        #gradients.clear()

                if args.dev:
                    dev_losses = transfer(model, decoder, sess, args, vocab,
                        dev0, dev1, args.output + '.epoch%d' % epoch)
                    dev_losses.output('dev')
                    # dev_losses.values[0], in loss e kol hast 
                    if dev_losses.values[0] < best_dev:
                        best_dev = dev_losses.values[0]
                        print ('saving model...')
                        model.saver.save(sess, args.model)

                gamma = max(args.gamma_min, gamma * args.gamma_decay)

        if args.test:
            print ('creating test loss')
            test_losses = transfer(model, decoder, sess, args, vocab,
                test0, test1, args.output)
            test_losses.output('test')
            values, names = test_losses.output_list()
            with open(args.output, 'w') as f:
                for n,v in zip(names, values):
                    f.write(str(n))
                    f.write('\t')
                    f.write(str(v))
                    f.write('\n')
            test_losses.output('test')

        if args.online_testing:
            while True:
                sys.stdout.write('> ')
                sys.stdout.flush()
                inp = sys.stdin.readline().rstrip()
                if inp == 'quit' or inp == 'exit':
                    break
                inp = inp.split()
                y = int(inp[0])
                sent = inp[1:]

                batch = get_batch([sent], [y], vocab.word2id)
                ori, tsf = decoder.rewrite(batch)
                print ('original:', ' '.join(w for w in ori[0]))
                print ('transfer:', ' '.join(w for w in tsf[0]))
