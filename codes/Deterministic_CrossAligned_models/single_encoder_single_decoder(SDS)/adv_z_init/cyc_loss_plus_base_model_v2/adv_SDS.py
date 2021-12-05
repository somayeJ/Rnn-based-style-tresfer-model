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
        # z = dim_h- dim_y
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        cyclic_soft = args.cyclic_soft
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
        self.cycl_factor = tf.placeholder(tf.float32,
            name='cycl_factor')
        self.rec_factor = tf.placeholder(tf.float32,
            name='rec_factor')
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

        labels = tf.reshape(self.labels, [-1, 1])
        # vocab.embedding: it is random initialization of embedding matrix
        embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))

        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
        dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

        #####   auto-encoder   #####
        # init_state of encoder is style (y)
        # linear(labels, dim_y): tensor with dim labels.shape(0) * dim_y, 
        # labels.shape = batch_size.shape , since labels is the list of styles of each input
        init_state = tf.concat([linear(labels, dim_y, scope='encoder'),
            tf.zeros([self.batch_size, dim_z])], 1) # Tensor("concat:0", shape=(?, 700), dtype=float32) 700= dim_h = dim y + dim_z
        # create GRU cell
        cell_e = create_cell(dim_h, n_layers, self.dropout) # dim_h = dim_y +dim_z =700
        print('cell_e', cell_e)

        _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
            initial_state=init_state, scope='encoder') # z = last hidden layer of the encoder, z.shape: (batch_size, dim_h)
        print('enc_inputs', enc_inputs) # shape=(?, ?, 100)
        print('init_state',init_state)  # shape=(?, 700)
        # for z, we consider, just the z part and reduce the style part
        z = z[:, dim_y:]
        # print(z): Tensor("strided_slice:0", shape=(?, 500), dtype=float32)
        # cell_e = create_cell(dim_z, n_layers, self.dropout)
        # _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
        #    dtype=tf.float32, scope='encoder')

        self.h_ori = tf.concat([linear(labels, dim_y,
            scope='generator'), z], 1)
        self.h_tsf = tf.concat([linear(1-labels, dim_y,
            scope='generator', reuse=True), z], 1)

        cell_g = create_cell(dim_h, n_layers, self.dropout)
        # sequence of hidden layers of decoder
        g_outputs, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs,
            initial_state=self.h_ori, scope='generator')
        #print(g_outputs) # Tensor("generator_2/transpose_1:0", shape=(?, ?, 700), dtype=float32) # shape= batch_size* max_len + 1* dim_h
        # attach h0 in the front, we concat h0= (y;z)  to the beginning of g_outputs
        # tf.expand_dims: Inserts a dimension of 1 into a tensor's shape  # Tensor("ExpandDims:0", shape=(?, 1, 700), dtype=float32)
        # self.h_ori.shape= batch_size * dim_y + dim_z # Tensor("concat_1:0", shape=(?, 700), dtype=float32)

        teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs], 1)
        # after expand_dims, self.h_ori.shape = batch_size * 1 *  dim_y + dim_z ,  # Tensor("ExpandDims:0", shape=(?, 1, 700), dtype=float32)
        # dim g_outputs: batch_size * max_len + 1 * dim_h
        #print(teach_h) # Tensor("concat_3:0", shape=(?, ?, 700), dtype=float32) # after concat, teach_h.shape = batch_size * 1 + dim_g_outputs(max_len + 1) * dim_y + dim_z
        g_outputs = tf.nn.dropout(g_outputs, self.dropout)
        #print('g_outputs',g_outputs) # batch_siz*batch_len*700
        
        g_outputs = tf.reshape(g_outputs, [-1, dim_h])
        g_logits = tf.matmul(g_outputs, proj_W) + proj_b
        # tf.reshape(t, [-1]): flattens tensor t
        # Computes sparse softmax cross entropy between logits and labels
        #print('self.targets',self.targets)
        #print('self.g_logits.shape',g_logits.shape)#[Dimension(None), Dimension(9357)]
        #print('tf.reshape(self.targets, [-1])',tf.reshape(self.targets, [-1]).shape)
        loss_rec = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=g_logits)
        #print('print(loss_rec)',loss_rec)
        loss_rec *= tf.reshape(self.weights, [-1])
        self.loss_rec = tf.reduce_sum(loss_rec) / tf.to_float(self.batch_size)
        #print('print(loss_rec)',loss_rec)
        #print('print(self.loss_rec)',self.loss_rec)

        # uptil here, AE, initialised, encoder with y, and decoder with y + z
        #####   feed-previous decoding   #####
        # two approahes are taken here for feeding (two kinds of inputs)
        go = dec_inputs[:,0,:]
        # soft_func: takes the outputs (all states of decoder) as input and returns inp = tf.matmul(prob, embedding) & also logits of the outputs
        soft_func = softsample_word(self.dropout, proj_W, proj_b, embedding,
            self.gamma)
        # hard_func: takes the outputs (all states of decoder) as input and returns(inp = embedding vector baray har kalame toolidi dar har time step) & also (logits of the outputs of the decoder)
        hard_func = argmax_word(self.dropout, proj_W, proj_b, embedding)
        # rnn_decode returns sequence of outputs (states?), with the first member as h-ori or h-tsf and sequence of logits of outputs
        token_seq_soft_ori, soft_h_ori, soft_logits_ori, soft_probs_ori, soft_log_probs_ori = rnn_decode(self.h_ori, go, max_len,
            cell_g, soft_func, scope='generator')
        # x_fake
        # return tf.concat(gen_token_seq,1), tf.concat(h_seq, 1), tf.concat(logits_seq, 1), tf.concat(probs_seq,1)
        token_seq_soft_tsf, soft_h_tsf, soft_logits_tsf, soft_probs_tsf, soft_log_probs_tsf = rnn_decode(self.h_tsf, go, max_len,
            cell_g, soft_func, scope='generator')
        token_seq_hard_ori, hard_h_ori, self.hard_logits_ori, words_ori, hard_log_probs_ori  = rnn_decode(self.h_ori, go, max_len,
            cell_g, hard_func, scope='generator')
        token_seq_hard_tsf, hard_h_tsf, self.hard_logits_tsf, words_tsf, hard_log_probs_tsf  = rnn_decode(self.h_tsf, go, max_len,
            cell_g, hard_func, scope='generator')
        print('soft_h_tsf',soft_h_tsf) #shape=(?, 16, 700)
        print('hard_h_tsf',hard_h_tsf) #shape=(?, 16, 700)
        print('hard_logits_tsf', self.hard_logits_tsf) #  shape=(?, 16, 9353)
        print('token_seq_hard_tsf',token_seq_hard_tsf, ) #len(token_seq_hard_tsf): 16, token_seq_hard_tsf[0]: shape=(?, 100)
        #('token_seq_hard_tsf', <tf.Tensor 'concat_19:0' shape=(?, 16, 100) dtype=float32>) 

        # Cyclic loss:  
        init_state_tsf =  tf.concat([linear(1-labels, dim_y, scope='encoder', reuse=True),
            tf.zeros([self.batch_size, dim_z])], 1)

        print('init_state_tsf', init_state_tsf) # shape=(?, 700)
        print('************************************************')
        gen_soft_tsf =tf.math.exp(soft_log_probs_tsf)
        gen_hard_tsf =tf.math.exp(hard_log_probs_tsf)
        print('soft_log_probs_tsf',soft_log_probs_tsf) # <tf.Tensor 'concat_13:0' shape=(?, 16, 9353) dtype=float32>)
        print('gen_soft_tsf',gen_soft_tsf) # <tf.Tensor 'Exp:0' shape=(?, 16, 9353) dtype=float32>) 
        print('************************************************')
        print('gen_hard_tsf',gen_hard_tsf) # <tf.Tensor 'Exp_1:0' shape=(?, 16, 9353) dtype=float32>
        print('hard_log_probs_tsf',hard_log_probs_tsf) # concat_23:0' shape=(?, 16, 9353) dtype=float32>

        if cyclic_soft:
            '''
                _, z_cyc = tf.nn.dynamic_rnn(cell_e, token_seq_soft_tsf ,
                    initial_state = init_state_tsf , scope='encoder') 
            else:
                _, z_cyc = tf.nn.dynamic_rnn(cell_e, token_seq_hard_tsf,
                   initial_state = init_state_tsf , scope='encoder') 
            '''
            _, z_cyc = tf.nn.dynamic_rnn(cell_e, gen_soft_tsf  ,
                initial_state = init_state_tsf , scope='encoder') 
        else:
            _, z_cyc = tf.nn.dynamic_rnn(cell_e, gen_hard_tsf,
               initial_state = init_state_tsf , scope='encoder') 

    
        self.z_cyc = z_cyc[:, dim_y:]
        self.h_cyc_tsf = tf.concat([linear(labels, dim_y,
            scope='generator', reuse=True),self.z_cyc], 1)
        g_outputs_cyc, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs,
            initial_state=self.h_cyc_tsf, scope='generator')
        g_outputs_cyc = tf.nn.dropout(g_outputs_cyc, self.dropout)
        g_outputs_cyc = tf.reshape(g_outputs_cyc, [-1, dim_h])
        g_logits_cyc = tf.matmul(g_outputs_cyc, proj_W) + proj_b
        print('token_seq_hard_tsf',  token_seq_hard_tsf)
        print('token_seq_soft_tsf', token_seq_soft_tsf)
        print('soft_probs_tsf', soft_probs_tsf)
        loss_cyc = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=g_logits_cyc)
        print('print(loss_cyc)',loss_cyc)
        loss_cyc *= tf.reshape(self.weights, [-1])
        print('self.weights', self.weights)
        self.loss_cyc = tf.reduce_sum(loss_cyc) / tf.to_float(self.batch_size)

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
        #self.loss = self.loss_rec + self.rho * self.loss_adv
        self.loss_self_rec = self.rec_factor * self.loss_rec 
        #self.loss_rec_kol = self.loss_self_rec + self.cycl_factor * self.loss_cyc
        self.loss_cycl_final = self.cycl_factor * self.loss_cyc
        self.loss =  self.loss_cycl_final + self.loss_self_rec + (self.rho * self.loss_adv) 
        theta_eg = retrive_var(['encoder', 'generator',
            'embedding', 'projection'])
        theta_d0 = retrive_var(['discriminator0'])
        theta_d1 = retrive_var(['discriminator1'])

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2)

        grad_self_rec, _ = zip(*opt.compute_gradients(self.loss_self_rec, theta_eg)) # compute_gradients: This is the first part of minimize(). It returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable"
        #grad_rec_kol, _ = zip(*opt.compute_gradients(self.loss_rec_kol, theta_eg))
        grad_adv, _ = zip(*opt.compute_gradients(self.loss_adv, theta_eg))
        grad_cycl_final,_ = zip(*opt.compute_gradients(self.loss_cycl_final, theta_eg))
        grad, _ = zip(*opt.compute_gradients(self.loss, theta_eg))
        grad, _ = tf.clip_by_global_norm(grad, grad_clip)

        self.grad_self_rec_norm = tf.global_norm(grad_self_rec)
        self.grad_cycl_final_norm = tf.global_norm(grad_cycl_final)
        self.grad_adv_norm = tf.global_norm(grad_adv)
        self.grad_norm = tf.global_norm(grad)

        self.optimize_tot = opt.apply_gradients(zip(grad, theta_eg)) #apply_gradients: This is the second part of minimize(). It returns an Operation that applies gradients.
        #self.optimize_self_rec = opt.minimize(self.loss_rec_kol , var_list=theta_eg) 
        #self.optimize_self_rec = opt.apply_gradients(zip(grad_self_rec, theta_eg)) 

        self.optimize_self_rec = opt.minimize(self.loss_self_rec, var_list=theta_eg)
        self.optimize_d0 = opt.minimize(self.loss_d0, var_list=theta_d0) # minimize: This method simply combines calls compute_gradients() and apply_gradients(), #  If you want to process the gradient before applying them call compute_gradients() and apply_gradients() explicitly instead of using this function.
        self.optimize_d1 = opt.minimize(self.loss_d1, var_list=theta_d1)

        #self.train_writer = tf.summary.FileWriter('./log')
        #self.train_writer.add_graph(sess.graph)

        self.saver = tf.train.Saver()

def transfer(model, decoder, sess, args, vocab, data0, data1, out_path):
    batches, order0, order1 = get_batches(data0, data1,
        vocab.word2id, args.batch_size, args,noisy=False)

    data0_rec, data1_rec = [], []
    data0_tsf, data1_tsf = [], []
    losses = Accumulator(len(batches), ['loss', 'rec', 'cyclic', 'adv', 'd0', 'd1'])
    for batch in batches:
        #decoder.rewrite(batch): hard logits ro tabdil be kalame mikone va az token ens be baed hazf mikone
        rec, tsf = decoder.rewrite(batch)
        half = batch['size'] / 2
        data0_rec += rec[:half]
        data1_rec += rec[half:]
        data0_tsf += tsf[:half]
        data1_tsf += tsf[half:]
        #losses.add: for in len (values= [loss, loss_rec, loss_adv, loss_d0, loss_d1]), value[i]/(batchsize) ro be har yek az moalefe ha ezafe mikonim
        loss, loss_rec, loss_cyc, loss_adv,  loss_d0, loss_d1 = sess.run([model.loss,
            model.loss_rec,model.loss_cyc, model.loss_adv, model.loss_d0, model.loss_d1],
            feed_dict=feed_dictionary(model, batch, args.rho, args.cycl_factor,args.rec_factor,args.gamma_min))
        losses.add([loss, loss_rec, loss_cyc, loss_adv, loss_d0, loss_d1])
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
                args.batch_size,args)
            random.shuffle(batches)
            start_time = time.time()
            step = 0
            losses = Accumulator(args.steps_per_checkpoint,
                ['loss', 'rec', 'cyclic', 'adv', 'd0', 'd1'])
            best_dev = float('inf')
            learning_rate = args.learning_rate
            rho = args.rho
            gamma = args.gamma_init
            dropout = args.dropout_keep_prob
            cycl_factor = args.cycl_factor
            rec_factor = args.rec_factor
            #gradients = Accumulator(args.steps_per_checkpoint,
            #    ['|grad_rec|', '|grad_adv|', '|grad|'])

            for epoch in range(1, 1+args.max_epochs):
                print( '--------------------epoch %d--------------------' % epoch)
                print ('learning_rate:', learning_rate, '  gamma:', gamma)

                for batch in batches:
                    feed_dict = feed_dictionary(model, batch, rho,cycl_factor,rec_factor, gamma,
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
                        optimize = model.optimize_self_rec

                    loss, loss_rec, loss_cyc , loss_adv, _ = sess.run([model.loss,
                        model.loss_rec, model.loss_cyc,  model.loss_adv, optimize],
                        feed_dict=feed_dict)
                    losses.add([loss, loss_rec,loss_cyc , loss_adv, loss_d0, loss_d1])

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
