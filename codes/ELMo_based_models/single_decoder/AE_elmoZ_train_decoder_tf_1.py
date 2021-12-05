# this is made based on AE_elmoZ_trained_model.py, in which the changes in the model are made to investigate the
# performance of the decoder after feeding the elmo embeddings, in this code, we want to train the decoder feeding it
# with elmo rep of the yelp data, this is the same as the code AE_elmoZ_train_decoder.py, just it is run by tf v1, so calling the according functions and libraries
import os
import sys
import time
import ipdb
import random
import cPickle as pickle
#import _pickle  as pickle
import numpy as np
import tensorflow as tf
#from tensorboardX import SummaryWriter
from vocab import Vocabulary, build_vocab
from accumulator import Accumulator
from options import load_arguments
from file_io import load_sent, write_sent,load_sent_full
from utils_tf_1 import *
from nn_tf_v_1 import *
import beam_search_tf_1, greedy_decoding_tf_1
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Model(object):
    def __init__(self, args, vocab):
        extra_features = args.extra_features
        elmo_seq_rep = args.elmo_seq_rep
        test_arg = args.test
        train_arg = args.train
        # y: style, 200
        dim_y = args.dim_y
        # z = dim_h- dim_y , 500
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        # 100
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        max_len = args.max_seq_length
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        n_filters = args.n_filters
        beta1, beta2 = 0.5, 0.999
        grad_clip = 30.0
        # we fill in these parts in feed_dictionary dictionary
        with tf.name_scope('dropout'):
            self.dropout = tf.placeholder(tf.float32,
                name='dropout')
            tf.summary.scalar('dropout_rate',self.dropout)
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
        self.tokens_features = tf.placeholder(tf.float32, [None, None],
            name = 'token_features')
        self.elmo_emb = tf.placeholder(tf.float32, [None, None],
            name = 'elmo_emb')

        labels = tf.reshape(self.labels, [-1, 1])
        # vocab.embedding: it is random initialization of embeddding matrix
        # get_variable: Gets an existing variable with these parameters or create a new one 
        #(I think we use "get_variable" for the first time it creates and for the rest of the times it gets this existing variable).
        with  tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            self.embedding = tf.get_variable('embedding',
                initializer=vocab.embedding.astype(np.float32))
        # 'projection', scope in which logits are made
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [3272, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        enc_inputs = tf.nn.embedding_lookup(self.embedding, self.enc_inputs,'enc_inputs')
        dec_inputs = tf.nn.embedding_lookup(self.embedding, self.dec_inputs,'dec_inputs')

        #####   auto-encoder   #####
        # init_state of encoder is style (y)
        # linear(labels, dim_y): tensor with dim labels.shape(0) * dim_y, 
        # labels.shape(0) = batch_size.shape , since labels is the list of styles of each input
        with  tf.name_scope('encoder'):
            init_state = tf.concat([linear(labels, dim_y, scope='encoder'),
                tf.zeros([self.batch_size, 3072])], 1)
            # create GRU cell
            # dim_h = dim_y + dim_z
            with tf.name_scope('dropout'):
                cell_e = create_cell(3272, n_layers, self.dropout)
            # z = last hidden layer of the encoder, z.shape: (batch_size, dim_h)
            _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
                initial_state=init_state, scope='encoder')
            # for z we consider, just the z part and reduce the style part
        z = z[:, dim_y:]
        # cell_e = create_cell(dim_z, n_layers, self.dropout)
        # _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
        #    dtype=tf.float32, scope='encoder')

        with tf.name_scope('generator'):
            self.h_ori = tf.concat([linear(labels, dim_y,
                scope='generator'), self.elmo_emb], 1)
            self.h_tsf = tf.concat([linear(1-labels, dim_y,
                scope='generator', reuse=True),  self.elmo_emb], 1)
            cell_g = create_cell(3272, n_layers, self.dropout)

            # sequence of hidden layers of decoder
            g_outputs, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs,
                initial_state=self.h_ori, scope='generator')
            # attach h0 in the front, we concat h0= (y,z) to the beginning of g_outputs
            # tf.expand_dims: Inserts a dimension of 1 into a tensor's shape
            # self.h_ori.shape= batch_size , dim_y + dim_z 
            teach_h = tf.concat([tf.expand_dims(self.h_ori, 1), g_outputs], 1)
            # after expand_dims, self.h_ori.shape = batch_size , 1 ,  dim_y + dim_z 
            # after concat, self.h_ori.shape = batch_size , 1 , dim_y + dim_z + dim_g_outputs(dim_h)
            g_outputs = tf.nn.dropout(g_outputs, self.dropout)
            g_outputs = tf.reshape(g_outputs, [-1, 3272])

            g_logits = tf.matmul(g_outputs, proj_W) + proj_b
            # tf.reshape(t, [-1]): flattens tensor t
            # Computes sparse softmax cross entropy between logits and labels
            with tf.name_scope('rec_loss'):
                loss_rec = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(self.targets, [-1]), logits=g_logits)
                loss_rec *= tf.reshape(self.weights, [-1])
                self.loss_rec = tf.reduce_sum(loss_rec) / tf.to_float(self.batch_size)
            #tf.summary.scalar('rec_loss',self.loss_rec)
            # uptil here , AE, initialised, encoder with y+z(z=[0]), and decoder with y + z
            #####   feed-previous decoding   #####
            # two approahes are taken here for feeding (two kinds of inputs)
            go = dec_inputs[:,0,:]
            # soft_func: takes the outputs (all states of decoder) as input and returns
            #inp = tf.matmul(prob (prob = gumbel_softmax(logits, gamma), which is softmax(logits+sample _uniform)), embedding) # & also logits of the outputs
            soft_func = softsample_word(self.dropout, proj_W, proj_b, self.embedding,
                self.gamma)
            # hard_func: takes the outputs (all states of decoder) as input and returns(inp = embedding vector baray har kalame toolidi dar har time step)
            # & also (logits of the outputs of the decoder)
            hard_func = argmax_word(self.dropout, proj_W, proj_b, self.embedding)
            #soft_h_ori, soft_logits_ori = rnn_decode(self.h_ori, go, max_len,
                #cell_g, soft_func, scope='generator')
            # x_fake
            #soft_h_tsf, soft_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
                #cell_g, soft_func, scope='generator')

            # rnn_decode returns the tf.concat(h_seq, 1), tf.concat(logits_seq, 1), h_seq is the seq of the  outputs of the decoder
            # where the first elemnt is the first param passed to the method, logits_seq is the sequence of outputs of the decoder with the size of max_len
            # the first param is the initializer of the cell_g and the secod is the input
            '''
            hard_h_ori, self.hard_logits_ori = rnn_decode(self.h_ori, go, max_len,
                cell_g, hard_func, scope='generator')
            hard_h_tsf, self.hard_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
                cell_g, hard_func, scope='generator')
            '''

            hard_h_ori, self.hard_logits_ori = rnn_decode(self.h_ori, go, max_len,
                cell_g, hard_func, scope = 'generator')
            hard_h_tsf, self.hard_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
                cell_g, hard_func,scope='generator')


        #####   discriminator   #####
        # a batch's first half consists of sentences of one style,
        # and second half of the other
        # half = self.batch_size / 2
        # zeros, ones = self.labels[:half], self.labels[half:]
        # soft_h_tsf = soft_h_tsf[:, :1+self.batch_len, :]

        # self.loss_d0, loss_g0 = discriminator(teach_h[:half], soft_h_tsf[half:],
        #     ones, zeros, filter_sizes, n_filters, self.dropout,
        #     scope='discriminator0')
        # self.loss_d1, loss_g1 = discriminator(teach_h[half:], soft_h_tsf[:half],
        #     ones, zeros, filter_sizes, n_filters, self.dropout,
        #     scope='discriminator1')

        #####   optimizer   #####
        # self.loss_adv = loss_g0 + loss_g1
        # self.loss = self.loss_rec + self.rho * self.loss_adv
        #theta_eg = retrive_var(['encoder', 'generator', 'embedding', 'projection'])
        theta_g = retrive_var([ 'generator',  'projection'])
        # theta_d0 = retrive_var(['discriminator0'])
        # theta_d1 = retrive_var(['discriminator1'])
        with tf.name_scope('train'):
            opt = tf.train.AdamOptimizer(self.learning_rate, beta1, beta2)
            grad_rec, _ = zip(*opt.compute_gradients(self.loss_rec, theta_g))
            # grad_adv, _ = zip(*opt.compute_gradients(self.loss_adv, theta_eg))
            # grad, _ = zip(*opt.compute_gradients(self.loss, theta_eg))
            # grad, _ = tf.clip_by_global_norm(grad, grad_clip)
            # grad, _ = tf.clip_by_global_norm(grad, grad_clip)

            self.grad_rec_norm = tf.global_norm(grad_rec)
            # self.grad_adv_norm = tf.global_norm(grad_adv)
            # self.grad_norm = tf.global_norm(grad)
            # self.optimize_tot = opt.apply_gradients(zip(grad, theta_eg))

            self.optimize_rec = opt.minimize(self.loss_rec, var_list=theta_g)
            # self.optimize_d0 = opt.minimize(self.loss_d0, var_list=theta_d0)
            # self.optimize_d1 = opt.minimize(self.loss_d1, var_list=theta_d1)

        #self.merged= tf.summary.merge_all()
        #self.writer = tf.summary.FileWriter('./log')
        #self.writer.add_graph(sess.graph)
        self.saver = tf.train.Saver()

# returns batches consisting of several batch: each batch consists of half sentences from the first corpus with label 0
# & the other half from the other corpus, order0  && order1 (2nd & 3rd argument) show the indices of the sentences in the corpora
def transfer(model, decoder, sess, args, vocab, data0, data1, dev0_full, dev1_full, out_path):
    """
    :param model:
    :param decoder:
    :param sess:
    :param args:
    :param vocab:
    :param data0:
    :param data1:
    :param dev0_full:
    :param dev1_full:
    :param out_path:
    :return:
    """
    if args.elmo_seq_rep:

        if args.test:
            batches, order0, order1 = get_batches_elmo(data0, data1,
                vocab.word2id, args.batch_size, ".test.", args.elmo_seq_rep)
        elif args.dev:
            batches, order0, order1 = get_batches_elmo(data0, data1,
                vocab.word2id, args.batch_size, ".dev.", args.elmo_seq_rep)

    else:
        batches, order0, order1 = get_batches(data0, data1,
                vocab.word2id, args.batch_size, args.extra_features)

    data0_rec, data1_rec = [], []
    data0_tsf, data1_tsf = [], []
    #losses = Accumulator(len(batches), ['loss', 'rec', 'adv', 'd0', 'd1'])
    losses = Accumulator(len(batches), ['rec'])
    for batch in batches:
        # rewrite returns the reconstructed sentences and the transferred sentences of the batch by calling runn_decode through hard_logits_ori & hard_logits_tsf
        rec, tsf = decoder.rewrite(batch)
        half = int(batch['size'] / 2)
        data0_rec += rec[:half]
        data1_rec += rec[half:]
        data0_tsf += tsf[:half]
        data1_tsf += tsf[half:]
        # losses.add: for in len (values= [loss, loss_rec, loss_adv, loss_d0, loss_d1]), value[i]/(batchsize) ro be har yek az moalefe ha ezafe mikonim
        # loss, loss_rec, loss_adv, loss_d0, loss_d1 = sess.run([model.loss,
        # model.loss_rec, model.loss_adv, model.loss_d0, model.loss_d1],
        # feed_dict=feed_dictionary(model, batch, args.rho, args.gamma_min))
        loss_rec = sess.run([model.loss_rec],
            feed_dict=feed_dictionary(model, batch, args.rho, args.gamma_min))
        #losses.add([loss, loss_rec, loss_adv, loss_d0, loss_d1])
        losses.add(loss_rec)
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
    import tensorflow.compat.v1 as tf 
    tf.disable_v2_behavior()
    args = load_arguments()
    #####   data preparation   #####
    if args.train:
        # load_sent_full returns full sentences
        train0_full = load_sent_full(args.train + '.1', args.max_train_size)
        train1_full = load_sent_full(args.train + '.1', args.max_train_size)
        # load_sent returns tokenized sentences
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        print ('#sents of training file 0:', len(train0))
        print ('#sents of training file 1:', len(train1))
        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)
    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print ('vocabulary size:', vocab.size)

    if args.dev:
        dev0_full = load_sent_full(args.dev + '.0')
        dev1_full = load_sent_full(args.dev + '.1')
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')

    if args.test:
        test0_full = load_sent_full(args.test + '.0')
        test1_full = load_sent_full(args.test + '.1')
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)
        # This decoder is inference decoder
        if args.beam > 1:
            #decoder = beam_search.Decoder(sess, args, vocab, model, embedding=model.embedding)
            #decoder = beam_search_tf_1.Decoder(sess, args, vocab, model) # tu train in shekli bood 
            decoder = beam_search_tf_1.Decoder(sess, args, vocab, model, model.embedding)

        else:
            decoder = greedy_decoding_tf_1.Decoder(sess, args, vocab, model)

        if args.train:
            # get_batches: returns a list consisting of several dicts, each corresponding to one batch: dictionaries have lists, like enc_inputs,
            # and the dic lists lengths consits of  half sentences from the first corpus with label 0 which are padded and words replaced with their ids
            # & the other half from the other corpus, order0 && order1 show the indices of the sentences in the corpora
            #batches, _, _ = get_batches(train0, train1, train0_full, train1_full, vocab.word2id,
                #args.batch_size, args.extra_features, noisy=True)
            if args.elmo_seq_rep:
                batches, _, _ = get_batches_elmo(train0, train1, vocab.word2id,
                                            args.batch_size, ".train.", args.elmo_seq_rep,noisy=True)
            else:
                batches, _, _ = get_batches(train0, train1,  vocab.word2id,
                        args.batch_size, noisy=True)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            # losses = Accumulator(args.steps_per_checkpoint,
            # ['loss', 'rec', 'adv', 'd0', 'd1'])
            losses =  Accumulator(args.steps_per_checkpoint,
                [ 'rec'])
            best_dev = float('inf')
            learning_rate = args.learning_rate
            rho = args.rho
            gamma = args.gamma_init
            dropout = args.dropout_keep_prob
            #gradients = Accumulator(args.steps_per_checkpoint,
            #    ['|grad_rec|', '|grad_adv|', '|grad|'])
            log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
            print("TensorBoard event log path: {}".format(log_event_path))

            for epoch in range(1, 1+args.max_epochs):
                print ('--------------------epoch %d--------------------' % epoch)
                print ('learning_rate:', learning_rate, '  gamma:', gamma)
                # in feed_dict = feed_dictionary(model, batch, rho, gamma,
                #                         dropout, learning_rate), everything is assigned to the model parameters, for ex: model.enc_inputs: batch['enc_inputs']
                for batch in batches:
                    feed_dict = feed_dictionary(model, batch,  rho, gamma,
                        dropout, learning_rate)
                    # loss_d0, _ = sess.run([model.loss_d0, model.optimize_d0],
                    #     feed_dict=feed_dict)
                    # loss_d1, _ = sess.run([model.loss_d1, model.optimize_d1],
                    #     feed_dict=feed_dict)

                    # # do not back-propagate from the discriminator
                    # # when it is too poor
                    # if loss_d0 < 1.2 and loss_d1 < 1.2:
                    #     optimize = model.optimize_tot
                    # else:
                    #     optimize = model.optimize_rec

            #loss, loss_rec, loss_adv, _ = sess.run([model.loss,
                        #model.loss_rec, model.loss_adv, optimize],
                    optimize = model.optimize_rec
                    loss_rec, _ = sess.run([model.loss_rec, optimize],
                        feed_dict=feed_dict)
                    #model.writer.add_summary(summary, step)
                    losses.add([loss_rec])
                    #grad_rec, grad_adv, grad = sess.run([model.grad_rec_norm,
                    #    model.grad_adv_norm, model.grad_norm],
                    #    feed_dict=feed_dict)
                    #gradients.add([grad_rec, grad_adv, grad])
                    step += 1
                    if step % args.steps_per_checkpoint == 0:
                        # yani faghat dar mazareb e steos_per_checkpoint khorooji ro chap mikone
                        losses.output('step %d, time %.0fs,'
                            % (step, time.time() - start_time))
                        #tf.summary.scalar('loss_rec',loss_rec)
                        losses.clear()
                        #gradients.output()
                        #gradients.clear()
                if args.dev:
                    dev_losses = transfer(model, decoder, sess, args, vocab,
                        dev0, dev1, dev0_full, dev1_full, args.output + '.epoch%d' % epoch)
                    dev_losses.output('dev')
                    if dev_losses.values[0] < best_dev:
                        best_dev = dev_losses.values[0]
                        print ('saving model...')
                        model.saver.save(sess, args.model)
                gamma = max(args.gamma_min, gamma * args.gamma_decay)
        #merged = tf.summary.merge_all()
        if args.test:
            test_losses = transfer(model, decoder, sess, args, vocab,
                test0, test1, test0_full, test1_full, args.output)
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


