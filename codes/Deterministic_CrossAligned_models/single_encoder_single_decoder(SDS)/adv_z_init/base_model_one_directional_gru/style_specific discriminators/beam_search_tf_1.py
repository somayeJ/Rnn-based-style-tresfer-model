import tensorflow as tf
from nn_tf_v_1 import *
from utils_tf_1 import strip_eos
from copy import deepcopy

class BeamState(object):
    def __init__(self, h, inp, sent, nll):
        self.h, self.inp, self.sent, self.nll = h, inp, sent, nll

class Decoder_modified(object):

    def __init__(self, sess, args, vocab, model, embedding):
        self.extra_features= args.extra_features
        self.dim_token_feat = args.dim_token_feat
        if self.extra_features:

            dim_h = args.dim_y + args.dim_z + (self.dim_token_feat)
        else:
            dim_h = args.dim_y + args.dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        self.vocab = vocab
        self.model = model
        self.max_len = args.max_seq_length
        self.beam_width = args.beam
        self.sess = sess
        self.extra_features= args.extra_features

        cell = create_cell(dim_h, n_layers, dropout=1)

        self.inp = tf.placeholder(tf.int32, [None])
        self.h = tf.placeholder(tf.float32, [None, dim_h])
        # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        #     #tf.get_variable_scope().reuse_variables()
        #     embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        with tf.variable_scope('projection', reuse=tf.AUTO_REUSE):
            if args.extra_features:
                proj_W = tf.get_variable('W', [dim_h, vocab.size])
            else:
                proj_W = tf.get_variable('W', [dim_h , vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        with tf.variable_scope('generator'):

            if self.extra_features:
                inp0 = tf.nn.embedding_lookup(embedding, self.inp)
                print('helloooooo', inp0.shape, self.h.shape)
                no_feats= args.dim_token_feat
                token_extra_features = [[-2.0] * (no_feats)]
                batch_size = tf.shape(inp0)[0]
                token_extra_features = tf.tile(token_extra_features, [batch_size, 1])
                inp = tf.concat([inp0, token_extra_features], -1)
            else:
                inp = tf.nn.embedding_lookup(embedding, self.inp)
            outputs, self.h_prime = cell(inp, self.h)
            logits = tf.matmul(outputs, proj_W) + proj_b
            log_lh = tf.log(tf.nn.softmax(logits))
            self.log_lh, self.indices = tf.nn.top_k(log_lh, self.beam_width)

    def decode(self, h):
        # a batch of sentences are passed to this function
        go = self.vocab.word2id['<go>']
        batch_size = len(h)
        init_state = BeamState(h, [go] * batch_size,
            [[] for i in range(batch_size)], [0] * batch_size)
        beam = [init_state]

        for t in range(self.max_len):
            exp = [[] for i in range(batch_size)]
            for state in beam:
                if self.extra_features:

                    log_lh, indices, h = self.sess.run(
                        [self.log_lh, self.indices, self.h_prime],
                        feed_dict={self.inp: state.inp, self.h: state.h})
                else:
                    log_lh, indices, h = self.sess.run(
                        [self.log_lh, self.indices, self.h_prime],
                        feed_dict={self.inp: state.inp, self.h: state.h})

                for i in range(batch_size):
                    for l in range(self.beam_width):
                        exp[i].append(BeamState(h[i], indices[i,l],
                            state.sent[i] + [indices[i,l]],
                            state.nll[i] - log_lh[i,l]))

            beam = [deepcopy(init_state) for _ in range(self.beam_width)]
            for i in range(batch_size):
                a = sorted(exp[i], key=lambda k: k.nll)
                for k in range(self.beam_width):
                    beam[k].h[i] = a[k].h
                    beam[k].inp[i] = a[k].inp
                    beam[k].sent[i] = a[k].sent
                    beam[k].nll[i] = a[k].nll
        return beam[0].sent

    def rewrite(self, batch):
        model = self.model
        # h_ori: y + last hidden layer of the encoder
        # h_tsf: transferred_y + last hidden layer of the encoder
        if self.extra_features:
            h_ori, h_tsf = self.sess.run([model.h_ori, model.h_tsf],
                feed_dict = {model.dropout: 1,
                           # batch['size']: len (batch)
                           model.batch_size: batch['size'],
                           # batch['enc_inputs']: inputs which are reversed which means first padding are added then the sentence tokens
                           model.enc_inputs: batch['enc_inputs'],
                           model.labels: batch['labels'],
                             model.tokens_features: batch['token_features_enc'],
                             model.tokens_features_dec: batch['token_features_dec'],
                             model.tokens_features_target: batch['token_features_target']
                             })
        else:
            h_ori, h_tsf = self.sess.run([model.h_ori, model.h_tsf],
                feed_dict = {model.dropout: 1,
                           # batch['size']: len (batch)
                           model.batch_size: batch['size'],
                           # batch['enc_inputs']: inputs which are reversed which means first padding are added then the sentence tokens
                           model.enc_inputs: batch['enc_inputs'],
                           model.labels: batch['labels']})
        ori = self.decode(h_ori)
        ori = [[self.vocab.id2word[i] for i in sent] for sent in ori]
        ori = strip_eos(ori)
        '''
        ori_concat = [' '.join(sent) for sent in ori]
        f= open('ori_gen.txt','a')
        for s in ori_concat:
            f.write(s)
        f.close()
        # 
        '''
        tsf = self.decode(h_tsf)
        tsf = [[self.vocab.id2word[i] for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)
        '''
        tsf_concat = [' '.join(sent) for sent in tsf]
        f= open('tsf_gen.txt','a')
        for s in tsf_concat:
            f.write(s)
        f.close()
        '''
        return ori, tsf

class Decoder(object):

    def __init__(self, sess, args, vocab, model):
        self.elmo_seq_rep = args.elmo_seq_rep
        if self.elmo_seq_rep:
            dim_h =3072 + args.dim_y
        else:
            dim_h = args.dim_y + args.dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        self.dim_y = args.dim_y
        self.vocab = vocab
        self.model = model
        self.max_len = args.max_seq_length
        self.beam_width = args.beam
        self.sess = sess
        #self.embedding = embedding0

        cell = create_cell(dim_h, n_layers, dropout=1)

        self.inp = tf.placeholder(tf.int32, [None])
        self.h = tf.placeholder(tf.float32, [None, dim_h])

        tf.get_variable_scope().reuse_variables()
        #embedding = self.embedding
        embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        with tf.variable_scope('generator'):
            inp = tf.nn.embedding_lookup(embedding, self.inp)
            outputs, self.h_prime = cell(inp, self.h)
            logits = tf.matmul(outputs, proj_W) + proj_b
            log_lh = tf.log(tf.nn.softmax(logits))
            self.log_lh, self.indices = tf.nn.top_k(log_lh, self.beam_width)

    def decode(self, h):
        go = self.vocab.word2id['<go>']
        batch_size = len(h)
        init_state = BeamState(h, [go] * batch_size,
            [[] for i in range(batch_size)], [0] * batch_size)
        beam = [init_state]

        for t in range(self.max_len):
            exp = [[] for i in range(batch_size)]
            for state in beam:
                log_lh, indices, h = self.sess.run(
                    [self.log_lh, self.indices, self.h_prime],
                    feed_dict={self.inp: state.inp, self.h: state.h})
                for i in range(batch_size):
                    for l in range(self.beam_width):
                        exp[i].append(BeamState(h[i], indices[i,l],
                            state.sent[i] + [indices[i,l]],
                            state.nll[i] - log_lh[i,l]))

            beam = [deepcopy(init_state) for _ in range(self.beam_width)]
            for i in range(batch_size):
                a = sorted(exp[i], key=lambda k: k.nll)
                for k in range(self.beam_width):
                    beam[k].h[i] = a[k].h
                    beam[k].inp[i] = a[k].inp
                    beam[k].sent[i] = a[k].sent
                    beam[k].nll[i] = a[k].nll
        return beam[0].sent

    def rewrite(self, batch):
        model = self.model
        if self.elmo_seq_rep:
            h_ori, h_tsf = self.sess.run([model.h_ori, model.h_tsf],
                feed_dict={model.dropout: 1,
                           model.batch_size: batch['size'],
                           model.enc_inputs: batch['enc_inputs'],
                           model.labels: batch['labels'],
                           model.elmo_emb: batch['elmo_embeddings']})


        else:
            h_ori, h_tsf = self.sess.run([model.h_ori, model.h_tsf],
                feed_dict={model.dropout: 1,
                           model.batch_size: batch['size'],
                           model.enc_inputs: batch['enc_inputs'],
                           model.labels: batch['labels']})

        ori = self.decode(h_ori)
        ori = [[self.vocab.id2word[i] for i in sent] for sent in ori]
        ori = strip_eos(ori)
        tsf = self.decode(h_tsf)
        tsf = [[self.vocab.id2word[i] for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)

        return ori, tsf

'''
    def decode(self, h):
        go = self.vocab.word2id['<go>']
        batch_size = len(h)
        init_state = BeamState(h, [go] * batch_size,
            [[] for i in range(batch_size)], [0] * batch_size)
        beam = [init_state]

        for t in range(self.max_len):
            exp = [[] for i in range(batch_size)]
            for state in beam:
                log_lh, indices, h = self.sess.run(
                    [self.log_lh, self.indices, self.h_prime],
                    feed_dict={self.inp: state.inp, self.h: state.h})
                for i in range(batch_size):
                    for l in range(self.beam_width):
                        exp[i].append(BeamState(h[i], indices[i,l],
                            state.sent[i] + [indices[i,l]],
                            state.nll[i] - log_lh[i,l]))

            beam = [deepcopy(init_state) for _ in range(self.beam_width)]
            for i in range(batch_size):
                a = sorted(exp[i], key=lambda k: k.nll)
                for k in range(self.beam_width):
                    beam[k].h[i] = a[k].h
                    beam[k].inp[i] = a[k].inp
                    beam[k].sent[i] = a[k].sent
                    beam[k].nll[i] = a[k].nll
        return beam[0].sent

    def rewrite(self, batch):
        model = self.model
        new_model = self.new_model
        h_ori, h_tsf= self.sess.run([new_model.h_ori, new_model.h_tsf],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.enc_inputs: batch['enc_inputs'],
                       model.labels: batch['labels']}) # az it can be seen, the z which is the output of untrained model is fed to the decoder
        ori = self.decode(h_ori)
        ori = [[self.vocab.id2word[i] for i in sent] for sent in ori]
        ori = strip_eos(ori)

        tsf = self.decode(h_tsf)
        tsf = [[self.vocab.id2word[i] for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)
        return ori, tsf
'''