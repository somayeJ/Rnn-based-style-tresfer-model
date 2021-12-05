import tensorflow as tf
from nn_tf_v_1 import *
from utils_tf_1 import strip_eos
from copy import deepcopy

class BeamState(object):
    def __init__(self, h, inp, sent, nll):
        self.h, self.inp, self.sent, self.nll = h, inp, sent, nll


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
        self.dim_z = args.dim_z
        self.vocab = vocab
        self.model = model
        self.max_len = args.max_seq_length
        self.beam_width = args.beam
        self.sess = sess

        #self.half = model.batch_size 
        #self.embedding = embedding0

        cell0 = create_cell(dim_h, n_layers, dropout=1)
        cell1 = create_cell(dim_h, n_layers, dropout=1)

        self.inp = tf.placeholder(tf.int32, [None])
        self.batch_size = tf.placeholder(tf.int32, shape=())
        #self.inp1 = tf.placeholder(tf.int32, [None])
        self.h = tf.placeholder(tf.float32, [None, dim_h])
        #self.h1 = tf.placeholder(tf.float32, [None, dim_h])
        self.z0 = tf.placeholder(tf.float32,[None, self.dim_z])
        self.z1 = tf.placeholder(tf.float32,[None, self.dim_z])


        tf.get_variable_scope().reuse_variables()
        #embedding = self.embedding
        embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        with tf.variable_scope('projection0'):
            proj_W0 = tf.get_variable('W', [dim_h+self.dim_z, vocab.size])
            proj_b0 = tf.get_variable('b', [vocab.size])

        with tf.variable_scope('generator0'):
            inp0 = tf.nn.embedding_lookup(embedding, self.inp)
            outputs0, self.h_prime0 = cell0(inp0, self.h)
            outputs0 = tf.concat([outputs0,self.z0],1)
            logits0 = tf.matmul(outputs0, proj_W0) + proj_b0
            log_lh0 = tf.log(tf.nn.softmax(logits0))
            self.log_lh0, self.indices0 = tf.nn.top_k(log_lh0, self.beam_width)
        with tf.variable_scope('projection1'):
            proj_W1 = tf.get_variable('W', [dim_h+self.dim_z, vocab.size])
            proj_b1 = tf.get_variable('b', [vocab.size])

        with tf.variable_scope('generator1'):
            inp1 = tf.nn.embedding_lookup(embedding, self.inp)
            outputs1, self.h_prime1 = cell1(inp1, self.h)
            outputs1 = tf.concat([outputs1,self.z1],1)
            logits1 = tf.matmul(outputs1, proj_W1) + proj_b1
            log_lh1= tf.log(tf.nn.softmax(logits1))
            self.log_lh1, self.indices1 = tf.nn.top_k(log_lh1, self.beam_width)
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
    def decode0(self, h,z0,batch):
        go = self.vocab.word2id['<go>']
        batch_size = int(len(h))
        init_state = BeamState(h, [go] * batch_size,
            [[] for i in range(batch_size)], [0] * batch_size)
        beam = [init_state]

        for t in range(self.max_len):
            exp = [[] for i in range(batch_size)]
            for state in beam:
                log_lh, indices, h = self.sess.run(
                    [self.log_lh0, self.indices0, self.h_prime0],
                    feed_dict={self.inp: state.inp, self.h: state.h, self.z0:z0, self.batch_size: batch['size']})
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
    def decode1(self, h,z1,batch):
        go = self.vocab.word2id['<go>']
        batch_size = int(len(h))
        init_state = BeamState(h, [go] * batch_size,
            [[] for i in range(batch_size)], [0] * batch_size)
        beam = [init_state]

        for t in range(self.max_len):
            exp = [[] for i in range(batch_size)]
            for state in beam:
                log_lh, indices, h = self.sess.run(
                    [self.log_lh1, self.indices1, self.h_prime1],
                    feed_dict={self.inp: state.inp, self.h: state.h, self.z1:z1, self.batch_size: batch['size']})
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
            z0,z1,h_ori0, h_tsf0,  h_ori1, h_tsf1= self.sess.run([model.z0,model.z1,model.h_ori0, model.h_tsf0, model.h_ori1, model.h_tsf1 ],
                feed_dict={model.dropout: 1,
                           model.batch_size: batch['size'],
                           model.enc_inputs: batch['enc_inputs'],
                           model.labels: batch['labels'],
                           model.len:batch['len']})


        ori0 = self.decode0(h_ori0,z0,batch)
        ori0 = [[self.vocab.id2word[i] for i in sent] for sent in ori0]
        ori0 = strip_eos(ori0)
        tsf0 = self.decode1(h_tsf0,z0,batch)
        tsf0 = [[self.vocab.id2word[i] for i in sent] for sent in tsf0]
        tsf0 = strip_eos(tsf0)

        ori1 = self.decode1(h_ori1,z1,batch)
        ori1 = [[self.vocab.id2word[i] for i in sent] for sent in ori1]
        ori1 = strip_eos(ori1)
        tsf1 = self.decode0(h_tsf1,z1,batch)
        tsf1 = [[self.vocab.id2word[i] for i in sent] for sent in tsf1]
        tsf1 = strip_eos(tsf1)

        return ori0, tsf0, ori1, tsf1

