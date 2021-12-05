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
        att_hidden_size= 500
        dim_h = args.dim_y + args.dim_z
        dim_proj = dim_h 
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        self.dim_y = args.dim_y
        self.vocab = vocab
        self.model = model
        self.max_len = args.max_seq_length
        self.beam_width = args.beam
        self.sess = sess
        self.src_contexts = tf.placeholder(tf.float32, [ None,None,700],
            name='src_contexts')
        self.h0 = tf.placeholder(tf.float32, [None,700],
            name='z')
        self.weights_input = tf.placeholder(tf.float32, [None, None],
            name='weights_input')
        self.batch_len_input = tf.placeholder(tf.int32,
            name='batch_len_input')
        self.batch_size = tf.placeholder(tf.int32,
            name='batch_size')
        #self.embedding = embedding0

        cell = create_cell(dim_h, n_layers, dropout=1)

        self.inp = tf.placeholder(tf.int32, [None])
        self.h = tf.placeholder(tf.float32, [None, dim_h])

        tf.get_variable_scope().reuse_variables()
        #embedding = self.embedding
        embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_proj, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])
        '''
        with tf.variable_scope('attention'):
            W_score1 = tf.get_variable('W_score1', [dim_h+2*350, att_hidden_size], tf.float32, tf.random_normal_initializer())
            b_score1 = tf.get_variable('b_score1', [att_hidden_size], tf.float32, tf.zeros_initializer())
            W_score2 = tf.get_variable('W_score2', [att_hidden_size, 1], tf.float32, tf.random_normal_initializer())
        '''
        with tf.variable_scope('attention'):
            W_score1 = tf.get_variable('W_score1', [dim_h,att_hidden_size ])
            W_score2 = tf.get_variable('W_score2', [dim_h, att_hidden_size])
            
            #b_score1 = tf.get_variable('b_score1', [att_hidden_size])
            #b_score2 = tf.get_variable('b_score2', [att_hidden_size])

            W_score3 = tf.get_variable('W_score3', [att_hidden_size, 1])
            #b_score3 = tf.get_variable('b_score3', [1], tf.float32, tf.zeros_initializer())

        with tf.variable_scope('generator'):
            inp = tf.nn.embedding_lookup(embedding, self.inp)
            expanded_query = tf.expand_dims(self.h0, 1)
            expanded_query1=tf.reshape(tf.matmul(tf.reshape(expanded_query, [ self.batch_size*1, dim_h ]), W_score1),[self.batch_size,1, att_hidden_size])
            src_context1=tf.reshape(tf.matmul(tf.reshape(self.src_contexts, [ self.batch_size*self.batch_len_input, dim_h ]), W_score2),[self.batch_size, self.batch_len_input, att_hidden_size])

            pre_scores = tf.tanh(expanded_query1+src_context1)
            score_h = tf.matmul(tf.reshape(pre_scores, [self.batch_size*self.batch_len_input, att_hidden_size]),W_score3)
            #expanded_tgt = tf.tile(tf.reshape(self.h0, [ self.batch_size,1, dim_h]), [ 1, self.batch_len_input, 1 ])
            scores = tf.reshape(score_h, [ self.batch_size* self.batch_len_input ])
            
            #score_h.shape: [batch_size* batch_len_input, att_hidden_size]
            #scores.shape [ batch_size*  self.batch_len ])
    
            # src_mask:
            src_mask = tf.reshape(self.weights_input, [ self.batch_size, self.batch_len_input ])
            #src_mask.shape =  [batch_size,1, batch_len_input]
            scores *= tf.reshape(src_mask,[-1])
            scores = tf.reshape(scores,[ self.batch_size, self.batch_len_input])
            att = tf.nn.softmax(scores,axis=1)
            expanded_att = tf.tile(tf.reshape(att, [ self.batch_size,self.batch_len_input, 1]), [1, 1,  2*350 ])
            print('expanded_att.shape',expanded_att.shape)#: [batch_size, batch_len_input, 2*250]
            print('src_contexts',self.src_contexts)
            context_vector = tf.reduce_sum(expanded_att*self.src_contexts, axis=1)
            print(context_vector.shape)#: (?, 1,  500)?

            #x = tf.concat([tf.expand_dims(context_vector, 1),inp], axis=-1) 
            x = tf.concat([tf.reshape(context_vector,[-1,700]),inp], axis=1) 
            self.outputs, self.h_prime = cell(x, self.h)
            #batch_size = self.h.shape[0]

            logits = tf.matmul(self.outputs, proj_W) + proj_b
            log_lh = tf.log(tf.nn.softmax(logits))
            self.log_lh, self.indices = tf.nn.top_k(log_lh, self.beam_width)
    #it returns values && inices:
    #values: The k=self.beam_width largest elements along each last dimensional slice. 
    #indices: The k=self.beam_width indices of values within the last dimension of input.
    def decode(self, h, batch,src_contexts,z):
        go = self.vocab.word2id['<go>']
        batch_size = len(h)
        #BeamState(self, h, inp, sent, nll)
        init_state = BeamState(h, [go] * batch_size,
            [[] for i in range(batch_size)], [0] * batch_size)
        beam = [init_state]
        for t in range(self.max_len):
            exp = [[] for i in range(batch_size)]
            for state in beam:
                if t==0:
                    log_lh, indices, h, output = self.sess.run(
                        [self.log_lh, self.indices, self.h_prime, self.outputs],
                        feed_dict={self.inp: state.inp, self.h: state.h,
                         self.batch_len_input:batch['len_input'], 
                         self.weights_input: batch['weights_input'],
                         self.batch_size:batch['size'],
                         self.src_contexts:src_contexts,
                         self.h0:z})
                else:
                    log_lh, indices, h, output = self.sess.run(
                        [self.log_lh, self.indices, self.h_prime, self.outputs],
                        feed_dict={self.inp: state.inp, self.h: state.h,
                         self.batch_len_input:batch['len_input'], 
                         self.weights_input: batch['weights_input'],
                         self.batch_size:batch['size'],
                         self.src_contexts:src_contexts,
                         self.h0:output})
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
        h_ori, h_tsf ,src_contexts_ori, src_contexts_tsf,z_ori,z_tsf= self.sess.run([model.h_ori, model.h_tsf,model.src_contexts0_ori,model.src_contexts0_tsf, model.z1_ori,model.z1_tsf],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.enc_inputs: batch['enc_inputs'],
                       model.labels: batch['labels'],
                       model.weights_input: batch['weights_input'],
                       model.batch_len_input : batch['len_input']
                       })

        ori = self.decode(h_ori,batch,src_contexts_ori,z_ori)
        ori = [[self.vocab.id2word[i] for i in sent] for sent in ori]
        ori = strip_eos(ori)
        tsf = self.decode(h_tsf,batch,src_contexts_tsf,z_tsf)
        tsf = [[self.vocab.id2word[i] for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)

        return ori, tsf


