import numpy as np
from utils_tf_1 import strip_eos

class Decoder(object):

    def __init__(self, sess, args, vocab, model):
        self.sess, self.vocab, self.model, self.args = sess, vocab, model,args

    def rewrite(self, batch):
        half = batch['size'] / 2
        print(11111,len(batch['enc_inputs']) , len( batch['labels']))
        model = self.model
        logits_ori0, logits_tsf0, logits_ori1, logits_tsf1 = self.sess.run(
            [model.hard_logits_ori0, model.hard_logits_tsf0, model.hard_logits_ori1, model.hard_logits_tsf1],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.enc_inputs: batch['enc_inputs'],
                       model.dec_inputs: batch['dec_inputs'],
                       model.labels: batch['labels']
                       })

        ori0 = np.argmax(logits_ori0, axis=2).tolist()
        ori0 = [[self.vocab.id2word[i] for i in sent] for sent in ori0]
        ori0 = strip_eos(ori0)

        tsf0 = np.argmax(logits_tsf0, axis=2).tolist()
        tsf0 = [[self.vocab.id2word[i] for i in sent] for sent in tsf0]
        tsf0 = strip_eos(tsf0)

        ori1 = np.argmax(logits_ori1, axis=2).tolist()
        ori1 = [[self.vocab.id2word[i] for i in sent] for sent in ori1]
        ori1 = strip_eos(ori1)

        tsf1 = np.argmax(logits_tsf1, axis=2).tolist()
        tsf1 = [[self.vocab.id2word[i] for i in sent] for sent in tsf1]
        tsf1 = strip_eos(tsf1)

        return ori0, tsf0, ori1, tsf1
