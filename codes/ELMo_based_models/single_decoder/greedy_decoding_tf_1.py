import numpy as np
from utils_tf_1 import strip_eos

class Decoder(object):

    def __init__(self, sess, args, vocab, model):
        self.sess, self.vocab, self.model = sess, vocab, model

    def rewrite(self, batch):
        half = batch['size'] / 2
        print("len(batch['enc_inputs']) , len( batch['labels']",len(batch['dec_inputs']) , len( batch['labels']))
        model = self.model
        logits_ori, logits_tsf,  = self.sess.run(
            [model.hard_logits_ori, model.hard_logits_tsf],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.dec_inputs: batch['dec_inputs'],
                       model.labels: batch['labels'],
                       model.elmo_emb: batch['elmo_embeddings']
                       })

        ori = np.argmax(logits_ori, axis=2).tolist()
        ori = [[self.vocab.id2word[i] for i in sent] for sent in ori]
        ori = strip_eos(ori)

        tsf = np.argmax(logits_tsf, axis=2).tolist()
        tsf = [[self.vocab.id2word[i] for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)


        return ori, tsf