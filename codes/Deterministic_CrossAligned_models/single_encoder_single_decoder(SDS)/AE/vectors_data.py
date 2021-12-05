# Somaye
# Defining vectors for each input token, in which syntactic features are infused
# https://spacy.io/models/en
# models: python -m spacy download en_core_web_sm, python -m spacy download en_core_web_md, python -m spacy download en_core_web_lg

# python vectors_data.py --train ../data/yelp/sentiment.train --dev ../data/yelp/sentiment.dev --output ../tmp/features_vectors/sentiment.dev   --vocab ../tmp/features_vectors/yelp.vocab
from __future__ import unicode_literals
import spacy
from spacy.attrs import ORTH, LIKE_URL
from file_io import load_sent, write_sent
from options import load_arguments
import sys
import os
from vocab import Vocabulary, build_vocab
from accumulator import Accumulator
from options import load_arguments
from file_io import load_sent, write_sent,load_sent_full # features_vectors, normalize_features
from utils import *

from nn import *
import beam_search, greedy_decoding
from datetime import datetime
import codecs
import pandas as pd
nlp = spacy.load('en_core_web_lg')
'''
nlp = spacy.load('en_core_web_sm')
#nlp = spacy.load('en_core_web_md')
#nlp = spacy.load('en_core_web_lg')
doc = nlp(u"When Sebastian Thrun started working on self-driving cars at Google "
          u"in 2007, few people outside of the company took him seriously.")

dep_labels = []
for token in doc:
    while token.head != token:
        dep_labels.append(token.dep_)
        token = token.head
print(dep_labels)


doc = nlp(u"Check out https://spacy.io")
for token in doc:
    print(token.text, token.orth, token.like_url)

attr_ids = [ORTH, LIKE_URL]
doc_array = doc.to_array(attr_ids)
print(doc_array.shape)
print(len(doc), len(attr_ids))

assert doc[0].orth == doc_array[0, 0]
assert doc[1].orth == doc_array[1, 0]
assert doc[0].like_url == doc_array[0, 1]

assert list(doc_array[:, 1]) == [t.like_url for t in doc]
print(list(doc_array[:, 1]))
'''
if __name__ == '__main__':

    args = load_arguments()
    #####   data preparation   #####
    if args.train:
        # full sequences
        train0_full = load_sent_full(args.train + '.0', args.max_train_size)
        train1_full = load_sent_full(args.train + '.1', args.max_train_size)
        train0 = load_sent_full(args.train + '.0', args.max_train_size)
        train1 = load_sent_full(args.train + '.1', args.max_train_size)
        # tokenized sequences
        doc_feat_list0 , doc_feat_list1,  data_tokens0, data_tokens1,  word_features_d_0, word_features_d_1  = features_vectors(train0_full[:4],train1_full[:4], nlp)

        #train0 = load_sent(args.train + '.0', args.max_train_size)
        #train1 = load_sent(args.train + '.1', args.max_train_size)
        #print('doc_feat_list0', doc_feat_list0, '00000000000')
        #print('doc_feat_list1', doc_feat_list1, '1111111111111')
        print '#sents of training file 0:', len(train0_full)
        print '#sents of training file 1:', len(train1_full)
        #print(train0,'000000000000000000000')
        #print(all_tokens_extra_features)
        #print(normalized_feat)
        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab)
    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print 'vocabulary size:', vocab.size
    _, normalized_feat_numeric = normalize_features(word_features_d_1, vocab.word2id)
    print(normalized_feat_numeric)
    print(normalized_feat_numeric.loc[normalized_feat_numeric.sent_id == 0, :])
    # tabdil be np
    print(normalized_feat_numeric.loc[normalized_feat_numeric.sent_id == 0, :].iloc[:, :-1].values)
    # tabdil be python list
    z=normalized_feat_numeric.loc[normalized_feat_numeric.sent_id == 0, :].iloc[:, :-1].values.tolist()
    print(z)
    print(normalized_feat_numeric.iloc[1, :-1].shape[0])

