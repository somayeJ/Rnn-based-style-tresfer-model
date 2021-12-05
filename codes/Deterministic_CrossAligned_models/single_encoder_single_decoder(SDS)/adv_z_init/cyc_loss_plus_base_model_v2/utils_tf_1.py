from __future__ import unicode_literals
import os
import spacy
from spacy.attrs import ORTH, LIKE_URL
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler, scale , MinMaxScaler
#from Elmo_keras_write import create_elmo_embeddings,read_elmo_file, write_elmo_file
#from allennlp.commands.elmo import ElmoEmbedder
def write_elmo_file(directory_write, file):
    np.savetxt(directory_write, file)
    return

def read_elmo_file( directory, file):
    x = np.loadtxt(directory +file)
    return x.tolist()
def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]


def feed_dictionary(model, batch, rho, cycl_factor,rec_factor,gamma, dropout=1, learning_rate=None):

    if 'token_features_enc' in batch.keys():
        feed_dict = {model.dropout: dropout,
                     model.learning_rate: learning_rate,
                     model.rho: rho,
                     model.cycl_factor: cycl_factor,
                     model.rec_factor: rec_factor,
                     model.gamma: gamma,
                     model.batch_len: batch['len'],
                     model.batch_size: batch['size'],
                     model.enc_inputs: batch['enc_inputs'],
                     model.dec_inputs: batch['dec_inputs'],
                     model.targets: batch['targets'],
                     model.weights: batch['weights'],
                     model.labels: batch['labels'],
                     model.tokens_features: batch['token_features_enc'],
                     model.tokens_features_dec: batch['token_features_dec'],
                     model.tokens_features_target: batch['token_features_target']
                     }
    elif 'elmo_embeddings' in batch.keys():
        feed_dict = {model.dropout: dropout,
                     model.learning_rate: learning_rate,
                     model.rho: rho,
                     model.cycl_factor: cycl_factor,
                     model.rec_factor: rec_factor,
                     model.gamma: gamma,
                     model.batch_len: batch['len'],
                     model.batch_size: batch['size'],
                     model.enc_inputs: batch['enc_inputs'],
                     model.dec_inputs: batch['dec_inputs'],
                     model.targets: batch['targets'],
                     model.weights: batch['weights'],
                     model.labels: batch['labels'],
                     model.elmo_emb: batch['elmo_embeddings']}

    else :

        feed_dict = {model.dropout: dropout,
                     model.learning_rate: learning_rate,
                     model.rho: rho,
                     model.cycl_factor: cycl_factor,
                     model.rec_factor: rec_factor,
                     model.gamma: gamma,
                     model.batch_len: batch['len'],
                     model.batch_size: batch['size'],
                     model.enc_inputs: batch['enc_inputs'],
                     model.dec_inputs: batch['dec_inputs'],
                     model.targets: batch['targets'],
                     model.weights: batch['weights'],
                     model.labels: batch['labels']}

    return feed_dict

# makeup(x0, len(x1))
def makeup(_x, n):
    x = []
    for i in range(n):
        # % : baghimande taghsim
        x.append(_x[i % len(_x)])
    return x

def reorder(order, _x):
    x = list(range(len(_x)))
    for i, a in zip(order, _x):
        x[i] = a
    return x

# noise model from paper "Unsupervised Machine Translation Using Monolingual Corpora Only"
def noise(x, unk, word_drop=0.0, k=3):
    n = len(x)
    for i in range(n):
        if random.random() < word_drop:
            x[i] = unk

    # slight shuffle such that |sigma[i]-i| <= k
    sigma = (np.arange(n) + (k+1) * np.random.rand(n)).argsort()
    return [x[sigma[i]] for i in range(n)]


def get_batch(x, y, word2id, args, noisy=False, min_len=5):
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    x_input, go_x, x_eos, weights , sents= [], [], [], [], []

    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    for sent in x:
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent)
        padding = [pad] * (max_len - l)
        _sent_id = noise(sent_id, unk) if noisy else sent_id
        if args.rev_tokens:
            x_input.append(padding + _sent_id[::-1])
        else:
            x_input.append(_sent_id + padding)
        go_x.append([go] + sent_id + padding)
        x_eos.append(sent_id + [eos] + padding)
        weights.append([1.0] * (l+1) + [0.0] * (max_len-l))
        sents.append(sent)

    return {'enc_inputs': x_input,
            'dec_inputs': go_x,
            'targets':    x_eos,
            'weights':    weights,
            'labels':     y,
            'size':       len(x),
            'len':        max_len+1,
            'text_inputs': sents}

def get_batches(x0, x1, word2id, batch_size, args ,noisy=False):
    if args.downsampling:
        if len(x0) < len(x1):
            x1 = x1[:len(x0)] 
        if len(x1) < len(x0):
            x0 = x0[len(x1)]
    else:
        if len(x0) < len(x1):
            x0 = makeup(x0, len(x1))
        if len(x1) < len(x0):
            x1 = makeup(x1, len(x0))
    n = len(x0)
    if args.keep_data_order:
        print('------------The data will not be sorted based on the sequence lengths and its order will be saved------------')
        order0, order1 =[], []
    else:
        # it reorders the list based on their length
        order0 = range(n)
        z = sorted(zip(order0, x0), key=lambda i: len(i[1]))
        order0, x0 = zip(*z)

        order1 = range(n)
        z = sorted(zip(order1, x1), key=lambda i: len(i[1]))
        order1, x1 = zip(*z)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)

        batches.append(get_batch(x0[s:t] + x1[s:t],
            [0]*(t-s) + [1]*(t-s), word2id, args, noisy))
        s = t
    # returns batches consisting of several batch: each batch consists of half sentences from the first corpus with label 0
    # & the other half from the other corpus, order0  && order1 show the indices of the sentences in the corpora 
    return batches, order0, order1
