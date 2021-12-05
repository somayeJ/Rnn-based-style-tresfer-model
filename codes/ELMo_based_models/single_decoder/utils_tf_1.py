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


def feed_dictionary(model, batch, rho, gamma, dropout=1, learning_rate=None):

    if 'token_features_enc' in batch.keys():
        feed_dict = {model.dropout: dropout,
                     model.learning_rate: learning_rate,
                     model.rho: rho,
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
                     model.gamma: gamma,
                     model.batch_len: batch['len'],
                     model.batch_size: batch['size'],
                     #model.enc_inputs: batch['enc_inputs'],
                     model.dec_inputs: batch['dec_inputs'],
                     model.targets: batch['targets'],
                     model.weights: batch['weights'],
                     model.labels: batch['labels'],
                     model.elmo_emb: batch['elmo_embeddings']}

    else :

        feed_dict = {model.dropout: dropout,
                     model.learning_rate: learning_rate,
                     model.rho: rho,
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





def get_batch(x, y, word2id, noisy=False, min_len=5):
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    rev_x, go_x, x_eos, weights = [], [], [], []
    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    for sent in x:
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent)
        padding = [pad] * (max_len - l)
        _sent_id = noise(sent_id, unk) if noisy else sent_id
        # _sent_id[::-1]): reverse of the_sent_id
        rev_x.append(padding + _sent_id[::-1])
        go_x.append([go] + sent_id + padding)
        x_eos.append(sent_id + [eos] + padding)
        weights.append([1.0] * (l+1) + [0.0] * (max_len-l))

    return {'enc_inputs': rev_x,
            'dec_inputs': go_x,
            'targets':    x_eos,
            'weights':    weights,
            'labels':     y,
            'size':       len(x),
            'len':        max_len+1}

def get_batches(x0, x1, word2id, batch_size, noisy=False):
    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
    n = len(x0)

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
            [0]*(t-s) + [1]*(t-s), word2id, noisy))
        s = t

    return batches, order0, order1

def get_batch_elmo(x, y, word2id, elmo_rep ,noisy=False, min_len=5):
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']
    rev_x, go_x, x_eos, weights = [], [], [], []
    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    documents =[]
    elmo=1
    if len(elmo_rep) == 0:
        elmo_embeddings = elmo_rep
        print('='*5)
        print("len_elmo_embeddings=0")
        print('='*5)
        # in ghesmat payin ro hazf kardam ta betoonam ba tensorflow paeen run konam
        '''
        for i, sent in enumerate(x):
            print(1111111, "elmooooooooooooooooooo")
            documents.append({'tokens':sent, 'label':str(y[i])})
            sent_id = [word2id[w] if w in word2id else unk for w in sent]
            l = len(sent)
            padding = [pad] * (max_len - l)
            _sent_id = noise(sent_id, unk) if noisy else sent_id
            rev_x.append(padding + _sent_id[::-1])
            go_x.append([go] + sent_id + padding)
            x_eos.append(sent_id + [eos] + padding)
            weights.append([1.0] * (l+1) + [0.0] * (max_len-l))
        elmo = ElmoEmbedder()
        elmo_embeddings, labels= create_elmo_embeddings(elmo, documents, max_sentences=0)
        # shape(elmo_embeddings) : ba_size*2 * 3072
        '''
    else:
        elmo_embeddings = elmo_rep
        for sent in x:
            print(sent)
            sent_id = [word2id[w] if w in word2id else unk for w in sent]
            l = len(sent)
            padding = [pad] * (max_len - l)
            _sent_id = noise(sent_id, unk) if noisy else sent_id
            # _sent_id[::-1]): reverse of the_sent_id
            rev_x.append(padding + _sent_id[::-1])
            go_x.append([go] + sent_id + padding)
            x_eos.append(sent_id + [eos] + padding)
            weights.append([1.0] * (l + 1) + [0.0] * (max_len - l))
    return {'enc_inputs': rev_x,
            'dec_inputs': go_x,
            'targets':    x_eos,
            'weights':    weights,
            'labels':     y,
            'size':       len(x),
            'len':        max_len+1,
            'elmo_embeddings':elmo_embeddings}


def get_batches_elmo(x0, x1, word2id, batch_size,mode, elmo_rep_directory, noisy=False):
    """
    if we want to add embeddings of the seq based on Elmo model
    :param x0:
    :param x1:
    :param word2id:
    :param batch_size:
    :param noisy:
    :return:
    """
    if os.path.isfile(elmo_rep_directory+mode+"elmo.0"):
        print("------------Reading elmo embeddings for", mode, "data from this directory", elmo_rep_directory, "--------------")
        elmo_emb_0 = read_elmo_file(elmo_rep_directory, mode+"elmo.0")
        elmo_emb_1 = read_elmo_file(elmo_rep_directory, mode + "elmo.1")
    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
        if os.path.isfile(elmo_rep_directory+mode+"elmo.0"):
            elmo_emb_0 = makeup(elmo_emb_0, len(x1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
        if os.path.isfile(elmo_rep_directory+mode+"elmo.0"):
            elmo_emb_1 = makeup(elmo_emb_1, len(x0))

    n = len(x0)
    """
    x0 =[["hi"], ["hi", "somaye", "and", "goodbye"],["i","am"]]
    order0 = range(len(x0))
    z = sorted(zip(order0, x0), key=lambda i: len(i[1]))
    z = [(0, ['hi']), (2, ['i', 'am']), (1, ['hi', 'somaye', 'and', 'goodbye'])]  
    order0, x0 = zip(*z)
    x0 = (['hi'], ['i', 'am'], ['hi', 'somaye', 'and', 'goodbye'])
    order0 = (0, 2, 1)
    """
    order0 = range(n)
    z = sorted(zip(order0, x0), key=lambda i: len(i[1]))
    order0, x0 = zip(*z)
    if os.path.isfile(elmo_rep_directory+mode+"elmo.0"):
        elmo_rep_0 = [elmo_emb_0[i] for i in order0]

    order1 = range(n)
    z = sorted(zip(order1, x1), key=lambda i: len(i[1]))
    order1, x1 = zip(*z)
    if os.path.isfile(elmo_rep_directory+mode+"elmo.0"):
        elmo_rep_1 = [elmo_emb_1[i] for i in order1]

    batches = []
    s = 0

    while s < n:
        t = min(s + batch_size, n)
        if os.path.isfile(elmo_rep_directory+mode+"elmo.0"):
            elmo_seq =  elmo_rep_0[s:t] + elmo_rep_1[s:t]
        else:
            elmo_seq = []
        batches.append(get_batch_elmo(x0[s:t] + x1[s:t],
            [0]*(t-s) + [1]*(t-s), word2id, elmo_seq, noisy))
        s = t


    return batches, order0, order1