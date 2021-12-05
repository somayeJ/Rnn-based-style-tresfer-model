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


def features_vectors(data0, data1, nlp):
    '''
    # me # https://spacy.io/api/token
    This method functions to produce list of tokens for inputs instead of method load-sentences in (data_tokens0 & data_tokens1),
    and also is used to produce list of features for each token
    :param data0: list of seqs from corpus0
    :param data1: list of seqs from corpus1
    :param nlp:
    :return:
    doc_feat_list0, doc_feat_list1, # list of token_features list for each sequence,
    data_tokens0, data_tokens1, # list of token_lists of the corresponding sequences in the corpus
    word_features_d_0, word_features_d_1, # all_tokens_extra_features dict
    '''
    data_tokens0 = []
    data_tokens1 = []
    word_features_d_0 = {'words': [], 'like_url': [], 'pos': [], 'is_alpha': [], 'is_stop': [], 'ent_type': [],
                       'dep_': [], 'dep': [], 'sent_id':[]}
    word_features_d_1 = {'words': [], 'like_url': [], 'pos': [], 'is_alpha': [], 'is_stop': [], 'ent_type': [],
                       'dep_': [], 'dep': [], 'sent_id':[]}
    if len(data0)>0:
        doc_list0 = [nlp(seq) for seq in data0]
        doc_feat_list0 = []
        doc_feat_list1 = []
        big_dep_dict = {}
        for (indx, doc0) in enumerate(doc_list0):
            # print(doc)
            seq_feat0 = []
            seq_tokens0 = []
            for token0 in doc0:
                seq_tokens0.append(token0.text)
                word_features_d_0['words'].append(token0.text)
                word_features_d_0['dep_'].append(token0.dep_)
                word_features_d_0['like_url'].append(token0.like_url)
                word_features_d_0['sent_id'].append(indx)
                if token0.dep < 1000:
                    word_features_d_0['dep'].append(token0.dep)
                elif str(token0.dep) in big_dep_dict.keys():
                    word_features_d_0['dep'].append(big_dep_dict[str(token0.dep)])
                else:
                    big_dep_dict[str(token0.dep)] = -1 * len(big_dep_dict)
                    word_features_d_0['dep'].append(big_dep_dict[str(token0.dep)])
                word_features_d_0['pos'].append(token0.pos)
                word_features_d_0['is_alpha'].append(token0.is_alpha)
                word_features_d_0['is_stop'].append(token0.is_stop)
                word_features_d_0['ent_type'].append(token0.ent_type)
                seq_feat0.append(
                    [token0.like_url, token0.pos, token0.dep, token0.is_alpha, token0.is_stop, token0.ent_type])

            doc_feat_list0.append(seq_feat0)
            data_tokens0.append(seq_tokens0)
        # print('words_d_0',len(word_features_d['words']),word_features_d['words'])
        # print('word_features_d',word_features_d)
    if len(data1)>0:
        doc_list1 = [nlp(seq) for seq in data1]
        for indx,doc1 in enumerate(doc_list1):
            # print(doc)
            seq_feat1 = []
            seq_tokens1 = []
            for token1 in doc1:
                seq_tokens1.append(token1.text)
                word_features_d_1['words'].append(token1.text)
                word_features_d_1['dep_'].append(token1.dep_)
                word_features_d_1['like_url'].append(token1.like_url)
                word_features_d_1['pos'].append(token1.pos)
                word_features_d_1['sent_id'].append(indx)
                if token1.dep < 1000:
                    word_features_d_1['dep'].append(token1.dep)
                    # print(token1.dep)
                    # print(str(token1.dep),big_dep_dict.keys(), big_dep_dict)
                elif str(token1.dep) in big_dep_dict.keys():
                    word_features_d_1['dep'].append(big_dep_dict[str(token1.dep)])

                else:
                    big_dep_dict[str(token1.dep)] = -1 * len(big_dep_dict)
                    word_features_d_1['dep'].append(big_dep_dict[str(token1.dep)])

                word_features_d_1['is_alpha'].append(token1.is_alpha)
                word_features_d_1['is_stop'].append(token1.is_stop)
                word_features_d_1['ent_type'].append(token1.ent_type)
                seq_feat1.append(
                    [token1.like_url, token1.pos, token1.dep, token1.is_alpha, token1.is_stop, token1.ent_type])
                # print(token1.ent_type)
            doc_feat_list1.append(seq_feat1)
            data_tokens1.append(seq_tokens1)
        #print(word_features_d_0)

        # print('words_d_0_1',len(word_features_d['words']), len(tokenss1), word_features_d['words'])

    return doc_feat_list0 , doc_feat_list1, data_tokens0, data_tokens1,word_features_d_0, word_features_d_1

def normalize_features(word_features_d, word2id):
    '''
    # me # https://scikit-learn.org/stable/modules/preprocessing.html, https://machinelearningmastery.com/normalize-standardize-time-series-data-python/
    :param word_features_df: all_tokens_extra_features dictionary
    word2id = vocab.word2id

    :return:
    1st output: a df with columns of['like_url', 'is_alpha', 'is_stop', 'ent_type', 'dep', 'pos','sent_id', 'sent_id', 'words', 'dep_']
    2st output:  a df which is a subset of the first df with numeric columns of ['like_url', 'is_alpha', 'is_stop', 'ent_type', 'dep', 'pos','sent_id']
    '''
    w2id = []
    for word in word_features_d['words']:
        # print(vocab.word2id['<unk>'])
        if word in word2id.keys():
            w2id.append(word2id[word])
        else:
            w2id.append(word2id['<unk>'])

    print('normalize_features',len(w2id), len(word_features_d['words']))
    word_features_d['w2id'] = w2id
    word_features_df = pd.DataFrame(data=word_features_d)
    #print('oooooooooooooooo', type(word_features_df))
    #print(word_features_df)
    # selecting the numerical columns
    X_train = word_features_df[['like_url', 'is_alpha', 'is_stop', 'ent_type', 'dep', 'pos']]

    # print(X_train)
    # normalization of the features:
    # scaler = StandardScaler().fit(X_train)
    scaler = MinMaxScaler(copy=True, feature_range=(-1, 1)).fit(X_train)
    #print(scaler)
    # print('feature',feature)
    # print(type(word_features_df[feature]))
    # print('scaler.mean_ ', scaler.mean_  )
    # print(word_features_df[feature])
    # print('scaler.scale_ ', scaler.scale_ )
    # print('data_transform', scaler.transform(X_train))
    normalized_df = scaler.transform(X_train)
    #print('type(normalized_df)', type(normalized_df))
    normalized_df_2 = pd.DataFrame({'like_url':normalized_df[:, 0], 'is_alpha': normalized_df[:, 1], 'is_stop': normalized_df[:, 2], 'ent_type': normalized_df[:, 3], 'dep': normalized_df[:, 4], 'pos': normalized_df[:, 5]})
    normalized_df_2['sent_id'] = np.array(word_features_d['sent_id'])
    normalized_df_2['words'] = np.array(word_features_d['words'])
    normalized_df_2['dep_'] = np.array(word_features_d['dep_'])
    # w2id_np = np.asarray(word_features_d['w2id'])
    # normalized_df_wid = np.concatenate( (normalized_df,w2id_np ), axis=1))
    #print(normalized_df_2)
    return normalized_df_2, normalized_df_2[['like_url', 'is_alpha', 'is_stop', 'ent_type', 'dep', 'pos','sent_id']]
def get_batch(x, y, word2id, noisy=False, min_len=5):
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    rev_x, go_x, x_eos, weights , sents= [], [], [], [], []
    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    for sent in x:
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent)
        padding = [pad] * (max_len - l)
        _sent_id = noise(sent_id, unk) if noisy else sent_id
        rev_x.append(padding + _sent_id[::-1])
        go_x.append([go] + sent_id + padding)
        x_eos.append(sent_id + [eos] + padding)
        weights.append([1.0] * (l+1) + [0.0] * (max_len-l))
        sents.append(sent)

    return {'enc_inputs': rev_x,
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
            [0]*(t-s) + [1]*(t-s), word2id, noisy))
        s = t
    # returns batches consisting of several batch: each batch consists of half sentences from the first corpus with label 0
    # & the other half from the other corpus, order0  && order1 show the indices of the sentences in the corpora 
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

def get_batch_modified(x, y,order0, order1,s, add_features, n_0, n_1, normalized_feat_numeric_0, normalized_feat_numeric_1 ,word2id,noisy=False, min_len=5):
    print(type(word2id))
    pad = word2id.get('<pad>')
    go = word2id.get('<go>')
    eos = word2id.get('<eos>')
    unk = word2id.get('<unk>')

    rev_x, go_x, x_eos, weights, token_features_rev,token_features_dec,token_features_target = [], [], [], [], [], [],[]
    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    # extracting the features of the sequences belonging to the first corpus
    # label haro 3 shash ro dar nazar nemigirim chon, AE hast model
    for idx, sent in enumerate(x[:len(x)/2]):
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent)
        padding = [pad] * (max_len - l)
        _sent_id = noise(sent_id, unk) if noisy else sent_id
        # sent_id[::-1]: is the reverse of seq, sent_id
        rev_x.append(padding + _sent_id[::-1])
        go_x.append([go] + sent_id + padding)
        x_eos.append(sent_id + [eos] + padding)
        weights.append([1.0] * (l + 1) + [0.0] * (max_len - l))
        if add_features:
            # token_features_of_seq = []
            # sotoon akhar, shomare jomleha ast, bara hamin ignoresh mikonim
            no_feats =  normalized_feat_numeric_0.iloc[1, :-1].shape[0]
            padding_feat = [[-2] * no_feats for pad_token in padding]
            sent_no_0 = order0[s + idx]
            token_features_of_seq = normalized_feat_numeric_0.loc[normalized_feat_numeric_0.sent_id == sent_no_0, :].iloc[:, :-1].values.tolist()
            print('sent_normalized_feat',n_0.loc[n_0.sent_id ==sent_no_0,'words'])
            print('sent', sent)
            token_features_rev.append(padding_feat + token_features_of_seq[::-1])
            # we add -2 for go token, && for eos token
            token_features_dec.append([[-2] * no_feats]+ token_features_of_seq + padding_feat)
            token_features_target.append(token_features_of_seq +[[-2] * no_feats]+ padding_feat)

    # extracting the features of the sequences belonging to the second corpus
    for idx, sent in enumerate(x[len(x)/2:]):
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent)
        padding = [pad] * (max_len - l)
        _sent_id = noise(sent_id, unk) if noisy else sent_id
        # sent_id[::-1]: is the reverse of seq, sent_id
        rev_x.append(padding + _sent_id[::-1])
        go_x.append([go] + sent_id + padding)
        x_eos.append(sent_id + [eos] + padding)
        weights.append([1.0] * (l+1) + [0.0] * (max_len-l))
        if add_features:
            # token_features_of_seq = []
            no_feats = normalized_feat_numeric_1.iloc[1, :-1].shape[0]
            padding_feat = [[-2] * no_feats for pad_token in padding]
            sent_no_1 = order1[s + idx]
            #seq_no = word_features_d_0['sent_id'].index(sent_no_0)
            #token_features_seq = doc_feat_list0[seq_no]
            token_features_of_seq = normalized_feat_numeric_1.loc[normalized_feat_numeric_1.sent_id == sent_no_1, :].iloc[:, :-1].values.tolist()
            token_features_rev.append(padding_feat + token_features_of_seq[::-1])
            # we add -2 for go token, && for eos token
            token_features_dec.append([[-2] * no_feats] + token_features_of_seq + padding_feat)
            token_features_target.append(token_features_of_seq +[[-2] * no_feats]+ padding_feat)



    return {'enc_inputs': rev_x,
            'dec_inputs': go_x,
            'targets':    x_eos,
            'weights':    weights,
            'labels':     y,
            'size':       len(x),
            'len':        max_len+1,
            'token_features_enc': token_features_rev,
            'token_features_dec':token_features_dec,
            'token_features_target':token_features_target }


def get_batches_modified(x0_tokens, x1_tokens, x0_full, x1_full,word2id, batch_size, add_features, noisy=False):
    '''
    # get_batches: returns batches consisting of dictionaries for each batch: each dict len is twice as batch size & consists of
    # a batch_size of seqs from the label0 corpus and a batch_size from the other corpus, order0  && order1 show the indices of the sentences in the corpora,
    # & ( sentences are padded to the max size of the input batch words, paddings are at the begining of the seq, and words are
     replaced with their ids and also are reversed ))

    :param x0: A list of seq_tokens list (for each input seq) for corpus x0
    :param x1: A list of seq_tokens list (for each input seq) for
    corpusx1
    :param word2id: vocab.word2I=id
    :param batch_size:
    :param noisy:
    :return:
    #batches:  a list consisting of several dictionaries for each batch: each dict len is twice as batch size & consists of
    #            {'enc_inputs': rev_x,
    #             'dec_inputs': go_x,
    #             'targets':    x_eos,
    #             'weights':    weights,
    #             'labels':     y,
    #             'size':       len(x),
    #             'len':        max_len+1},
    # a batch_size of seqs from the label0 corpus and a batch_size from the other corpus, order0 && order1 show the indices of the sentences in the corpora,
    #( sentences are padded (words replaced with their ids))
    '''

    if add_features:
        nlp = spacy.load('en_core_web_lg')
        if len(x0_full)<len(x1_full):
            x0_full = makeup(x0_full, len(x1_full))
        if len(x1_full)<len(x0_full):
            x1_full = makeup(x1_full, len(x0_full))
        doc_feat_list0, doc_feat_list1, x0, x1, word_features_d_0, word_features_d_1 = features_vectors(x0_full, x1_full, nlp)
        n_complete_0, word_features_d_0_norm = normalize_features(word_features_d_0, word2id)
        n_complete_1, word_features_d_1_norm = normalize_features(word_features_d_1, word2id)
    else:
        n_complete_0 =[]
        n_complete_1 = []
        word_features_d_0_norm = []
        word_features_d_1_norm = []
        x0 , x1 = [], []
        x0.extend(x0_tokens)
        x1.extend(x1_tokens)

        # size 2 corpora ro barabar e ham mikonim
        if len(x0) < len(x1):
            x0 = makeup(x0, len(x1))
        if len(x1) < len(x0):
            x1 = makeup(x1, len(x0))
    n = len(x0)
    # it reorders the list based on their length
    order0 = range(n)
    z = sorted(zip(order0, x0), key=lambda i: len(i[1]))
    # order0 = The tuple of sentence ids
    # x0 = tuple of lists of tokens for each sentence
    order0, x0 = zip(*z)

    order1 = range(n)
    z = sorted(zip(order1, x1), key=lambda i: len(i[1]))
    order1, x1 = zip(*z)


    # it reorders the list based on their length
    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batch = get_batch( x0[s:t] + x1[s:t],
            [0]*(t-s) + [1]*(t-s), order0, order1,  s,  add_features, n_complete_0, n_complete_1, word_features_d_0_norm,word_features_d_1_norm, word2id, noisy)
        batches.append(batch)
        #batches.append(get_batch( x0[s:t] + x1[s:t],
            #[0]*(t-s) + [1]*(t-s), order0, order1, s,  add_features, word_features_d_0_norm,word_features_d_1_norm, word2id, noisy))
        s = t
        x= x0[s:t] + x1[s:t]
        for k,xs in zip(batch['enc_inputs'],x) :
            print(k, xs)
    return batches, order0, order1


def get_batches_emb(x0, word2id, batch_size, noisy=False):
    '''

    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
    '''

    n = len(x0)
    # it reorders the list based on their length
    order0 = range(n)
    z = sorted(zip(order0, x0), key=lambda i: len(i[1]))
    order0, x0 = zip(*z)
    order1 = range(n)
    '''
    z = sorted(zip(order1, x1), key=lambda i: len(i[1]))
    order1, x1 = zip(*z)
    '''
    # it reorders the list based on their length
    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(x0[s:t] ,
            [0]*(t-s), word2id, noisy))
        s = t
    # returns batches consisting of several batch: each batch consisits of half sentences from the first corpus with label 0
    # & the other half from the other corpus, order0  && order1 show the indices of the sentences in the corpora
    return batches, order0, order1


