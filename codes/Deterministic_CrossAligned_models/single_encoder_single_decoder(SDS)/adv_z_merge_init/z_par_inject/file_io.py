from nltk import word_tokenize, sent_tokenize
import codecs
import pandas as pd
from sklearn.preprocessing import StandardScaler, scale , MinMaxScaler
import numpy as np
from vocab import Vocabulary, build_vocab

def load_doc(path):
    data = []
    with open(path) as f:
        for line in f:
            sents = sent_tokenize(line)
            doc = [word_tokenize(sent) for sent in sents]
            data.append(doc)
    return data

def load_sent(path, max_size=-1):
    '''

    :param path:
    :param max_size:
    :return: # a list of lists of sentence-tokens for each input sequence
    '''

    data = []
    with open(path) as f:
        for line in f:
            if len(data) == max_size:
                break
            data.append(line.split())
    #print(data[:4])
    return data

def load_sent_full(path, max_size=-1):
    '''
    # me
    :param path:
    :param max_size:
    :return:
    '''
    data = []
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            if len(data) == max_size:
                break
            data.append(line.strip())
    return data


def load_vec(path):
    x = []
    with open(path) as f:
        for line in f:
            p = line.split()
            p = [float(v) for v in p]
            x.append(p)
    return x

def write_doc(docs, sents, path):
    with open(path, 'w') as f:
        index = 0
        for doc in docs:
            for i in range(len(doc)):
                f.write(' '.join(sents[index]))
                f.write('\n' if i == len(doc)-1 else ' ')
                index += 1

def write_sent(sents, path):
    with open(path, 'w') as f:
        for sent in sents:
            f.write(' '.join(sent) + '\n')

def write_vec(vecs, path):
    with open(path, 'w') as f:
        for vec in vecs:
            for i, x in enumerate(vec):
                f.write('%.3f' % x)
                f.write('\n' if i == len(vec)-1 else ' ')

# these files are in utils.py

'''

def features_vectors(data0, data1,  nlp):


    data_tokens0 = []
    data_tokens1 = []
    word_features_d ={'words':[], 'like_url':[], 'pos':[], 'is_alpha':[] , 'is_stop':[], 'ent_type':[], 'dep_':[],'dep':[]}

    doc_list0 = [nlp(seq) for seq in data0]
    doc_list1 = [nlp(seq) for seq in data1]
    doc_feat_list0 = []
    doc_feat_list1 = []
    big_dep_dict = {}
    for doc0 in doc_list0 :
        #print(doc)
        seq_feat0 = []
        seq_tokens0 =[]
        for token0 in doc0:
            seq_tokens0.append(token0.text)
            word_features_d['words'].append(token0.text)
            word_features_d['dep_'].append(token0.dep_)
            word_features_d['like_url'].append(token0.like_url)
            if token0.dep<1000:
                word_features_d['dep'].append(token0.dep)
            elif str(token0.dep)   in big_dep_dict.keys():
                    word_features_d['dep'].append(big_dep_dict[str(token0.dep)])

            else:
                    big_dep_dict[str(token0.dep)] = -1 * len(big_dep_dict)
                    word_features_d['dep'].append(big_dep_dict[str(token0.dep)])

            word_features_d['pos'].append(token0.pos)
            word_features_d['is_alpha'].append(token0.is_alpha)
            word_features_d['is_stop'].append(token0.is_stop)
            word_features_d['ent_type'].append(token0.ent_type)
            seq_feat0.append([token0.like_url, token0.pos, token0.dep, token0.is_alpha, token0.is_stop, token0.ent_type])

        doc_feat_list0.append(seq_feat0)
        data_tokens0.append(seq_tokens0)
    #print('words_d_0',len(word_features_d['words']),word_features_d['words'])
    #print('word_features_d',word_features_d)
    for doc1 in doc_list1 :
        #print(doc)
        seq_feat1 = []
        seq_tokens1 = []
        for token1 in doc1:
            seq_tokens1.append(token1.text)
            word_features_d['words'].append(token1.text)
            word_features_d['dep_'].append(token1.dep_)
            word_features_d['like_url'].append(token1.like_url)
            word_features_d['pos'].append(token1.pos)
            if token1.dep < 1000:
                word_features_d['dep'].append(token1.dep)
                #print(token1.dep)
                #print(str(token1.dep),big_dep_dict.keys(), big_dep_dict)
            elif str(token1.dep) in big_dep_dict.keys():
                    word_features_d['dep'].append(big_dep_dict[str(token1.dep)])

            else:
                    big_dep_dict[str(token1.dep)] = -1 * len(big_dep_dict)

                    word_features_d['dep'].append(big_dep_dict[str(token1.dep)])

            word_features_d['is_alpha'].append(token1.is_alpha)
            word_features_d['is_stop'].append(token1.is_stop)
            word_features_d['ent_type'].append(token1.ent_type)
            seq_feat1.append([ token1.like_url, token1.pos, token1.dep, token1.is_alpha, token1.is_stop, token1.ent_type])
            #print(token1.ent_type)
        doc_feat_list1.append(seq_feat1)
        data_tokens1.append(seq_tokens1)
    print(word_features_d)

    #print('words_d_0_1',len(word_features_d['words']), len(tokenss1), word_features_d['words'])
    return  doc_feat_list0, data_tokens0, doc_feat_list1, data_tokens1, word_features_d # all_tokens_extra_features


def normalize_features(word_features_d, vocab):


    w2id = []
    for word in word_features_d['words']:
        # print(vocab.word2id['<unk>'])
        if word in vocab.word2id.keys():
            w2id.append(vocab.word2id[word])
        else:
            w2id.append(vocab.word2id['<unk>'])
    print(len(w2id), len(word_features_d['words']))
    word_features_d['w2id'] = w2id
    word_features_df = pd.DataFrame(data=word_features_d)
    print(word_features_df)
    # selecting the numerical columns
    X_train = word_features_df[['like_url', 'is_alpha', 'is_stop', 'ent_type', 'dep', 'pos']]
    #print(X_train)
    # normalization of the features:
    # scaler = StandardScaler().fit(X_train)
    scaler = MinMaxScaler(copy=True,feature_range=(-1, 1)).fit(X_train)
    print(scaler)
    # print('feature',feature)
    # print(type(word_features_df[feature]))
    # print('scaler.mean_ ', scaler.mean_  )
    # print(word_features_df[feature])
    # print('scaler.scale_ ', scaler.scale_ )
    #print('data_transform', scaler.transform(X_train))
    normalized_df= scaler.transform(X_train)
    w2id_np = np.asarray(word_features_d['w2id'])
    #normalized_df_wid = np.concatenate( (normalized_df,w2id_np ), axis=1))
    print(normalized_df)
    return scaler.transform(X_train)

'''