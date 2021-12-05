
# me: modofied Elmo_keras.py for writing the elmo embeddings of files of yelp data in directory_write
# Insert the directory from which the data is loaded
import keras
import os
import sys
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import random
import torch
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Activation, Dropout
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
directory = "../data/yelp/sentiment" # yelp_dataset
directory_write = "../data/yelp_elmo_rep/sentiment"

def read_data(directory,mode,label):
    files_dir = directory+mode+label
    with open(files_dir, 'r') as f1:
        reviews_raw=f1.readlines()
        print(directory,mode,label, len(reviews_raw))
    data = []
    for review in reviews_raw:
        data.append({"text": review.replace("<br />", " ").replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ').strip(), "label": label})
    return data


#train_data, test_data = download_and_load_datasets()
def load_dataset(directory, mode, label):
    data = read_data(directory,mode,label)

    return data


def load_all_dataset(directory,  label):
    """
    :param directory: the directory from which the data is loaded
    :return: data1, data2
    data1: list of dictionaries corresponding to each review consisting of the three components as follows:
    {'tokens': 'text', label: '1 or 0' , 'tokens':[...]}
    """
    train_data = load_dataset(directory,".train.", label)
    dev_data = load_dataset(directory,".dev.",label)
    test_data = load_dataset(directory, ".test.",label)
    return  train_data, dev_data,test_data



#Tokenize text. Note, it would be better to first split it into sentences.
def tokenize_text(documents, max_tokens):
    for document in documents:
        document['tokens'] = keras.preprocessing.text.text_to_word_sequence(document['text'], lower=False)
        if len(document['tokens']) == 0:
            document['tokens'].append('the')
        document['tokens'] = document['tokens'][0:max_tokens]




def get_sent_emb(line):
    '''
    creating the sentence embedding: [mean, min, max]
    :param line:
    :param word_dict:
    :return:
    '''
    # inserting 0 in np.mean(res, 0) instead of np.mean(res) returns back an array to us
    mm = np.mean(line, 0)
    mi = np.min(line, 0)
    ma = np.max(line, 0)
    emb = np.concatenate((mm, mi, ma))
    #emb = mm
    return emb

def create_elmo_embeddings(elmo, documents, max_sentences = 0):
    num_sentences = min(max_sentences, len(documents)) if max_sentences > 0 else len(documents)
    print("\n\n:: Lookup of "+str(num_sentences)+" ELMo representations. This takes a while ::")
    embeddings = []
    labels = []
    # tokens : a list of list of tokens
    tokens = [document['tokens']for document in documents]
    for i,document in enumerate(documents):
        if len(document['tokens'])==0:
            print("!!!!!!!!!!!!!!!!!!")
            print(i, document['label'])
            print('text', document['text'])
    documentIdx = 0
    for elmo_embedding,tokens_list in zip(elmo.embed_sentences(tokens),tokens):
        document = documents[documentIdx]
        # Average the 3 layers returned from ELMo, baray e har token
        #print(tokens())
        avg_elmo_embedding = np.average(elmo_embedding, axis=0)

        '''
        print ('\n', len(tokens_list),len(elmo_embedding), len(elmo_embedding[0]), len(elmo_embedding[0][0]), 44444444444444)
        15 3 15 1024 44444444444444
        elmo_embedding.shape = 3*no_tokens*1024
        
        avg_elmo_embedding[0][0] : float 
        print ('\n',len(avg_elmo_embedding),len(avg_elmo_embedding[0]),222222222222222222222)
        15(it changes,no_tokrns) 1024
        avg_elmo_embedding.shape = no_tokens*1024
        '''
        emb = get_sent_emb(avg_elmo_embedding)
        #embeddings.append(avg_elmo_embedding)
        embeddings.append(emb)

        labels.append(document['label'])

        # Some progress info
        documentIdx += 1
        percent = 100.0 * documentIdx / num_sentences
        line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
        status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} sentences'
        sys.stdout.write(status.format(percent, line, documentIdx, num_sentences))
        if max_sentences > 0 and documentIdx >= max_sentences:
            break

    '''
    embeddings[0][0][0]: float 
    print('\n',len(embeddings), len(embeddings[0]),len(embeddings[0][0]) , 888888888888888888)
    128 3072 
    embeddings.shape = 128 * 3072 
    '''
    return embeddings, labels

def write_elmo_file(directory_write, file):
    np.savetxt(directory_write, file)
    return

def read_elmo_file( directory, file):
    x = np.loadtxt(directory +file)
    return x.tolist()

train_data_0, dev_data_0, test_data_0 = load_all_dataset(directory,"0")
train_data_1, dev_data_1, test_data_1 = load_all_dataset(directory,"1")

max_tokens = 100

tokenize_text(train_data_0, max_tokens)
tokenize_text(dev_data_0, max_tokens)
tokenize_text(test_data_0, max_tokens)
tokenize_text(train_data_1, max_tokens)
tokenize_text(dev_data_1, max_tokens)

tokenize_text(test_data_1, max_tokens)

#elmo = ElmoEmbedder() #Set cuda_device to the ID of your GPU if you have one
#elmo = ElmoEmbedder(cuda_device=0) #Set cuda_device to the ID of your GPU if you have one
elmo = ElmoEmbedder(cuda_device=torch.cuda.current_device()) #Set cuda_device to the ID of your GPU if you have one
# https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu, based on which the changes are made for running on gpu

train_x_0, train_y_0 = create_elmo_embeddings(elmo, train_data_0, 0)
test_x_0, test_y_0  = create_elmo_embeddings(elmo, test_data_0, 0)
dev_x_0, dev_y_0 = create_elmo_embeddings(elmo, dev_data_0, 0)

train_x_1, train_y_1 = create_elmo_embeddings(elmo, train_data_1, 0)
test_x_1, test_y_1 = create_elmo_embeddings(elmo, test_data_1, 0)
dev_x_1, dev_y_1 = create_elmo_embeddings(elmo, dev_data_1, 0)

write_elmo_file(directory_write+".train.elmo.0", train_x_0)
write_elmo_file(directory_write+".test.elmo.0", test_x_0)
write_elmo_file(directory_write+".dev.elmo.0", dev_x_0)

write_elmo_file(directory_write+".train.elmo.1", train_x_1)
write_elmo_file(directory_write+".test.elmo.1", test_x_1)
write_elmo_file(directory_write+".dev.elmo.1", dev_x_1)

'''
train0elmo= read_elmo_file(directory_write,".train.elmo.0")
'''
print('\n',111111111111111111)
print(len(train0elmo[0]), len(train_x_0[0]))
print(type(train0elmo[0][0]), type(train_x_0[0][0]))
print(type(train0elmo),type( train_x_0))
