# me: modofied Elmo_keras.py for yelp data
# Insert the directory from which the data is loaded
import keras
import os
import sys
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Activation, Dropout

directory = "../data/yelp/sentiment" # yelp_dataset

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
def load_dataset(directory, mode):
    neg_data = read_data(directory,mode,"0")
    pos_data = read_data(directory,mode,"1")
    return pos_data+neg_data


def load_all_dataset(directory):
    """
    :param directory: the directory from which the data is loaded
    :return: data1, data2
    data1: list of dictionaries corresponding to each review consisting of the three components as follows:
    {'tokens': 'text', label: '1 or 0' , 'tokens':[...]}
    """
    #train_data = load_dataset(directory,".train.")
    #dev_data = load_dataset(directory,".dev.")
    test_data = load_dataset(directory, ".test.")
    return  test_data


#random.shuffle(train_data)
#random.shuffle(test_data)


#Tokenize text. Note, it would be better to first split it into sentences.
def tokenize_text(documents, max_tokens):
    for document in documents:
        document['tokens'] = keras.preprocessing.text.text_to_word_sequence(document['text'], lower=False)
        if len(document['tokens']) == 0:
            document['tokens'].append('the')
        document['tokens'] = document['tokens'][0:max_tokens]

max_tokens = 100
#tokenize_text(train_data, max_tokens)
#tokenize_text(dev_data, max_tokens)
#tokenize_text(test_data, max_tokens)
#print (train_data[0]['tokens'], type(train_data)) # list of tokens
# Lookup the ELMo embeddings for all documents (all sentences) in our dataset. Store those
# in a numpy matrix so that we must compute the ELMo embeddings only once.

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

#elmo = ElmoEmbedder() #Set cuda_device to the ID of your GPU if you have one
#elmo = ElmoEmbedder(cuda_device=0) #Set cuda_device to the ID of your GPU if you have one
#train_x, train_y = create_elmo_embeddings(elmo, train_data, 0)
#test_x, test_y = create_elmo_embeddings(elmo, test_data, 0)
#dev_x, dev_y  = create_elmo_embeddings(elmo, dev_data, 0)

# :: Pad the x matrix to uniform length ::
def pad_x_matrix(x_matrix):
    for sentenceIdx in range(len(x_matrix)):
        sent = x_matrix[sentenceIdx]
        sentence_vec = np.array(sent, dtype=np.float32)
        padding_length = max_tokens - sentence_vec.shape[0]
        if padding_length > 0:
            x_matrix[sentenceIdx] = np.append(sent, np.zeros((padding_length, sentence_vec.shape[1])), axis=0)

    matrix = np.array(x_matrix, dtype=np.float32)
    return matrix
'''
train_x = pad_x_matrix(train_x)
train_y = np.array(train_y)

dev_x = pad_x_matrix(dev_x)
dev_y = np.array(dev_y)
'''

#test_x = pad_x_matrix(test_x)
#test_y = np.array(test_y)

#print("Shape test_x:", test_x.shape)
# Simple model for sentence / document classification using CNN + global max pooling
'''
model = Sequential()
model.add(Conv1D(filters=250, kernel_size=3, padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, validation_data=(dev_x, dev_y), epochs=10, batch_size=32)


model.summary()
'''


