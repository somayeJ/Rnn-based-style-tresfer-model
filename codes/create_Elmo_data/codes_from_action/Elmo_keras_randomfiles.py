# me: modified Elmo_keras.py for random data which are created for investigating the performance of the models to
# produce embeddins rep of sequences
# Insert the directory from which the data is loaded
import keras
import os
import sys
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Activation, Dropout
from scipy import spatial
from Tool import get_sub_dirnames
import random
from emb_test import *

#directory = "../../../data/yelp/sentiment" # yelp_dataset
directory = "../data/"

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
    data= read_data(directory,mode,".txt")
    return data


def load_all_dataset(directory):
    """
    :param directory: the directory from which the data is loaded
    :return: data1, data2, ...
    data1: list of dictionaries corresponding to each review consisting of the three components as follows:
    {'tokens': 'text', label: '1 or 0' , 'tokens':[...]}
    """
    yelp_test_0_paraphrase = load_dataset(directory,"yelp_test_0_paraphrase")
    yelp_test_1_paraphrase = load_dataset(directory, "yelp_test_1_paraphrase")
    yelp_test_0 = load_dataset(directory,"yelp_test_0")
    yelp_test_1 = load_dataset(directory, "yelp_test_1")
    amazon_1 = load_dataset(directory,"amazon_1")
    amazon_0 = load_dataset(directory, "amazon_0")
    amazon_0_paraphrase = load_dataset(directory, "amazon_0_paraphrase")
    amazon_1_paraphrase = load_dataset(directory, "amazon_1_paraphrase")
    synthetic = load_dataset(directory,"synthetic")
    synthetic_none_sense = load_dataset(directory, "synthetic_none_sense")
    return amazon_1,amazon_0, amazon_0_paraphrase, amazon_1_paraphrase,synthetic, synthetic_none_sense,yelp_test_0,yelp_test_1,yelp_test_0_paraphrase, yelp_test_1_paraphrase


amazon_1,amazon_0,amazon_0_paraphrase, amazon_1_paraphrase, synthetic, synthetic_none_sense,yelp_test_0,yelp_test_1,yelp_test_0_paraphrase, yelp_test_1_paraphrase = load_all_dataset(directory)
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
tokenize_text(amazon_1, max_tokens)
tokenize_text(amazon_0, max_tokens)
tokenize_text(amazon_0_paraphrase, max_tokens)
tokenize_text(amazon_1_paraphrase, max_tokens)
tokenize_text(synthetic, max_tokens)
tokenize_text(synthetic_none_sense, max_tokens)
tokenize_text(yelp_test_0, max_tokens)
tokenize_text(yelp_test_1, max_tokens)
tokenize_text(yelp_test_0_paraphrase, max_tokens)
tokenize_text(yelp_test_1_paraphrase, max_tokens)

#print (train_data[0]['tokens'], type(train_data)) # list of tokens

# Lookup the ELMo embeddings for all documents (all sentences) in our dataset. Store those
# in a numpy matrix so that we must compute the ELMo embeddings only once.

def create_elmo_embeddings(elmo, documents, max_sentences = 0):
    num_sentences = min(max_sentences, len(documents)) if max_sentences > 0 else len(documents)
    print("\n\n:: Lookup of "+str(num_sentences)+" ELMo representations. This takes a while ::")
    embeddings = []
    labels = []
    tokens = [document['tokens'] for document in documents]

    for i,document in enumerate(documents):
        if len(document['tokens'])==0:
            print("!!!!!!!!!!!!!!!!!!")
            print(i, document['label'])
            print('text', document['text'])
    documentIdx = 0
    for elmo_embedding in elmo.embed_sentences(tokens):
        document = documents[documentIdx]
        # Average the 3 layers returned from ELMo, yani bara ha token size 1024 hast
        #print(tokens())
        avg_elmo_embedding = np.average(elmo_embedding, axis=0)
        embeddings.append(avg_elmo_embedding)
        labels.append(document['label'])

        # Some progress info
        documentIdx += 1
        percent = 100.0 * documentIdx / num_sentences
        line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
        status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} sentences'
        sys.stdout.write(status.format(percent, line, documentIdx, num_sentences))

        if max_sentences > 0 and documentIdx >= max_sentences:
            break

    return embeddings


elmo = ElmoEmbedder() #Set cuda_device to the ID of your GPU if you have one
#elmo = ElmoEmbedder(cuda_device=0) #Set cuda_device to the ID of your GPU if you have one
amazon_1_emb = create_elmo_embeddings(elmo, amazon_1, 0)
print('\n',len(amazon_1_emb[0]), len(amazon_1_emb[0][0]),len(amazon_1_emb), type(amazon_1_emb),type(amazon_1_emb[0]),type(amazon_1_emb[0][0]),type(amazon_1_emb[0][0][0]))
#print(amazon_0_emb)
amazon_0_emb = create_elmo_embeddings(elmo, amazon_0, 0)
amazon_0_paraphrase_emb = create_elmo_embeddings(elmo, amazon_0_paraphrase, 0)
amazon_1_paraphrase_emb = create_elmo_embeddings(elmo, amazon_1_paraphrase, 0)
synthetic_emb  = create_elmo_embeddings(elmo, synthetic, 0)
synthetic_none_sense_emb  = create_elmo_embeddings(elmo, synthetic_none_sense, 0)
yelp_test_0_emb = create_elmo_embeddings(elmo, yelp_test_0, 0)
yelp_test_1_emb = create_elmo_embeddings(elmo, yelp_test_1, 0)
yelp_test_0_paraphrase_emb  = create_elmo_embeddings(elmo, yelp_test_0_paraphrase, 0)
yelp_test_1_paraphrase_emb  = create_elmo_embeddings(elmo, yelp_test_1_paraphrase, 0)
#print('1111111111111',len(random_amazon_1_emb ), len(random_amazon_1_emb [0][0]))
#print(random_amazon_1_emb)
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

res = gen_score(amazon_1_emb,synthetic_none_sense_emb )
print('content preservation score based on comparing_files',"=",res[0])



'''
train_x = pad_x_matrix(train_x)

dev_x = pad_x_matrix(dev_x)

test_x = pad_x_matrix(test_x)

print("Shape Train X:", train_x.shape)
print("Shape Test Y:", test_x.shape)
print("Shape dev Y:", dev_x.shape)

# Simple model for sentence / document classification using CNN + global max pooling
model = Sequential()
model.add(Conv1D(filters=250, kernel_size=3, padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, validation_data=(dev_x, dev_y), epochs=10, batch_size=32)


model.summary()

'''