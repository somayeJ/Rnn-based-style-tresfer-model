
# me: modofied Elmo_keras.py for writing the elmo embeddings of files of yelp data in directory_write
# Insert the directory from which the data is loaded
import keras
import os
import sys
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Activation, Dropout

def read_data(file_dir, label =''):
    """
    :param file_dir: the diectory+filename from which the data is loaded
    :return: data as a list of dictionaries consisting of the two components of {'tokens': 'text', label: '1 or 0' }
    """
    with open(file_dir, 'r') as f1:
        reviews_raw=f1.readlines()
        print(file_dir,len(reviews_raw))
    data = []
    for review in reviews_raw:
        data.append({"text": review.replace("<br />", " ").replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ').strip(), "label": label})
    return data

# Tokenize text. Note, it would be better to first split it into sentences.
def tokenize_text(documents, max_tokens =100):
    '''
    :documents: a list of dictionaries consisting of the two components of {'tokens': 'text', label: '1 or 0'}
    '''
    for document in documents:
        document['tokens'] = keras.preprocessing.text.text_to_word_sequence(document['text'], lower=False)
        if len(document['tokens']) == 0:
            document['tokens'].append('the')
        document['tokens'] = document['tokens'][0:max_tokens]
    return

# This functions tokenizes the texts of the given documents & returns the elmo_embedings of the tokens of each seq
def create_elmo_embeddings(elmo, documents, max_sentences = 0):
    '''
    :elmo: allennlp.commands.elmo.ElmoEmbedder()
    :documents: a list of dictionaries consisting of the two components of {'tokens': 'text', label: '1 or 0'} 
    '''
    num_sentences = min(max_sentences, len(documents)) if max_sentences > 0 else len(documents)
    print("\n\n:: Lookup of "+str(num_sentences)+" ELMo representations. This takes a while ::")
    embeddings = []
    labels = []
    tokenize_text(documents)
    # creating  a list of tokens_lists
    tokens = [document['tokens']for document in documents]

    for i,document in enumerate(documents):
        if len(document['tokens'])==0:
            print("!!the length of tex is zero!!")
            print(i, document['label'])
            print('text', document['text'])

    documentIdx = 0
    for elmo_embedding in elmo.embed_sentences(tokens):
        document = documents[documentIdx]
        # Average the 3 layers returned from ELMo, baray e har token
        avg_elmo_embedding = np.average(elmo_embedding, axis=0)
        embeddings.append(avg_elmo_embedding)
        labels.append(document['label'])
        print('type(avg_elmo_embedding)',type(avg_elmo_embedding))

        # Some progress info
        documentIdx += 1
        percent = 100.0 * documentIdx / num_sentences
        line = '[{0}{1}]'.format('=' * int(percent / 2), ' ' * (50 - int(percent / 2)))
        status = '\r{0:3.0f}%{1} {2:3d}/{3:3d} sentences'
        sys.stdout.write(status.format(percent, line, documentIdx, num_sentences))
        if max_sentences > 0 and documentIdx >= max_sentences:
            break

    print('\n','len embedding data',len(embeddings), len(embeddings[0]),len(embeddings[0][0]) )
    #128 19 3072 
    return np.asarray(embeddings), labels

def write_elmo_file(directory_write, emb_array):
    file_0 = open(directory_write, 'w')
    for two_D_row in emb_array:
        np.savetxt(file_0,two_D_row)
    return

def read_elmo_file(file_name):
    x = np.loadtxt(file_name)
    return x.tolist()
