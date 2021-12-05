
# me: modofied Elmo_keras.py for writing the elmo embeddings of files of yelp data in directory_write, if we want to compute the elmo rep of train, test and dev
# seqs, we should call load_all_dataset() function, and determine the mode and label,otherwise, we load load_dataset() function  and determine the mode and label.
import keras
import os
import sys
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import random
import torch
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Activation, Dropout
from StyleMarkerRemover import StyleMarkerRemover_elmo, StyleMarkerRemover
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
directory = "../../../../data/yelp/sentiment.test" # yelp_dataset
directory_amazon = "../../../../data/amazon/binary/sentiment.test" # yelp_dataset

#directory_write_yelp = "../../../../data/yelp_elmo_rep_tokens/sentiment.test" # yelp_dataset
directory_write_amazon = "../../../../data/amazon_elmo_seq/sentiment.test" # yelp_dataset

#directory_read = "../../../../tmp/CrossAligned_AE_emb/sentiment.test"
#directory_write = "../../../../tmp/CrossAligned_AE_emb/elmo_rep_seq/sentiment.test.elmo."
def read_data_comapre(directory1,directory2,mode,label):
    files_dir1 = directory1+mode+label
    files_dir2 = directory2+mode+label

    with open(files_dir1, 'r') as f1, open(files_dir2, 'r') as f2:
        reviews_raw1=f1.readlines()
        reviews_raw2=f2.readlines()[:len(reviews_raw1)]
        i=0
        while len(reviews_raw2)< len(reviews_raw1):
            reviews_raw2.append(reviews_raw2[i])
            i+=1
        print(directory2,mode,label, len(reviews_raw2))
    data = []
    for review in reviews_raw2:
        data.append({"text": review.replace("<br />", " ").replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ').strip(), "label": label})
    return data


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
def load_dataset_compare(directory1,directory2, mode, label):
    data = read_data_comapre(directory1,directory2,mode,label)

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

def create_elmo_embeddings(elmo, documents,suff, seq_embedding=False,max_sentences = 0):
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
        if seq_embedding:
            embeddings.append(get_sent_emb(avg_elmo_embedding)) # seq emb: the concatenation of the [average, min , max ]
        elif suff == 'tsf0':
            remover_class_orig = StyleMarkerRemover_elmo(q_file_list,word_list_file_style_orig)
            embeddings.append(avg_elmo_embedding) #avg_elmo_embedding.shape = no_tokens*1024
        

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

'''
train_data_0, dev_data_0, test_data_0 = load_all_dataset(directory,"0")
train_data_1, dev_data_1, test_data_1 = load_all_dataset(directory,"1")
max_tokens = 100
tokenize_text(train_data_0, max_tokens)
tokenize_text(dev_data_0, max_tokens)
tokenize_text(test_data_0, max_tokens)
tokenize_text(train_data_1, max_tokens)
tokenize_text(dev_data_1, max_tokens)
tokenize_text(test_data_1, max_tokens)
'''

#data_rec0 = load_dataset(directory_read,".0", ".rec")
#data_rec1 = load_dataset(directory_read,".1", ".rec")
#data_tsf0 = load_dataset(directory_read,".0", ".tsf")
#data_tsf1 = load_dataset(directory_read,".1", ".tsf")
#yelp0 = load_dataset(directory,".", "0")
#yelp1 = load_dataset(directory,".", "1")
amazon0 = load_dataset_compare(directory,directory_amazon, ".", "0")
amazon1 = load_dataset_compare(directory,directory_amazon, ".", "1")


max_tokens = 100
print('-----------------tokenizing------------------')
#tokenize_text(data_rec0, max_tokens)
#tokenize_text(data_rec1, max_tokens)
#tokenize_text(data_tsf0, max_tokens)
#tokenize_text(data_tsf1, max_tokens)
#tokenize_text(yelp0, max_tokens)
#tokenize_text(yelp1, max_tokens)

tokenize_text(amazon0, max_tokens)
tokenize_text(amazon1, max_tokens)

#elmo = ElmoEmbedder() #Set cuda_device to the ID of your GPU if you have one
#elmo = ElmoEmbedder(cuda_device=0) #Set cuda_device to the ID of your GPU if you have one
elmo = ElmoEmbedder(cuda_device=torch.cuda.current_device()) #Set cuda_device to the ID of your GPU if you have one
# https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu, based on which the changes are made for running on gpu

'''

data_rec0_elmo, labels_rec0 = create_elmo_embeddings(elmo, data_rec0,'rec0', seq_embedding=False)
print('-----------------saving the files------------------')
write_elmo_file(directory_write+"rec.0", data_rec0_elmo)
data_rec1_elmo, labels_rec1 = create_elmo_embeddings(elmo, data_rec1,'rec1' ,seq_embedding=False)
print('-----------------saving the files------------------')
write_elmo_file(directory_write+"rec.1", data_rec1_elmo)

data_tsf0_elmo, labels_tsf0 = create_elmo_embeddings(elmo, data_tsf0,'tsf0', seq_embedding=False)
print('-----------------saving the files------------------')
write_elmo_file(directory_write+"tsf.0", data_tsf0_elmo)
data_tsf1_elmo, labels_tsf1 = create_elmo_embeddings(elmo, data_tsf1,'tsf1' ,seq_embedding=False)
print('-----------------saving the files------------------')
write_elmo_file(directory_write+"tsf.1", data_tsf1_elmo)
yelp0_elmo,l0 =create_elmo_embeddings(elmo,yelp0, 'yelp0')
yelp1_elmo,l1 =create_elmo_embeddings(elmo,yelp1,'yelp1')

'''
amazon0_elmo,l0 =create_elmo_embeddings(elmo,amazon0, 'amazon0',  seq_embedding=True)
amazon1_elmo,l1 =create_elmo_embeddings(elmo,amazon1,'amazon1', seq_embedding= True)


write_elmo_file(directory_write_amazon+".elmo.0", amazon0_elmo)
write_elmo_file(directory_write_amazon+".elmo.1", amazon1_elmo)

'''
train0elmo= read_elmo_file(directory_write,".train.elmo.0")
'''


