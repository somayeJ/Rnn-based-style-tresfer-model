# me: comparing the elmo rep of the seqs of the files given in the directories: directory_orig and directory_tgt, to calacute the similarity of the seqs and files (avg of all seq similarities)
import keras
import os
from scipy import spatial
import sys
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import random
import torch
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Activation, Dropout
from StyleMarkerRemover import StyleMarkerRemover_elmo
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#remove_style_markers = False
#models = [".rec", ".tsf"] # like rec or tsf
#labels = [".0", ".1"] # like 0 or 1

directory_orig =  "../../../../data/yelp/" #  orig files
#directory_orig_elmo =  "../../../../data/test_yelp_elmo_2_rep/" # elm_rep of orig files
#directory_orig_elmo =  "../../../../data/yelp_elmo_rep_seqs/"
directory_orig_amazon_elmo =  "../../../../data/amazon_elmo_seq/"


directory_tgt = "../../../../tmp/CrossAligned_AE_emb/" # trg files
#directory_tgt_elmo = "../../../../tmp/CrossAligned_AE_emb/test_elmo_rep_test_gen_files/" #elm_rep of trg files
directory_tgt_elmo = "../../../../tmp/CrossAligned_AE_emb/elmo_rep_seqs/" #elm_rep of trg files
'''

if not os.path.isfile(directory_orig_elmo+'sentiment.test.elmo.0'):
    print("-----------------------There is no elm_rep of orig files in this directory-----------------------",directory_orig_elmo)
    print("-----------------------You should generate the elm_rep of orig files  using Elmo_token_rep_write.py-----------------------")
    exit()
if not os.path.isfile(directory_tgt_elmo+'sentiment.test.elmo.rec.0'):
    print("-----------------------There is no elm_rep of trg files in this directory-------------------",directory_tgt_elmo)
    print("-----------------------You should generate the elm_rep of trg files  using Elmo_token_rep_write.py---------------------")
    exit()
else:
    print("Start to calcualte the distance between the elmo rep of the seqs in the files of directories:", directory_orig_elmo, directory_tgt_elmo)
'''

def read_data(directory,file_name):
    files_dir = directory+file_name
    with open(files_dir, 'r') as f1:
        #reviews_raw=f1.readlines()[:2]
        reviews_raw=f1.readlines()
        print(files_dir, len(reviews_raw))
    data = []
    for review in reviews_raw:
        data.append({"text": review.replace("<br />", " ").replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ').strip()})
    return data

#train_data, test_data = download_and_load_datasets()
def load_dataset(directory, file_name):
    data = read_data(directory,file_name)
    return data

#Tokenize text. Note, it would be better to first split it into sentences.
def tokenize_text(documents, max_tokens):
    for document in documents:
        document['tokens'] = keras.preprocessing.text.text_to_word_sequence(document['text'], lower=False)
        if len(document['tokens']) == 0:
            document['tokens'].append('the')
        document['tokens'] = document['tokens'][0:max_tokens]

def write_elmo_file(directory_write, file):
    np.savetxt(directory_write, file)
    return

def read_elmo_file( directory, file):
    x = np.loadtxt(directory +file)
    return x.tolist()[:2]

def com_sent(emb0, emb1):

    result = 1 - spatial.distance.cosine(emb0, emb1)
    return result

def com_file(q_file, r_file, w_file):
    """
    :param q_file:
    :param r_file:
    :param w_file:
    :param word_dict:
    :return: writes the distance between the embedding ses of the two sentences on each file in w_file
    """
    #q_file = open(q_file, "r")
    #r_file = open(r_file, "r")
    w_file = open(w_file, "w")
    q_file_lines = q_file
    r_file_lines = r_file
    res = []

    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score = com_sent(line0, line1)
        res.append(score)
        #print(score, line0, line1)
        w_file.write(str(score)+"\n")
    w_file.close()
    return res

def gen_score(directory_tgt_elmo,q_file_elmo, r_file_elmo, w_file_suffix):
    '''
    emb = Embedding(100)
    # word_dict: a dict with the name as the key and word embeddings vectors as the value
    word_dict = emb.get_all_emb()
    subdir_names = ['multi_decoder', 'embedding', 'memory']
    # get_sub_dirnames(test_dir_name) = os.listdir(test_dir_name)
    subdir_names = get_sub_dirnames(test_dir_name)
    subdir_names = [i for i in subdir_names if not i.endswith("txt")]

    for dir_name in subdir_names:
        for index_name in ["0", "1"]:
            q_file = test_dir_name+"sentiment.test."+index_name
            #r_file = test_dir_name+"/"+dir_name+"/style"+index_name+".txt"
            r_file = test_dir_name + "sentiment.test." + index_name +suffix
            print(q_file, r_file)
            w_file = test_dir_name+"style"+index_name+"_semantics.txt"
            com_file(q_file, r_file, w_file, word_dict)
    '''
    res =[]

    w_file = directory_tgt_elmo +"comparison" + w_file_suffix
    res.append(np.mean(com_file(q_file_elmo, r_file_elmo,w_file)))

    return res
def gen_score_no_style_markers(directory_tgt_elmo,q_file_elmo_all_tokens, r_file_elmo_all_tokens, w_file_suffix,q_file,r_file):
    '''
    q_file: origional file
    r_file: generated file (style_shifted)
    '''
    res =[]
    w_file = directory_tgt_elmo +"comparison" + w_file_suffix
    
    print("__________Filtering style markers__________")
    if w_file_suffix[-1] =='0':
    	word_list_file_style_orig =  "../opinion-lexicon-English/words-cleaned." + '0' # 0: negative
    	word_list_file_style_tgt =  "../opinion-lexicon-English/words-cleaned." + '1' # 1: positive

    else:
    	word_list_file_style_orig =  "../opinion-lexicon-English/words-cleaned." + '1'
    	word_list_file_style_tgt =  "../opinion-lexicon-English/words-cleaned." + '0' 

    remover_class_orig = StyleMarkerRemover_elmo(q_file,word_list_file_style_orig)
    kept_tokens_indecies_orig = remover_class_orig.get_all_content_sequences()
    print('kept_tokens_indecies_orig', kept_tokens_indecies_orig)
    q_file_elmo=[]
    for elmo_seq in q_file_elmo_all_tokens:
    	q_file_elmo.append([token for token in elmo_seq if elmo_seq.index(token) in kept_tokens_indecies_orig ]) # list from which style markers are removed
    print('*********************************q_file_elmo', len(q_file_elmo), q_file_elmo[:2])
    remover_class_tgt = StyleMarkerRemover_elmo(r_file,word_list_file_style_tgt)
    kept_tokens_indecies_tgt = remover_class_tgt.get_all_content_sequences()
    print('kept_tokens_indecies_tgt',kept_tokens_indecies_tgt)
    r_file_elmo=[]

    for elmo_seq in r_file_elmo_all_tokens:
    	r_file_elmo.append([token for token in elmo_seq if elmo_seq.index(token) in kept_tokens_indecies_tgt]) # list from which style markers are removed
    print('*********************************r_file_elmo', len(r_file_elmo), r_file_elmo[:2])
    res.append(np.mean(com_file(q_file_elmo, r_file_elmo,w_file)))

    return res

print("______________________________reading the embedding files from tgt files____________________________________")

data_elmo_rec0 = read_elmo_file(directory_tgt_elmo,'sentiment.test.elmo.rec'+'.0' )
#print(1111111111111111,len(data_elmo_rec0), type(data_elmo_rec0[0]), type(data_elmo_rec0[0][0]), len (data_elmo_rec0[0]))
data_elmo_rec1 = read_elmo_file(directory_tgt_elmo,'sentiment.test.elmo.rec'+'.1' )
#data_elmo_tsf0 = read_elmo_file(directory_tgt_elmo,'sentiment.test.elmo.tsf'+'.0' )
#data_elmo_tsf1 = read_elmo_file(directory_tgt_elmo,'sentiment.test.elmo.tsf'+'.1' )

print("______________________________reading the embedding files from orig files____________________________________")

#yelp_elmo_test0 = read_elmo_file(directory_orig_elmo, 'sentiment.test.elmo.0')
#yelp_elmo_test1 = read_elmo_file(directory_orig_elmo, 'sentiment.test.elmo.1')
amazon_elmo_test0 = read_elmo_file(directory_orig_amazon_elmo, 'sentiment.test.elmo.0')
amazon_elmo_test1 = read_elmo_file(directory_orig_amazon_elmo, 'sentiment.test.elmo.1')
#print(33333333333333,len(yelp_elmo_test0),type(yelp_elmo_test0[0]),type(yelp_elmo_test0[0][0]),len(yelp_elmo_test0[0]))
print("______________________________Computing distance between sequences____________________________________________")

for remove_style_markers in [False]:
    if remove_style_markers: # it means we are considering tsf files
        max_tokens = 100
        yelp_test0 = load_dataset(directory_orig,"sentiment.test.0")
        yelp_test0_tsf = load_dataset(directory_tgt,"sentiment.test.0.tsf")
        tokenize_text(yelp_test0, max_tokens) # dictionary
        tokenize_text(yelp_test0_tsf, max_tokens)
        yelp_test0_tokens= [document["text"] for document in yelp_test0] # lists of seqs tokens_list  
        yelp_test0_tsf_tokens= [document["text"] for document in yelp_test0_tsf]
        #print('---content preservation score baesd on comparing yelp_sentiment_test0','and its style_shifted file:---', gen_score_no_style_markers(directory_tgt_elmo,yelp_elmo_test0,data_elmo_tsf0,'test0Andtsf0',yelp_test0_tokens,yelp_test0_tsf_tokens))
        print('---content preservation score baesd on comparing yelp_sentiment_test0' ,'and its style_shifted file:---', gen_score(directory_tgt_elmo,yelp_elmo_test0,data_elmo_tsf0,'test0Andtsf0'))


        yelp_test1 = load_dataset(directory_orig,"sentiment.test.1")
        yelp_test1_tsf = load_dataset(directory_tgt,"sentiment.test.1.tsf")
        tokenize_text(yelp_test1, max_tokens) # dictionary
        tokenize_text(yelp_test1_tsf, max_tokens)
        yelp_test1_tokens= [document["text"] for document in yelp_test1] # list
        yelp_test1_tsf_tokens= [document["text"] for document in yelp_test1_tsf]
        #print('---content preservation score baesd on comparing yelp_sentiment_test1','and its style_shifted file:---', gen_score_no_style_markers(directory_tgt_elmo,yelp_elmo_test1,data_elmo_tsf1,'test1Andtsf1',yelp_test1_tokens,yelp_test1_tsf_tokens))
        print('---content preservation score baesd on comparing yelp_sentiment_test1' ,'and its style_shifted file:---', gen_score(directory_tgt_elmo,yelp_elmo_test1,data_elmo_tsf1,'test1Andtsf1'))


    else:
        #print('---content preservation score baesd on comparing yelp_sentiment_test0' ,'and its reconstructed file:---', gen_score(directory_tgt_elmo,yelp_elmo_test0,data_elmo_rec0,'amazon0Andyelprec0'))
        #print('---content preservation score baesd on comparing yelp_sentiment_test1' ,'and its reconstructed file:---', gen_score(directory_tgt_elmo,yelp_elmo_test1,data_elmo_rec1,'amazon1Andrec1'))

        print('---content preservation score baesd on comparing amazon_sentiment_test0' ,'and yelp reconstructed file:---', gen_score(directory_tgt_elmo,amazon_elmo_test0,data_elmo_rec0,'amazon0Andyelprec0'))
        print('---content preservation score baesd on comparing amazon_sentiment_test1' ,'and yelp reconstructed file:---', gen_score(directory_tgt_elmo,amazon_elmo_test1,data_elmo_rec1,'amazon1Andyelprec1'))

'''
if remove_style_markers: # it means we are considering tsf files
    max_tokens = 100
    yelp_test0 = load_dataset(directory_orig,"sentiment_test_0")
    yelp_test0_tsf = load_dataset(directory_tgt,"sentiment.test.0.tsf")
    tokenize_text(yelp_test0, max_tokens) # dictionary
    tokenize_text(yelp_test0_tsf, max_tokens)
    yelp_test0_tokens= [document["text"] for document in yelp_test0] # lists of seqs tokens_list  
    yelp_test0_tsf_tokens= [document["text"] for document in yelp_test0_tsf]
    print('---content preservation score baesd on comparing yelp_sentiment_test0','and its style_shifted file:---', gen_score_no_style_markers(directory_tgt_elmo,yelp_elmo_test0,data_elmo_tsf0,'test0Andtsf0',yelp_test0_tokens,yelp_test0_tsf_tokens))


    yelp_test1 = load_dataset(directory_orig,"sentiment_test_1")
    yelp_test1_tsf = load_dataset(directory_tgt,"sentiment.test.1.tsf")
    tokenize_text(yelp_test1, max_tokens) # dictionary
    tokenize_text(yelp_test1_tsf, max_tokens)
    yelp_test1_tokens= [document["text"] for document in yelp_test1] # list
    yelp_test1_tsf_tokens= [document["text"] for document in yelp_test1_tsf]
    print('---content preservation score baesd on comparing yelp_sentiment_test1','and its style_shifted file:---', gen_score_no_style_markers(directory_tgt_elmo,yelp_elmo_test1,data_elmo_tsf1,'test1Andtsf1',yelp_test1_tokens,yelp_test1_tsf_tokens))

else:
    print('---content preservation score baesd on comparing yelp_sentiment_test0' ,'and its reconstructed file:---', gen_score(directory_tgt_elmo,yelp_elmo_test0,data_elmo_rec0,'test0Andrec0'))
    print('---content preservation score baesd on comparing yelp_sentiment_test1' ,'and its reconstructed file:---', gen_score(directory_tgt_elmo,yelp_elmo_test1,data_elmo_rec1,'test1Andrec1'))

'''



