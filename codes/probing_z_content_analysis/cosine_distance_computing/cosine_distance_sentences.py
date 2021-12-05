#encoding=utf-8
# In this code, we do not calculate the embeddings of te sequences and we have the embedding of the full sequence
# U should determine whether we want to consider the output full vector or [ mean, min, max], by setting the value of the
# full_vector or mean_min_max_vector  equal to true at the beginning of the code
# Running code in terminal: we should determine the address of the diectory of the files we want to compute their distance
# In the terminal (example: python cosine_distance_sentences.py  ../output_emb_files/AE_pre_emb/)
# We also need to give the full name of the two files, r_file and q_file in the method gen_score() down here
#from Embedding import Embedding
import sys
import numpy as np
from scipy import spatial
#from Tool import get_sub_dirnames
import random
word_dict ={}
full_vector = True
mean_min_max_vector = False

def get_sent_emb(line, word_dict):
    '''
    creating the sentence embedding: [mean, min, max]
    :param line:
    :param word_dict:
    :return:
    '''
    # no_sequences_with_zero_found_word=0
    # line_split = line.strip().split()
    # res = []
    # for i in line_split:
    #    if i in word_dict:
    #        res.append(word_dict[i])
    # if len(res)==0:
    #   res.append(word_dict["the"])
    #    print "all word not found"
    # res = np.array(res)
    # inserting 0 in np.mean(res, 0) instead of np.mean(res) returns back an array to us
    print (type(line), len(line))
    line_emb_0 = map(lambda x: float(x), line.split())
    line_emb = np.array(line_emb_0)
    print(line_emb.shape)
    print(len(line_emb_0))
    print (type(line_emb), len(line_emb))
    print(np.mean(line_emb, 0))
    print (np.mean(line_emb))

    mm = np.mean(line_emb, 0)
    mi = np.min(line_emb, 0)
    ma = np.max(line_emb, 0)
    print(mm,mi, ma)
    print(np.array(mm))
    if full_vector:
        emb = line_emb
        print(1111,emb)
        #emb = mm
    elif mean_min_max_vector:
        emb = np.concatenate((mm, mi, ma))
    return emb

def com_sent(line0, line1, word_dict):
    emb0  = get_sent_emb(line0, word_dict)
    emb1  = get_sent_emb(line1, word_dict)
    result = 1 - spatial.distance.cosine(emb0, emb1)
    return result

def com_file(q_file, r_file, w_file, word_dict):
    """
    :param q_file:
    :param r_file:
    :param w_file:
    :param word_dict:
    :return: writes the distance between the embedding representations of the two seqs on each file in w_file
    """
    '''
    q_file = open(q_file, "r")
    r_file = open(r_file, "r")
    w_file = open(w_file, "w")
    q_file_lines = q_file.readlines()
    r_file_lines = r_file.readlines()
    print(len(q_file_lines), len(q_file_lines[0]))
    
    q_file.close()
    r_file.close()
    '''
    res=[]
    q_file_lines=np.load(q_file,allow_pickle=False ).tolist()
    r_file_lines=np.load(r_file,allow_pickle=False ).tolist()
    print(len(q_file_lines), len(q_file_lines[0]))
    assert len(q_file_lines)==len(r_file_lines), "length error"
    i=0
    for line0, line1 in zip(q_file_lines, r_file_lines):
        #score = com_sent(line0, line1, word_dict)
        score = 1 - spatial.distance.cosine(line0, line1)
        print('i,score, len(line0), len(line1)',i+1,score, len(line0), len(line1))
        i+=1
        res.append(score)

    return res

def com_file_score(q_file, r_file, word_dict):
    '''
    the difference between this method and com_file is that, this one returns a list , and com_file is writes in a file
    :param q_file:
    :param r_file:
    :param word_dict:
    :return:
    '''
    q_file = open(q_file, "r")
    r_file = open(r_file, "r")
    q_file_lines = q_file.readlines()
    r_file_lines = r_file.readlines()
    q_file.close()
    r_file.close()
    res = []
    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)

        res.append(score)
    return res

def gen_score(test_dir_name):
    '''
    :param test_dir_name:
    :param suffix:
    :return:
    '''

    #q_file = test_dir_name + "yelp_test.1.npy"
    #q_file = test_dir_name + "yelp_test.0.npy"

    q_file = test_dir_name + "gyafc.1.npy"
    #q_file = test_dir_name + "gyafc.0.npy"

    #r_file = test_dir_name + "gyafc_para.0.npy"
    #r_file = test_dir_name + "gyafc_para.1.npy"

    #r_file = test_dir_name + "yelp_test_paraphrase_ori.0.npy"
    #r_file = test_dir_name + "yelp_test_paraphrase_ori.1.npy"

    r_file = test_dir_name + "synthetic.0.npy"# random english words, random_w 
    #r_file = test_dir_name + "synthetic.1.npy"#synthetic_none_sense_emb.txt"

    #r_file = test_dir_name + "test.0.npy"
    #q_file = test_dir_name + "test.1.npy"

    #r_file = test_dir_name +"amazon.1.npy"
    #r_file = test_dir_name + "amazon.0.npy"

    #r_file = test_dir_name + "yelp_test_paraphrase.0.npy"
    #r_file = test_dir_name + "yelp_test_paraphrase.1.npy"
    w_file = test_dir_name + "cosine_sim_between"+str(r_file)+str(q_file)+".txt"
    res_list = com_file(q_file, r_file, w_file, word_dict)
    
    res = np.mean(res_list)
    return res, q_file, r_file

if __name__=="__main__":
    test_dir_name_file = sys.argv[1]
    res,f1,f2 = gen_score(test_dir_name_file) # res is a list
    print('the cosine distance between',f1,'and',f2,res)

