#encoding=utf-8
# this code is for calculating the content preservation and is taken from the site https://github.com/fuzhenxin/text_style_transfer
# to run the code we should indicate the address of folder with files in terminal, this folder should consist of the original
# & their corresponding generated files
# and also give the address of where the embedding files are stored in the file of  Embedding.py which is called inside this code

# to run the code type the following in terminal python emb_test.py "folder address containing griginal and generated files"
# for example: python emb_test.py "/home/sjafarit/PhD/shared_folder/embeddings/data/yelp/"


from Embedding_gensim import Embedding
import sys
import numpy as np
from scipy import spatial
from Tool import get_sub_dirnames
import random

def get_sent_emb(line, word_dict):
    '''
    creating the sentence embedding: [mean, min, max]
    :param line:
    :param word_dict:
    :return:
    '''
    sequences_zero_found_word=0
    line_split = line.strip().split()
    res = []
    for i in line_split:
        if i in word_dict:
            res.append(word_dict[i])
    if len(res)==0:
        res.append(word_dict["the"])
        print "all word not found"
        sequences_zero_found_word=1
    res = np.array(res)
    # inserting 0 in np.mean(res, 0) instead of np.mean(res) returns back an array to us
    mm = np.mean(res, 0)
    mi = np.min(res, 0)
    ma = np.max(res, 0)
    emb = np.concatenate((mm, mi, ma))
    #emb = mm
    return emb, sequences_zero_found_word


def com_sent(line0, line1, word_dict):
    emb0, no0 = get_sent_emb(line0, word_dict)
    emb1, no1 = get_sent_emb(line1, word_dict)
    result = 1 - spatial.distance.cosine(emb0, emb1)
    return result, no0, no1

def com_file(q_file, r_file, w_file, word_dict):
    """

    :param q_file:
    :param r_file:
    :param w_file:
    :param word_dict:
    :return: writes the distance between the embedding ses of the two sentences on each file in w_file
    """
    no0s= []
    no1s = []
    q_file = open(q_file, "r")
    r_file = open(r_file, "r")
    w_file = open(w_file, "w")
    q_file_lines = q_file.readlines()
    r_file_lines = r_file.readlines()
    q_file.close()
    r_file.close()
    print('len','origional_test_file_lines',len(q_file_lines),'dev_file_lines', len(r_file_lines))
    
    if len(q_file_lines)>len(r_file_lines):
        q_file_lines =q_file_lines[:len(r_file_lines)]
    elif len(q_file_lines)<len(r_file_lines):
        r_file_lines =r_file_lines[:len(q_file_lines)]
    
    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score, no0, no1 = com_sent(line0, line1, word_dict)
        w_file.write(str(score)+"\n")
        no1s.append(no1)
        no0s.append(no0)
    w_file.close()
    print ("tedad jomalat ke 0 kalamashoon to dict peda nashode dar file 0 & file 1 ", len(no0s),sum(no0s),len(no1s), sum(no1s), len(r_file_lines),len(q_file_lines))
    return 

def com_file_score(q_file, r_file, word_dict):
    '''
    the difference between this ethod and com_file is that, this one returns a list , and com_file is writes in a file
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
    
    if len(q_file_lines)>len(r_file_lines):
        q_file_lines =q_file_lines[:len(r_file_lines)]
    elif len(q_file_lines)<len(r_file_lines):
        r_file_lines =r_file_lines[:len(q_file_lines)]
    
    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score, n0,n1 = com_sent(line0, line1, word_dict)
        print(score, line0, line1)

        res.append(score)
    return res

def gen_score(test_dir_name, suffix=''):
    emb = Embedding(100)
    # word_dict: a dict with the name as the key and word embeddings vectors as the value
    word_dict = emb.get_all_emb()
    subdir_names = ['multi_decoder', 'embedding', 'memory']
    # get_sub_dirnames(test_dir_name) = os.listdir(test_dir_name)
    subdir_names = get_sub_dirnames(test_dir_name)
    subdir_names = [i for i in subdir_names if not i.endswith("txt")]
    '''
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
    for index_name in ["0", "1"]:
        q_file = test_dir_name + "sentiment.test." + index_name
        # r_file = test_dir_name+"/"+dir_name+"/style"+index_name+".txt"
        r_file = test_dir_name + "sentiment.test." + index_name + suffix

        w_file = test_dir_name + "style" + index_name + "_semantics.txt"
        com_file(q_file, r_file, w_file, word_dict)
        res.append(np.mean(com_file_score(q_file, r_file, word_dict)))

    return res



def random_content_reservation(test_dir_name):
    with open (test_dir_name + "sentiment.test." + '0','r') as f00, open (test_dir_name + "sentiment.test." + '1','r') as f10:
        #f00 = test_dir_name + "sentiment.test." + '0'
        #f10 = test_dir_name + "sentiment.test." + '1'
        sents_0 = f00.readlines()[:50000]
        sents_1 = f10.readlines()[:50000]

    with open( test_dir_name+'test_0_samples.txt','w') as fi0,  open( test_dir_name+'test_1_samples.txt','w') as fi1:
        for s0, s1 in zip(sents_0,sents_1):
            fi0.write(s0)

            fi0.write('\n')
            fi1.write(s1)

            fi1.write('\n')

    emb = Embedding(300)
    word_dict = emb.get_all_emb()

    f0 = test_dir_name + 'test_0_samples.txt'
    f1 = test_dir_name + 'test_1_samples.txt'



    scores = com_file_score(f0, f1, word_dict)


    scores = np.array(scores, dtype=np.float32)
    print(scores)


    return np.mean(scores)


if __name__=="__main__":
    test_dir_name = sys.argv[1]
    #print('random_content_reservation', random_content_reservation(test_dir_name))
    res= gen_score(test_dir_name,'.tsf') # res is a list
    print('content preservation score baesd on comparing test_file_0 and its generated file',res[0],'content preservation score baesd on comparing test_file_1 and its generated file',res[1])

