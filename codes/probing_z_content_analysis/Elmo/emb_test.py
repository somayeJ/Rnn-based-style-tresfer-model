#encoding=utf-8
# this code is for calculating the content preservation and is taken from the site https://github.com/fuzhenxin/text
# _style_transfer  to run the code we should indicate the address of files in terminal and also give the address of
# where the embedding files are stored in Embedding.py file
# example of running code in terminal python emb_test.py ../eval/model_outputs/AE
#from Embedding import Embedding
import sys
import numpy as np
from scipy import spatial
from Tool import get_sub_dirnames
import random
word_dict ={}
def get_sent_emb(line, word_dict):
    '''
    creating the sentence embedding: [mean, min, max]
    :param line:
    :param word_dict:
    :return:
    '''
    '''
    no_sequences_with_zero_found_word=0
    line_split = line.strip().split()
    res = []
    for i in line_split:
        if i in word_dict:
            res.append(word_dict[i])
    if len(res)==0:
        res.append(word_dict["the"])
        print ("all word not found")
    res = np.array(res)
    '''
    # inserting 0 in np.mean(res, 0) instead of np.mean(res) returns back an array to us
    mm = np.mean(line, 0)
    mi = np.min(line, 0)
    ma = np.max(line, 0)
    emb = np.concatenate((mm, mi, ma))
    #emb = mm
    return emb


def com_sent(line0, line1, word_dict):
    emb0 = get_sent_emb(line0, word_dict)
    emb1 = get_sent_emb(line1, word_dict)
    result = 1 - spatial.distance.cosine(emb0, emb1)
    return result

def com_file(q_file, r_file, w_file, word_dict):
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

    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)
        w_file.write(str(score)+"\n")
    w_file.close()
    return

def com_file_score(q_file, r_file, word_dict):
    '''
    the difference between this ethod and com_file is that, this one returns a list , and com_file is writes in a file
    :param q_file:
    :param r_file:
    :param word_dict:
    :return:
    '''
    #q_file = open(q_file, "r")
    #r_file = open(r_file, "r")
    q_file_lines = q_file
    r_file_lines = r_file
    #q_file.close()
    #r_file.close()

    res = []
    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)

        res.append(score)
    return res

def gen_score(q_file,r_file):
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

    #q_file = test_dir_name + "sentiment.test.0.txt"
    # r_file = test_dir_name+"/"+dir_name+"/style"+index_name+".txt"
    #r_file = test_dir_name + "sentiment.test.0.paraphrase.txt"

    w_file = "../output_emb_files/Elmo_model/semantic_distance_ofq_fileANDr_file"
    com_file(q_file, r_file, w_file, word_dict)
    res.append(np.mean(com_file_score(q_file, r_file, word_dict)))

    return res




'''


if __name__=="__main__":
    test_dir_name = sys.argv[1]
    #print('random_content_reservation', random_content_reservation(test_dir_name))

    res= gen_score(test_dir_name,'.rec') # res is a list
    #print(res)
    #print('content preservation score baesd on comparing_files',"=",res[0])

'''