#encoding=utf-8
# this code is for calculating the content preservation and is taken from the site https://github.com/fuzhenxin/text_style_transfer
# to run the code we should indicate the address of files in terminal and also give the address of where the embedding files are stored in Embedding.py file
# example of running code in terminal python emb_test.py ../eval/model_outputs/AE, and also u should put the test files in the same file as the outputs, also in the 
# method gen_score(), u should put the suffix tsf or rec
from Embedding import Embedding
import sys
import numpy as np
from scipy import spatial
from Tool import get_sub_dirnames
import random
random_content_preservation = False

def get_sent_emb(line, word_dict):
    '''
    creating the sentence embedding: [mean, min, max]
    :param line:
    :param word_dict:
    :return:
    '''
    no_sequences_with_zero_found_word=0
    line_split = line.strip().split()
    res = []
    for i in line_split:
        if i in word_dict:
            res.append(word_dict[i])
    if len(res)==0:
        res.append(word_dict["the"])
        #print "all word not found"
    res = np.array(res)
    # inserting 0 in np.mean(res, 0) instead of np.mean(res) returns back an array to us
    mm = np.mean(res, 0)
    mi = np.min(res, 0)
    ma = np.max(res, 0)
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
    q_file = open(q_file, "r")
    r_file = open(r_file, "r")
    w_file = open(w_file, "w")
    q_file_lines = q_file.readlines()
    r_file_lines = r_file.readlines()
    print(q_file, r_file_lines[:1])
    print(r_file, q_file_lines[:1])
    q_file.close()
    r_file.close()
    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)
        w_file.write(str(score)+"\n")
    w_file.close()
    return

def com_file_random(q_file, generated_file, w_file, word_dict):

    q_file = open(q_file, "r")
    generated_file = open(generated_file, "r")
    w_file = open(w_file, "w")
    generated_file_lines = generated_file.readlines()
    q_file_lines = q_file.readlines() [:len(generated_file_lines)]

    print(q_file, generated_file_lines[:1])
    print(generated_file, generated_file_lines[:1])
    q_file.close()
    generated_file.close()
    assert len(q_file_lines)==len(generated_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, generated_file_lines):
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
    q_file = open(q_file, "r")
    r_file = open(r_file, "r")
    q_file_lines = q_file.readlines()
    r_file_lines = r_file.readlines()
    print(q_file, r_file_lines[:1])
    print(r_file, q_file_lines[:1])
    q_file.close()
    r_file.close()

    res = []
    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)

        res.append(score)
    return res
def com_file_score_random(random_file,generated_file, word_dict):
    '''
    the difference between this ethod and com_file is that, this one returns a list , and com_file is writes in a file
    :param q_file:
    :param r_file:
    :param word_dict:
    :return:
    '''
    random_file = open(random_file, "r")
    generated_file = open(generated_file, "r")
    generated_file_lines = generated_file.readlines()
    random_file_lines = random_file.readlines()[:len(generated_file_lines)]


    random_file.close()
    generated_file.close()

    res = []
    assert len(random_file_lines)==len(generated_file_lines), "length error"
    for line0, line1 in zip(random_file_lines, generated_file_lines):
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)

        res.append(score)
    return res


def gen_score(test_dir_name, suffix=''):
    emb = Embedding(100)
    # word_dict: a dict with the name as the key and word embeddings vectors as the value
    word_dict = emb.get_all_emb()
    '''
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
    for index_name in ["0", "1"]:
        q_file = test_dir_name + "sentiment.test." + index_name
        # r_file = test_dir_name+"/"+dir_name+"/style"+index_name+".txt"
        r_file = test_dir_name + "sentiment.test." + index_name + suffix

        w_file = test_dir_name + "style" + index_name + "_semantics.txt"
        com_file(q_file, r_file, w_file, word_dict)
        res.append(np.mean(com_file_score(q_file, r_file, word_dict)))

    return res, test_dir_name + "sentiment.test.", test_dir_name + "sentiment.test.0" + suffix


def gen_score_random(test_dir_name_generated, test_dir_name_random, suffix=''):
    '''

    :param test_dir_name_generated: path/sentiment.test.
    :param test_dir_name_origional: path/sentiment.train.
    :param suffix: this parameter is given value in main method, which is normaly a suffix for generated files
    :return:
    '''
    emb = Embedding(100)
    # word_dict: a dict with the name as the key and word embeddings vectors as the value
    word_dict = emb.get_all_emb()
    '''
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
    for index_name in ["0", "1"]:
        random_file = test_dir_name_random  + index_name
        # r_file = test_dir_name+"/"+dir_name+"/style"+index_name+".txt"
        generated_file = test_dir_name_generated  + index_name + suffix

        w_file = test_dir_name_generated + "style" + index_name + "_semantics.txt"
        com_file_random(random_file, generated_file, w_file, word_dict)
        res.append(np.mean(com_file_score_random(random_file,generated_file, word_dict)))

    return res, test_dir_name_generated + suffix, test_dir_name_random



def random_content_reservation(test_dir_name):
    with open (test_dir_name + "sentiment.test." + '0','r') as f00, open (test_dir_name + "sentiment.test." + '1','r') as f10:
        #f00 = test_dir_name + "sentiment.test." + '0'
        #f10 = test_dir_name + "sentiment.test." + '1'
        sents_0 = f00.readlines()[:50000]
        sents_1 = f10.readlines()[:50000]

    with open( test_dir_name+'test_0_samples.txt','w') as fi0,  open( test_dir_name+'test_1_samples.txt','w') as fi1:
        for s0, s1 in zip(sents_0,sents_1):
            fi0.write(s0)

            #fi0.write('\n')
            fi1.write(s1)

            #fi1.write('\n')

    emb = Embedding(100)
    word_dict = emb.get_all_emb()
    with open (test_dir_name + 'sentiment.train.amazon.0.txt','r') as f00, open (test_dir_name + 'sentiment.test.0.tsf','r') as f10:
        #f00 = test_dir_name + "sentiment.test." + '0'
        #f10 = test_dir_name + "sentiment.test." + '1'
        sents_0 = f00.readlines()[:50000]
        sents_1 = f10.readlines()[:50000]

    with open( test_dir_name+'test_0_samples_amazon.txt','w') as fi0,  open( test_dir_name+'test_0_samples_yelp.txt','w') as fi1:
        for s0, s1 in zip(sents_0,sents_1):
            fi0.write(s0)

            #fi0.write('\n')
            fi1.write(s1)


    #f0 = test_dir_name + 'test_0_samples.txt'
    #f1 = test_dir_name + 'test_1_samples.txt'
    f0 = test_dir_name + 'test_0_samples_amazon.txt'
    f1 = test_dir_name + 'test_0_samples_yelp.txt'



    scores = com_file_score(f0, f1, word_dict)


    scores = np.array(scores, dtype=np.float32)
    #print(scores)


    return np.mean(scores)


if __name__=="__main__":
    test_dir_name = sys.argv[1]
    #print('random_content_reservation', random_content_reservation(test_dir_name))
    if random_content_preservation:
        res, generated_file, random_file=gen_score_random(test_dir_name, sys.argv[2], '.tsf')
        print('content preservation score baesd on comparing generated_file', generated_file  ,".0",'and a random file', random_file,":",res[0],'content preservation score baesd on comparing generated_file1 and a random file',res[1])
    else:
        res, q_file, r_file= gen_score(test_dir_name,'.rec') # res is a list
        print('content preservation score baesd on comparing test_file_0' ,q_file  ,".0",'and its generated file', r_file,":",res[0],'content preservation score baesd on comparing test_file_1 and its generated file',res[1])

