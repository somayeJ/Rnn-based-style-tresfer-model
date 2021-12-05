#encoding=utf-8
# this code is for calculating the content preservation and is taken from the site https://github.com/fuzhenxin/text_style_transfer
# to run the code we should indicate the address of files in terminal and also give the address of where the embedding files are stored 
# in Embedding.py file, and also set the Boolean variable of remove_style_markers to True if u want to remove style markers from the sequences
# which are supposed to be compared
# example of running code in terminal python emb_test.py ../eval/model_outputs/AE
# this code uses StyleMarkerRemover class to remove style markers of the inputs files and compute content precisely
from Embedding import Embedding
from StyleMarkerRemover import StyleMarkerRemover
import sys
import numpy as np
from scipy import spatial
from Tool import get_sub_dirnames
import random,  statistics

random_content_preservation = False
remove_style_markers = False

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
    q_file_lines0 = q_file.readlines()
    r_file_lines0 = r_file.readlines()
    q_file.close()
    r_file.close()
    len_q = len(q_file_lines0)
    len_r = len(r_file_lines0)
    if len_q<len_r:
        r_file_lines = r_file_lines0[:len_q]
        q_file_lines = q_file_lines0[:]
        print(11111100000000,len_q,len_r)
    elif len_r<len_q:
        q_file_lines = q_file_lines0[:len_r]
        r_file_lines = r_file_lines0[:]
        print(222222200000000000,len_q,len_r)
    else:
        print(3333300000,len_q,len_r)
        r_file_lines = r_file_lines0[:]
        q_file_lines = q_file_lines0[:]
    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)
        w_file.write(str(score)+"\n")
    w_file.close()
    return
def com_file_remove_style_markers(q_file_lines, r_file_lines, w_file, word_dict):
    """

    :param q_file:
    :param r_file:
    :param w_file:
    :param word_dict:
    :return: writes the distance between the embedding ses of the two sentences on each file in w_file
    """

    w_file = open(w_file, "w")


    assert len(q_file_lines)==len(r_file_lines), "length error"
    for (number,(line0, line1)) in enumerate(zip(q_file_lines, r_file_lines)):
        #print (number, line0, line1)
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)
        w_file.write(str(score)+"\n")
    w_file.close()
    return

def com_file_score_remove_style_markers(q_file_lines, r_file_lines, word_dict):
    '''
    the difference between this ethod and com_file is that, this one returns a list , and com_file is writes in a file
    :param q_file:
    :param r_file:
    :param word_dict:
    :return:
    '''
    res = []
    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)

        res.append(score)
    return res

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
    q_file_lines0 = q_file.readlines()
    r_file_lines0 = r_file.readlines()
    q_file.close()
    r_file.close()
    len_q = len(q_file_lines0)
    len_r = len(r_file_lines0)
    if len_q<len_r:
        r_file_lines = r_file_lines0[:len_q]
        q_file_lines = q_file_lines0[:]
        print(111111,len_q,len_r)
    elif len_r<len_q:
        q_file_lines = q_file_lines0[:len_r]
        r_file_lines = r_file_lines0[:]
        print(2222222,len_q,len_r)
    else:
        print(333333333,len_q,len_r)
        r_file_lines = r_file_lines0[:]
        q_file_lines = q_file_lines0[:]
    res = []
    assert len(q_file_lines)==len(r_file_lines), "length error"
    for line0, line1 in zip(q_file_lines, r_file_lines):
        score = com_sent(line0, line1, word_dict)
        #print(score, line0, line1)

        res.append(score)
    return res

def gen_score(test_dir_name, orig_dir_name,suffix=''):
    emb = Embedding(100)
    # word_dict: a dict with the name as the key and word embeddings vectors as the value
    word_dict = emb.get_all_emb()
    #subdir_names = ['multi_decoder', 'embedding', 'memory']
    # get_sub_dirnames(test_dir_name) = os.listdir(test_dir_name)
    '''
    subdir_names = get_sub_dirnames(test_dir_name)
    subdir_names = [i for i in subdir_names if not i.endswith("txt")]
    print(subdir_names)

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
    indexes= ["0", "1"]

    for i, index_name in enumerate(indexes):
        q_file = sys.argv[2]  + "sentiment.test." + index_name
        # r_file = test_dir_name+"/"+dir_name+"/style"+index_name+".txt"
        r_file = test_dir_name + "sentiment.test." + index_name + suffix
        word_list_file_styleorig =  "./opinion-lexicon-English/words-cleaned." + index_name
        word_list_file_styletrf =  "./opinion-lexicon-English/words-cleaned." + indexes[i-1]
        
        #print(word_list_file_styleorig, word_list_file_styletrf)
        w_file = test_dir_name + "style" + index_name + "_semantics.txt"
        if remove_style_markers:
            print(222220000000022222,word_list_file_styleorig)
            #print (11111)
            remover_class_q = StyleMarkerRemover(q_file,word_list_file_styleorig)
            q_file_lines0 = remover_class_q.get_all_content_sequences()
            print("origionalcleaned",index_name, q_file_lines0[:2])
            remover_class_r = StyleMarkerRemover(r_file,word_list_file_styletrf)
            r_file_lines0 = remover_class_r.get_all_content_sequences()
            print("trfcleaned",index_name, r_file_lines0[:2])
            len_q = len(q_file_lines0)
            len_r = len(r_file_lines0)
            if len_q<len_r:
                r_file_lines = r_file_lines0[:len_q]
                q_file_lines = q_file_lines0[:]
                print(5555,len_q,len_r)
            elif len_r<len_q:
                q_file_lines = q_file_lines0[:len_r]
                r_file_lines =r_file_lines0[:]
                print(555500000,len_q,len_r)
            else:
                r_file_lines = r_file_lines0[:]
                q_file_lines = q_file_lines0[:]
                print(5555000000555555,len_q,len_r)

            com_file_remove_style_markers(q_file_lines, r_file_lines, w_file, word_dict)
            res.append(np.mean(com_file_score_remove_style_markers(q_file_lines, r_file_lines, word_dict)))
        else:

            com_file(q_file, r_file, w_file, word_dict)
            res.append(np.mean(com_file_score(q_file, r_file, word_dict)))    
    return res


def gen_score_random(test_dir_name_generated, test_dir_name_random,  suffix=''):
    emb = Embedding(100)
    # word_dict: a dict with the name as the key and word embeddings vectors as the value
    word_dict = emb.get_all_emb()
    subdir_names = ['multi_decoder', 'embedding', 'memory']
    # get_sub_dirnames(test_dir_name) = os.listdir(test_dir_name)
    '''
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
    res = []
    indexes = ["0", "1"]
    for i, index_name in enumerate(indexes):
        random_file = test_dir_name_random  + index_name # q_file
        # r_file = test_dir_name+"/"+dir_name+"/style"+index_name+".txt"
        generated_file = test_dir_name_generated +  index_name + suffix
        print(len(open(generated_file,'r').readlines()), len(open(random_file,'r').readlines()))
        
        word_list_file_styleorig = "./opinion-lexicon-English/words-cleaned." + index_name

        word_list_file_styletrf = "./opinion-lexicon-English/words-cleaned." + indexes[i - 1]
        # print(word_list_file_styleorig, word_list_file_styletrf)

        w_file = test_dir_name_generated + "style_random" + index_name + "_semantics.txt"
        if remove_style_markers:
            # print (11111)
            remover_class_generated = StyleMarkerRemover(generated_file, word_list_file_styletrf)
            generated_file_lines = remover_class_generated.get_all_content_sequences()
            print("trfcleaned", index_name, generated_file_lines[:2])
            remover_class_random = StyleMarkerRemover(random_file, word_list_file_styleorig)
            random_file_lines = remover_class_random.get_all_content_sequences()[:len(generated_file_lines)] # making random file equal in length compared to trf file
            #print(1111111111111111111111111,len(random_file_lines),len(generated_file_lines))

            print("origionalcleaned", index_name, random_file_lines[:2])

            com_file_remove_style_markers(random_file_lines, generated_file_lines, w_file, word_dict)
            res.append(np.mean(com_file_score_remove_style_markers(random_file_lines, generated_file_lines, word_dict)))
        else:
        	com_file(random_file, generated_file, w_file, word_dict)
        	res.append(np.mean(com_file_score(random_file, generated_file, word_dict)))
    return res



if __name__=="__main__":
    test_dir_name = sys.argv[1]
    #print('random_content_reservation', random_content_reservation(test_dir_name))
    if random_content_preservation:
        print("Comparing random files ...")
        res= gen_score_random(test_dir_name,sys.argv[2] ,'') # res is a list
        print('content preservation score baesd on comparing test_file_0 and a random file',res[0],'content preservation score baesd on comparing test_file_1 and a random file',res[1])

    else:
        print("Comparing original and genertaed files ...")
        res= gen_score(test_dir_name,sys.argv[2] ,'.rec') # res is a list
        print('content preservation score baesd on comparing test_file_0 and its generated file',res[0],'content preservation score baesd on comparing test_file_1 and its generated file',res[1])
    print('the average',  statistics.mean(res))

