#encoding=utf-8
# this file is a bit modified gor gensim embeddings
"""
author: zhenxinfu

"""
import numpy as np

class Embedding():
    def __init__(self, dim):
        dim_all = [50, 100, 200, 300]
        assert dim in dim_all, "dim wrong"
        self.emb = self.read_emb(dim)
    
    def read_emb(self, dim):
        
        #file_name = "../word_emb/glove.6B."+str(dim)+"d.txt"
        file_name = "../word_emb/word_vectors_Gensim_train_yelp_initialized_randomly/vectors.kv"
        

        try:
            f = open(file_name, "r")
        except Exception, e:
            assert False, "fail to read file "+file_name
        lines = f.readlines()
        f.close()
        from gensim.models import KeyedVectors
        emb_all = KeyedVectors.load(file_name, mmap='r')
        '''
        emb_all = dict()
        for line in lines:
            line_split = line.split()
            line_name = line_split[0]
            line_emb = line_split[1:]
            try:
                line_emb = map(lambda x: float(x), line_emb)
            except ValueError:
                print ('Line  is corrupt!',line)
                break
            line_emb = map(lambda x: float(x), line_emb)
            line_emb = np.array(line_emb)
            emb_all[line_name] = line_emb 
        '''
        return emb_all
    
    def get_all_emb(self):
        return self.emb
    

if __name__=="__main__":
    emb = Embedding(300)
    x = emb.get_all_emb()
    print (x["good"])
    print(x)



