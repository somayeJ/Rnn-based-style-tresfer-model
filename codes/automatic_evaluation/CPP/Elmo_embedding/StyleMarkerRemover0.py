#encoding=utf-8
"""
author: Somaye
in this class:
style markers of the input files will be removed based style dictionary of the page https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/tree/master/data/opinion-lexicon-English
which is used in paper "text style transfer in text exploration and evaluation"
"""
import numpy as np
class StyleMarkerRemover():
    def __init__(self, file_name,  word_dictionary):

        self.content_corpus = self.remove_style_markers(file_name, word_dictionary)
    
    def remove_style_markers(self, file_name, word_dictionary):
        f0 = open(file_name,  "r")
        f1 = open(word_dictionary, "r")
        lines = f0.readlines()
        print(lines[:2])
        word_list = [line.strip() for line in f1.readlines()]
        print(word_list[:2])
        f0.close()
        f1.close()
        content_lines =[] # lines after removing style markers
        for line in lines:
            line_split = line.strip().split()
            line_cleaned =[word for word in line_split if word not in word_list]
            if len(line_cleaned) == 0: 
                content_lines.append('the')
            else:
                content_lines.append(' '.join(line_cleaned))


        return content_lines
    
    def get_all_content_sequences(self):
        return self.content_corpus

class StyleMarkerRemover_elmo():
    def __init__(self, file,  word_dictionary):

        self.kept_words_indices = self.remove_style_markers(file, word_dictionary)
    
    def remove_style_markers(self, file, word_dictionary):
        f0 = open(file_name,  "r")
        f1 = open(word_dictionary, "r")
        lines = f0.readlines()
        print(lines[:2])
        word_list = [line.strip() for line in f1.readlines()]
        print(word_list[:2])
        f0.close()
        f1.close()
        content_lines = [] # lines after removing style markers
        kept_words_indices = [] # the indicies of the words which are kept
        kept_words_elmo = []
        for line in lines:
            line_split = line.strip().split()
            line_cleaned = [word for word in line_split if word not in word_list]
            kept_w_indices = [i for i,word in enumerate(line_split) if word not in word_list]
            if len(line_cleaned) == 0: 
                content_lines.append('the')
                kept_words_indices.append(kept_w_indices[:1])
            else:
                content_lines.append(' '.join(line_cleaned))
                kept_words_indices.append(kept_w_indices)

        return kept_words_indices
    
    def get_all_content_sequences(self):
        return  self.kept_words_indices




