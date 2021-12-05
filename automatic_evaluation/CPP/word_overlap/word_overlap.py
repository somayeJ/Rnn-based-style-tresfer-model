import sys
import argparse
import logging
import numpy as np
import statistics
import tensorflow as tf
from scipy.spatial.distance import cosine

#from linguistic_style_transfer_model.config import global_config
#from linguistic_style_transfer_model.utils import log_initializer, lexicon_helper

from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords

#from linguistic_style_transfer_model.config import global_config

remove_style_markers = False
sentiment_words_file_path_positive = "./opinion-lexicon-English/positive-words-cleaned.txt"
sentiment_words_file_path_negative = "./opinion-lexicon-English/negative-words-cleaned.txt"

def get_sentiment_words(file_name):
    with open(file_name,mode='r') as sentiment_words_file:
        words = sentiment_words_file.readlines()
    words = set(word.strip() for word in words)

    return words


def get_stopwords():
    nltk_stopwords = set(stopwords.words('english'))
    sklearn_stopwords = stop_words.ENGLISH_STOP_WORDS

    all_stopwords = set()
    all_stopwords |= spacy_stopwords
    all_stopwords |= nltk_stopwords
    all_stopwords |= sklearn_stopwords

    return all_stopwords


def word_overlap_score_evaluator(source_file_path, target_file_path, source_style, target_style):
    
    actual_word_lists, generated_word_lists = list(), list()
    with open(source_file_path) as source_file, open(target_file_path) as target_file:
        #assert len(source_file.readlines())==len(target_file.readlines()), "length error"
        for line_1, line_2 in zip(source_file, target_file):
            actual_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_1))
            generated_word_lists.append(tf.keras.preprocessing.text.text_to_word_sequence(line_2))

    english_stopwords = get_stopwords()
    #english_stopwords = set([]) # to test how keeping the stopwords affects the results
    sentiment_words_positive = get_sentiment_words(sentiment_words_file_path_positive)
    sentiment_words_negative = get_sentiment_words(sentiment_words_file_path_negative)
    #sentiment_words_positive = set([])
    #sentiment_words_negative = set([])

    sentiment_words_total = sentiment_words_negative | sentiment_words_positive
    #print('actual_word_lists',actual_word_lists)
    #print('generated_word_lists',generated_word_lists)
    scores = list()
    for word_list_1, word_list_2 in zip(actual_word_lists, generated_word_lists):
        score = 0
        words_1 = set(word_list_1)
        words_2 = set(word_list_2)
        #print('words_1 before removal',words_1)
        #print('words_2 before removal',words_2)

        if remove_style_markers :

            if source_style == 'positive':
                words_1 -= sentiment_words_positive
                #print('removing source_style == 1111111111111', words_1)
            elif source_style == 'negative':
                words_1 -= sentiment_words_negative
                #print('removing source_style == 000000000000', words_1)
            else: # putting the styles to '' and '' in args, in case we want to remove both styles from both seqs
                words_1 -= sentiment_words_negative
                words_1 -= sentiment_words_positive
                #print('removing both 000000000000 & 11111111111 from source seq', words_1)
            
            #print('word1 after stopwords removal', words_1)
            if target_style == 'positive':
                words_2 -= sentiment_words_positive
                #print('target_style == 1111111111', words_2)

            elif target_style == 'negative':
                words_2 -= sentiment_words_negative
                #print('target_style == 00000000', words_2)
            else: # putting the styles to '' and '' in args, in case we want to remove both styles from both seqs
                words_2 -= sentiment_words_negative
                words_2 -= sentiment_words_positive
                #print('removing both 000000000000 & 11111111111 from target seq', words_2)    
        
        words_2 -= english_stopwords
        words_1 -= english_stopwords
        #print('word2 after stopwords removal', words_2)

        word_intersection = words_1 & words_2
        word_union = words_1 | words_2
        #print('word_intersection',word_intersection, len(word_intersection))
        #print('word_union', word_union, len(word_union))
        if word_union:
            score = float(len(word_intersection)) / len(word_union)
            scores.append(score)
            #print(score, '--------------------------------------------------------------')

    word_overlap_score = statistics.mean(scores) if scores else 0
    #print(scores)

    del english_stopwords
    del sentiment_words_positive
    del sentiment_words_negative

    return word_overlap_score

def main(argv):
    #print(argv)
    word_overlap_score = word_overlap_score_evaluator(argv[0],argv[1],argv[2],argv[3]) # source_file_path, target_file_path, source_style, target_style
    

    #logger.info("Aggregate content preservation: {}".format(content_preservation_score))
    #logger.info("Aggregate word overlap: {}".format(word_overlap_score))
    print('word_overlap_score', word_overlap_score)


if __name__ == "__main__":
    main(sys.argv[1:])

  