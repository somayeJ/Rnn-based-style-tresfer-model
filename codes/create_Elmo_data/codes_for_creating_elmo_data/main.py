from __future__ import unicode_literals
import os
import spacy
from spacy.attrs import ORTH, LIKE_URL
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler, scale , MinMaxScaler
from Elmo_keras_write import create_elmo_embeddings,read_elmo_file, write_elmo_file, read_data
from allennlp.commands.elmo import ElmoEmbedder
import sys
import argparse
import pprint

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--input_dir',
            type=str,
            default='') # directory + file_name (without style), where data is read
    argparser.add_argument('--output_dir',
            type=str,
            default='') # directory + file_name (without style), where elmo_embeddings should be saved 
    argparser.add_argument('--style_1',
            type=str,
            default='.0')
    argparser.add_argument('--style_2',
            type=str,
            default='.1')

    args = argparser.parse_args()

    print ('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print ('------------------------------------------------')

    return args


if __name__ == '__main__':
    args = load_arguments()

    styles = []
    #elmo = ElmoEmbedder() #Set cuda_device to the ID of your GPU if you have one
    elmo = ElmoEmbedder(cuda_device=0) #Set cuda_device to the ID of your GPU if you have one

    if args.style_1:
        styles.append(args.style_1)
    if args.style_2:
        styles.append(args.style_2)

    if len(styles) == 0:
        directory = args.input_dir
        data = read_data(directory)
        embeddings, labels = create_elmo_embeddings(elmo, data)
        write_elmo_file(args.output_dir, embeddings)
        x = read_elmo_file (args.output_dir)
        print(len(x), len(x[0]))

    else:
        for s,style in enumerate(styles):
            directory = args.input_dir + style
            if s==0:
                data_0 = read_data(directory, s)
            elif s==1:
                data_1 = read_data(directory,s)
        embeddings_0, labels_0 = create_elmo_embeddings(elmo, data_0)
        embeddings_1, labels_1 = create_elmo_embeddings(elmo, data_1)
        print('type(embeddings_1',type(embeddings_1),type(embeddings_1[0]), type(embeddings_1[0][0]), type(embeddings_1[0][0][0]) )
        print('embeddings_1.shape',embeddings_1.shape)

        write_elmo_file(args.output_dir+'.0', embeddings_0)
        write_elmo_file(args.output_dir+'.1', embeddings_1)
    
        x = read_elmo_file (args.output_dir+'.0')
        print(len(x), len(x[0]))




