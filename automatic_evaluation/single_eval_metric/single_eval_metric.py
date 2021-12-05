# 
import os
import sys
import time
import numpy as np
import argparse
import pprint

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--content',
            type=float,
            default=0.0)
    argparser.add_argument('--style',
            type=float,
            default=0.0)
    argparser.add_argument('--fluency',
            type=float,
            default=0.0)
    argparser.add_argument('--w_c',
            type=float,
            default=1.0)
    argparser.add_argument('--w_f',
            type=float,
            default=1.0)
    argparser.add_argument('--w_s',
            type=float,
            default=1.0)
    argparser.add_argument('--smooth_param',
            type=float,
            default=0.0)

    args = argparser.parse_args()

    print ('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print ('------------------------------------------------')

    return args

def harmonic_mean( scores,weights):
    '''
    :param scores:
    :param weights:
    :return:
    '''
    h_mean = 0.0
    print('scores', scores)
    print('weights', weights)
    for (score, w) in zip(scores, weights):
        h_mean += w/score

    harmonic_mean = sum(weights) / h_mean
    
    return harmonic_mean

if __name__ == '__main__':
    args = load_arguments()
    scores = [args.style + args.smooth_param, args.fluency + args.smooth_param, args.content + args.smooth_param]
    weights = [args.w_s,args.w_f,args.w_c]
    print harmonic_mean(scores,weights)
