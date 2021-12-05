# me: classifier from page  https://github.com/shentianxiao/language-style-transfer/blob/master/code/classifier.py, but modified for the probing task
import os
import sys
import time
#import ipdb
import random
import numpy as np
import tensorflow as tf
from options import load_arguments
from file_io import load_sent
#from nn import cnn
from nn_tf_v_1 import cnn, feed_forward
from sklearn.metrics import confusion_matrix
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Model(object):

    def __init__(self, args):
        #dim_emb = args.dim_emb
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        n_filters = args.n_filters
        dim_emb = 500

        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.x = tf.placeholder(tf.float32, [None, dim_emb],    #batch_size * max_len
            name='x')
        self.y = tf.placeholder(tf.float32, [None],
            name='y')

        x= self.x #bad results
        #print(x)
        #x = tf.expand_dims(self.x, 2) # not compatible size
        #x = tf.expand_dims(self.x, 1)# bad results 
        print(x)

        #self.logits = cnn(x, filter_sizes, n_filters, self.dropout, 'cnn')
        self.logits = feed_forward(x, 'feed_forward')

        self.probs = tf.sigmoid(self.logits)


        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate) \
            .minimize(self.loss)

        self.saver = tf.train.Saver()

def create_model(sess, args):
    model = Model(args)
    if args.load_model:
        print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    return model

def evaluate(sess, args,  model, x, y):
    probs = []
    batches = get_batches(x, y,  args.batch_size)
    for batch in batches:
        p = sess.run(model.probs,
            feed_dict={model.x: batch['x'],
                       model.dropout: 1})
        probs += p.tolist()
        
    print(probs)    
    y_hat = [p > 0.5 for p in probs]
    print(y_hat)

    same = [p == q for p, q in zip(y, y_hat)]
    return 100.0 * sum(same) / len(y), probs

def get_batches(x, y,  batch_size, min_len=5):
    batches = []
    s = 0
    while s < len(x):
        t = min(s + batch_size, len(x))
        _x = []
        for sent_em in x[s:t]:
            _x.append(sent_em)

        batches.append({'x': _x,
                        'y': y[s:t]})
        s = t
    return batches

def read_elmo_file(file):
    x = np.loadtxt(file)
    x_list = x.tolist()
    return x

def prepare(args,path, suffix=''): 
    
    if args.use_elmp_reps: # sentiment.dev.elmo.0
        data0= read_elmo_file(path +".elmo"+".0")
        data1= read_elmo_file(path +".elmo"+".1")
    else: # sentiment.dev.0.npy
        data0 = np.load(path+'.0'+'.npy',allow_pickle=False ) 
        data1 = np.load(path+'.1'+'.npy',allow_pickle=False ) 

    print('data0.shape()',data0.shape)
    print('data1.shape()',data1.shape)

    #x = data0 + data1
    x = np.concatenate((data0, data1), axis=0)
    print(path, type(x))
    print( len(x))
    print('data0+data1',x.shape)
    y = [0] * len(data0) + [1] * len(data1)
    print('y[:3], y[-3:]',len(y),y[:3], y[-3:])
    #z = sorted(zip(x, y), key=lambda i: len(i[0]))
    #return zip(*z)
    return x,y

if __name__ == '__main__':
    args = load_arguments()

    if args.train:
        train_x, train_y = prepare(args,args.train)
        print('len(train_x[0]),len(train_y)',len(train_x),len(train_x[0]),len(train_y))
        print(train_y[:3], train_y[-3:])

    
    if args.dev:
        dev_x, dev_y = prepare(args,args.dev)
        print('len(dev_x),len(dev_x),len(dev_x[0]),len(dev_y)',len(dev_x),len(dev_x),len(dev_x[0]),len(dev_y))

    if args.test:
        test_x, test_y = prepare(args,args.test)
        print('len(test_x),len(test_x[0]),len(test_y)',len(test_x),len(test_x[0]),len(test_y))


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args)
        if args.train:
            batches = get_batches(train_x, train_y,
                args.batch_size)
            print("len(batches[0]['x']),len(batches[0]['x'][0]),len(batches[0]['y'])",len(batches[0]['x']),len(batches[0]['x'][0]),len(batches[0]['y']))
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            loss = 0.0
            best_dev = float('-inf')
            learning_rate = args.learning_rate

            for epoch in range(1, 1+args.max_epochs):
                print '--------------------epoch %d--------------------' % epoch

                for batch in batches:
                    step_loss, _ = sess.run([model.loss, model.optimizer],
                        feed_dict={model.x: batch['x'],
                                   model.y: batch['y'],
                                   model.dropout: args.dropout_keep_prob,
                                   model.learning_rate: learning_rate})

                    step += 1
                    loss += step_loss / args.steps_per_checkpoint

                    if step % args.steps_per_checkpoint == 0:
                        print 'step %d, time %.0fs, loss %.2f' \
                            % (step, time.time() - start_time, loss)
                        loss = 0.0

                if args.dev:
                    acc, _ = evaluate(sess, args,  model, dev_x, dev_y)
                    print 'dev accuracy %.2f' % acc
                    if acc > best_dev:
                        best_dev = acc
                        print 'Saving model...'
                        model.saver.save(sess, args.model)

        if args.test:
            acc, _ = evaluate(sess, args, model, test_x, test_y)
            print 'test accuracy %.2f' % acc
