# this code is to create the embeddings as the last state of the encoder, and takes train, dev and test data and with two styles saves them into files with their numpy format
# it can be run in the terminal by running a code like the following:
# python ...z_create.py   --load_model true --model ../../tmp/.../model  --vocab ../../tmp/.../yelp.vocab  --output ../probing_classification/output_emb/adv_MD/sentiment.train  --train ../../data/yelp/sentiment.train                                    
# --train or --dev or -- test shows the directory of the files we want to produce their embedding  representation, --output shows where want to store the files with embedding representations
# --model and --vocab: shows the address of the directory in which  model and the vocab file is stored from which we want to compute the emb_rep of the sequences 
# for different models, we need to replace the model part of that model in this file which is  from line 183 up and keep the rest

def seq_embeddings(model, sess, args, vocab, data0,data1):
    '''
    :param model:
    :param sess:
    :param args:
    :param vocab:
    :param data0:
    :param data1:
    :return:  the embedding vectors lists (last state of encoder) corresponding to the sentences in the data
    '''
    batches, order0, order1 = get_batches(data0,data1,
        vocab.word2id, args.batch_size, args)
    
    embed_data0 = []
    embed_data1 = []
    for batch in batches:
        embed_data =[]
        print("batch['size'], len(batch['enc_inputs']),len(batch['enc_inputs'])/2",batch['size'], len(batch['enc_inputs']),len(batch['enc_inputs'])/2)
        half = len(batch['enc_inputs'])/2
        print("half,len(batches), len(batch['enc_inputs'])",half,len(batches), len(batch['enc_inputs']))
        print("batch['enc_inputs'],batch['enc_inputs'][0]",batch['enc_inputs'][:10],len(batch['enc_inputs'][0]))
        sen0 =[]
        sen1 =[]
        z=sess.run( [model.z],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.enc_inputs: batch['enc_inputs'],
                       model.dec_inputs: batch['dec_inputs'],
                       model.labels: batch['labels']})
        print(len(z),len(z[0]),type(z), len(z[0][0])) # z_shape : 1*128*500
        print("len(z[0][:half], len(z[0][:half]",len(z[0][:half]),len(z[0][0])) # 64*500
        embed_data0.extend(z[0][:half])
        embed_data1.extend(z[0][half:])
        sen0.extend(batch['text_inputs'][:half])
        sen1.extend(batch['text_inputs'][half:])
    print(sen0)
    print(sen1)
    print("len(embed_data1),len(embed_data0), len(sen0), len(sen1) after loop",len(embed_data1),len(embed_data0), len(sen0), len(sen1) )
    return embed_data0, embed_data1

def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    
    return model

if __name__ == '__main__':
    #tf.reset_default_graph() # in tu code asli nist
    args = load_arguments()

    #####   data preparation   #####
    if args.train:
        data0 = load_sent(args.train + '.0', args.max_train_size)
        data1 = load_sent(args.train + '.1', args.max_train_size)
        print '#sents of training file 0:', len(data0)
        print '#sents of training file 1:', len(data1)

        if not os.path.isfile(args.vocab):
            build_vocab(data0 + data1, args.vocab)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print 'vocabulary size:', vocab.size

    if args.dev:
        data0 = load_sent(args.dev + '.0')
        data1 = load_sent(args.dev + '.1')

    if args.test:
        data0 = load_sent(args.test + '.0')
        data1 = load_sent(args.test + '.1')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)


        # def seq_embeddings(model, decoder, sess, args, vocab, data, out_path):
        z_seqs0, z_seqs1 = seq_embeddings(model, sess, args, vocab,
                                data0, data1)
        np.save(args.output+'.0'+'.npy',z_seqs0)
        np.save(args.output+'.1'+'.npy',z_seqs1)

        vectors0=np.load(args.output+'.0'+'.npy',allow_pickle=False )
        vectors1=np.load(args.output+'.1'+'.npy',allow_pickle=False )

        print('length of written files 0 and their seq embedding size',len(vectors0), len(vectors0[0]))
        print('length of written files 1 and their seq embedding size',len(vectors1), len(vectors1[0]))



