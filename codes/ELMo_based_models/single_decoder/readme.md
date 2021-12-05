The file to generate elmo-embeddings is in ../../automatic_evaluation/content_preservation/Elmo_embedding/Elmo_token_rep_write.py


There are two models in this folder, as following:

	1. AE_elmoZ_train_decoder_tf_1.py:: this is made based on AE_elmoZ_trained_model.py, in which the changes in the model are made to investigate the performance of the decoder after feeding the elmo embeddings, in this code, we want to train the decoder feeding it  with elmo rep of the yelp data


	2.adv_elmoZ_Single_dec_tf_v_1.py: This code is to to do the style transfer task by feeding the decoder by elmo rep of seqs and training the decoder by corresponding losses of the discriminator and decoder




The models and their corresponding output and vocab files are stored in the folder tmp/adv_Elmo_generator_emb/single_decoder/


For running this code , we need to do run the following in terminal:

python adv_elmoZ_Single_dec_tf_v_1.py  --test ../../../data/yelp/sentiment.test  --output ../../../tmp/adv_Elmo_generator_emb/single_decoder/adv/sentiment.test --vocab ../../../tmp/adv_Elmo_generator_emb/single_decoder/adv/yelp.vocab --model ../../../tmp/adv_Elmo_generator_emb/single_decoder/adv/model  --embedding ../../../word_emb/glove.6B.100d.txt  --elmo_seq_rep ../../../data/yelp_elmo_reps/yelp_elmo_rep_seqs/sentiment --load_model true --beam 8

	1. the argument --elmo_seq_rep ../../data/yelp_elmo_rep_seqs/sentiment  which shows the directory in which elmo_rep of yelp data is saved, (this argument is added to the argument shown in page "https://github.com/shentianxiao/language-style-transfer")

	2. To initialize with embedding files, we need to add the following argument: --embedding ../../word_emb/glove.6B.100d.txt



