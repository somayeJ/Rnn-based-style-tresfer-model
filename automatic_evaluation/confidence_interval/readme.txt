Run the code in python3
Run this in terminal: python main.py file1 file2 , where files 1 and 2 are : 

file1: shows the cosine sim scores or any other score computed by comparing style_shifted sequences and orig seqs of the sequences of style 1

file1: shows the cosine sim scores or any other score computed by comparing style_shifted sequences and orig seqs of the sequences of style 2

 these file with the cosine sim scores are saved in the corresponding output folder of each model in tmp folder, in  style0_semantics.txt && style1_semantics.txt files
confidence score can be modified either in main.py or ci.py in mean_confidence_interval method, 

Run as an example this in the terminal:
python main.py  ../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/yelp_corpus/adv_z_init/style0_semantics.txt  ../../../tmp/Deterministic_CrossAligned_models/single_encoder_single_decoder\(SDS\)/yelp_corpus/adv_z_init/style0_semantics.txt


main.py: for the files in which one number is saved which is the score 


main_wmd.py: for the files in which two numbers are saved where the 2nd one is the score 



