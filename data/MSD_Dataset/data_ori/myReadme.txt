******* load_data.py : use python3 to run, it reads a file with json files which we give the path to it in line 4 with 'r' mode & 2 files for wirting with modes 'w' that we xrite the data with two labels in it
******* paper: https://arxiv.org/pdf/2005.00701.pdf
Test.txt: there are parallel seqs, notice that the labels in test set are repeats of 0 and 1, so the expert sentence (label 0) and the next layman sentence (label 1) are parallel, e.g., 1st and 2ed are a parallel sentence group.
******* expert sentence (label 0) and the next layman sentence (label 1)
******* sentiment.train.ori.0 or 1 consists of the original data before splitting into train and dev (sentiment.train.style and sentiment.dev.style)
****dist_ori: test.txt 1350 (675:0, 675:1) && train.txt: 245023(130349:0, 114674:1)



********* merging the files:
1. the files are generated using merge_files.py

2. then .. are substitutted by '', ... was not available

3. the resulting files are the followings:

data_ori.0:  sentiment.train_ori.0 + sentiment.test.0
data_ori.1:  sentiment.train_ori.1 + sentiment.test.1




