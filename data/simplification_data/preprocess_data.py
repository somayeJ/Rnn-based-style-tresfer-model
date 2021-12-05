# lines = [pre_process(line) for line in lines0 if min_seq_len<len(line.split())<max_seq_len] but for the code for MSD dataset I changed this to lines = [pre_process(line) for line in lines0 if min_seq_len<len(pre_process(line).split())<max_seq_len]
# the code for the first split
'''
with open('./paper.txt') as f0,  open('sentiment.dev.0', 'w') as dev0, open('sentiment.test.0', 'w') as test0, open('sentiment.train.0', 'w') as train0:
	lines=f0.readlines()
	dev0.writelines("%s\n" % line.strip() for line in lines[:5376])
	test0.writelines("%s\n" % line.strip() for line in lines[5376:5376+5376])
	train0.writelines("%s\n" % line.strip() for line in lines[5376+5376:])


with open('./news.txt') as f1,  open('sentiment.dev.1', 'w') as dev1, open('sentiment.test.1', 'w') as test1, open('sentiment.train.1', 'w') as train1:
	lines=f1.readlines()
	dev1.writelines("%s\n" % line.strip() for line in lines[:5425])
	test1.writelines("%s\n" % line.strip() for line in lines[5425:5425+5425])
	train1.writelines("%s\n" % line.strip() for line in lines[5425+5425:])
'''
import random 
import re

# here paper is  style 0 & news is style 1, more details are in read me file,
def pre_process(line):
	tokens = []
	for token in line.strip().lower().split():
		my_new_token0 = re.sub('\d*[a-zA-Z\$\%\.\'\/]*\d+[a-zA-Z\$\%\.\d\'\/]*', "_num_", token) 
		my_new_token = re.sub('[?.!]+$', "", my_new_token0) 
		'''
		if token.isdigit():
			tokens.append("_num_")

		else:
			tokens.append(token)
		'''
		tokens.append(my_new_token)
	result = ' '.join([i for i in tokens])
	return result

def write_processed_files(ori_file_name, process_file_name,min_seq_len, max_seq_len):
	'''
	ori_file_name : path tp the ori_file that we want to read from and do the processing and partitioning
	file_dev_name & file_test_name & file_train_name: the path to the file where u want to write each of these partions after preprocessing
	n_dev& n_test: size of dev and test dataset
	# max_seq_len: seq_len that shows seqs >= this no should be eliminated
	'''
	with open(ori_file_name) as f, open(process_file_name,'w') as p:
		lines0=f.readlines()
		lines = [pre_process(line) for line in lines0 if min_seq_len<len(line.split())<max_seq_len]

		#lines2 = [line.strip().lower() for line in lines1]
		print(lines[:2])
		print(lines0[:2])
		print(len(lines0), len(lines))
		random.shuffle(lines)
		print(lines[:2])
		print(lines0[:2])

		p.writelines("%s\n" % line.strip() for line in lines)

	return

def write_split_files( process_file_name,file_dev_name,file_test_name,file_train_name,n_dev, n_test):
	'''
	ori_file_name : path tp the ori_file that we want to read from and do the processing and partitioning
	file_dev_name & file_test_name & file_train_name: the path to the file where u want to write each of these partions after preprocessing
	n_dev& n_test: size of dev and test dataset
	# max_seq_len: seq_len that shows seqs >= this no should be eliminated
	'''
	with  open(process_file_name) as p,  open(file_dev_name, 'w') as dev, open(file_test_name, 'w') as test, open(file_train_name, 'w') as train:
		lines=p.readlines()

		#lines2 = [line.strip().lower() for line in lines1]
		print(lines[:2])
		
		print( len(lines))
		random.shuffle(lines)
		print(lines[:2])

		dev.writelines("%s\n" % line.strip() for line in lines[:n_dev])
		test.writelines("%s\n" % line.strip() for line in lines[n_dev:n_dev+n_test])
		train.writelines("%s\n" % line.strip() for line in lines[n_dev+n_test:])
	return

process=False
if process:
	write_processed_files('./orig_data/normal.txt','./data_processed/data.0',3,34)
else:

	write_split_files('./data_processed/data.1', './data_split/sentiment.dev.1','./data_split/sentiment.test.1','./data_split/sentiment.train.1',7930,7930)
	# normal 0 longer, 3,34
	# simple 1: shorter, 3,16
