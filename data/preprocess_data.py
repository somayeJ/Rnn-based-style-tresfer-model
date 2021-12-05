# lines = [pre_process(line) for line in lines0 if min_seq_len<len(line.split())<max_seq_len] but for the code for MSD dataset I changed this to lines = [pre_process(line) for line in lines0 if min_seq_len<len(pre_process(line).split())<max_seq_len]
# This code was written for prerocessing the paper_news files & 
import random 
import re

# here paper is  style 0 & news is style 1, more details are in read me file,
def pre_process(line):
	tokens = []
	for token in line.strip().lower().split():
		my_new_token = re.sub('\d*[a-zA-Z\$\%\.\'\/]*\d+[a-zA-Z\$\%\.\d\'\/]*', "_num_", token) 
		print("re found number related expressions",my_new_token)
		'''
		if token.isdigit():
			tokens.append("_num_")

		else:
			tokens.append(token)
		'''
		tokens.append(my_new_token)
	result = ' '.join([i for i in tokens])
	return result

def write_processed_files(ori_file_name_read, file_name_write,min_seq_len, max_seq_len):
	'''
	ori_file_name: path to the ori_file that we want to read from
	file_name_write: path to the file that we want to write in
	'''
	with open(ori_file_name_read,) as f,  open(file_name_write, 'w') as fw:
		lines0=f.readlines()
		lines = [pre_process(line) for line in lines0 if min_seq_len<len(line.split())<max_seq_len]

		#lines2 = [line.strip().lower() for line in lines1]
		print(lines[:2])
		print(lines0[:2])
		print(len(lines0), len(lines))
		#random.shuffle(lines)
		print(lines[:2])
		print(lines0[:2])		
		fw.writelines("%s\n" % line.strip() for line in lines)

	return




write_processed_files('./amazon/amazon_preprocessed/original_split/binary_original_split_data/sentiment.test.1', './amazon/amazon_preprocessed/original_split/original_split_preprocessed/sentiment.test.1',1,26)
