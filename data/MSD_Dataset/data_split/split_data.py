# this file splits a file into three sets of train , dev and test
from random import shuffle 

file_read= '../preprocessed_data/data_preprocess.1'
file_write_dev = 'sentiment.dev.1'
file_write_test = 'sentiment.test.1'
file_write_train = 'sentiment.train.1'
len_dev= 5085
len_test= 5085

'''

file_read= '../preprocessed_data/data_preprocess.1'
file_write_dev = 'sentiment.dev.1'
file_write_test = 'sentiment.test.1'
file_write_train = 'sentiment.train.1'
len_dev= 5616
len_test= 5616

'''

def split_data(file_read,len_dev,len_test, file_write_dev,file_write_test,file_write_train):
	with open(file_read,'r')  as r1, open(file_write_dev,"w") as w_dev, open(file_write_test,"w") as w_test, open(file_write_train,"w") as w_train:
		data = r1.readlines()
		shuffle(data)
		print(len(data))
		for line in data[:len_dev]: 
			w_dev.write(line.strip())
			w_dev.write('\n')

		for line in data[len_dev:len_test+len_dev]: 
			w_test.write(line.strip())
			w_test.write('\n')

		for line in data[len_test+len_dev:]: 
			w_train.write(line.strip())
			w_train.write('\n')
		  
split_data(file_read,len_dev,len_test, file_write_dev,file_write_test,file_write_train)
