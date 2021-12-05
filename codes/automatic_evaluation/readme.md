
****entropy.py is the code I ran for the paper, 
	For using this code:
		1.set the max_src_len to the max input data length that ur data has
		2. choose the files with style 1 or 0 in the normalizing_att_z_seqs method in line 186
		3. set the mode to 0 or 1 in line 185 which is the same style as previous step
		4. specify where to read the files from and where to write the files in the beginning of the file, we write a file here where ent of att_w of each generated sequence is written in. we saved the ent of att_w of each seq to be able to compute the confidence intervals
		5. run this in the terminal: entropy.py

