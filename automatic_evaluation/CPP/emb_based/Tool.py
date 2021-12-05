#encoding=utf-8

import os

def get_sub_dirnames(dir_name):
    # os.listdir: This method returns a list containing the names of the entries in the directory given by path.
    return os.listdir(dir_name)

if __name__=="__main__":
    res = get_sub_dirnames("test")
    print [i for i in res if not i.endswith("txt")]



