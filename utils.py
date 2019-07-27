# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

def readtxt2array(txt_path, delimiter, exclude_first_line):
    x = []
    with open(txt_path, 'r') as f:
        for line in f.readlines()[exclude_first_line:]:
            if delimiter:
                split_line = line.split(delimiter)[:-1]
                x.append(split_line)
            else:
                x.append(line)                        
    x = np.asarray(x)   
    
    return x