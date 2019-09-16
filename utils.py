# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import glob


def readtxt(txt_path, delimiter=None, exclude_first_line=0):
    x = []
    with open(txt_path, 'r') as f:
        for line in f.readlines()[exclude_first_line:]:
            if delimiter:
                split_line = line.strip('\n').split(delimiter)
                x.append(split_line)
            else:
                x.append(line)

    return x


def write_list2txt(txt_path, data, delimiter=','):
    with open(txt_path, 'wt') as f:
        for item in data:
            for i, content in enumerate(item):
                if i == len(item)-1:
                    f.write(content)
                else:
                    f.write('{0}{1}'.format(content, delimiter))
            f.write('\n')


def list_dir(dir_path=None, subdir_only=False):
    if dir_path is not None:
        if subdir_only is True:
            fs = [f for f in glob.glob(dir_path) if '.' not in f]
        else:
            fs = glob.glob(dir_path)
    else:
        if subdir_only is True:
            fs = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        else:
            fs = [f.name for f in os.scandir(dir_path)]

    fs.sort()
    return fs
