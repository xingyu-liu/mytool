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



def write_list2txt(txt_path, data, delimiter=None):
    with open(txt_path,'wt') as f:
        for item in data:
            f.write('{0}{1}'.format(item,delimiter))


def list_dir(dir_path,subdir_only=False):
    if subdir_only == True:
        fs = [f for f in glob.glob(dir_path) if not '.' in f]        
    else:
        fs = glob.glob(dir_path)
    
    fs.sort()
    return fs
    

def bids2fsfast(bids_root,out_dir):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)       
    
    os.chdir(bids_root)
    subidlist = list_dir('sub-*',subdir_only=True)

    for subid in subidlist:
        sub_dir = os.path.join(out_dir,subid)
        os.mkdir(sub_dir) 
        
        os.chdir(os.path.join(bids_root,subid))
        sesidlist = list_dir('ses-*',subdir_only=True)

        write_list2txt(os.path.join(sub_dir,'sessid'),sesidlist,delimiter='\n')

        for sesid in sesidlist:
            ses_dir = os.path.join(sub_dir,sesid)
            os.mkdir(ses_dir) 
            
            with open(os.path.join(ses_dir,'subjectname'),'w') as f:
                f.write('{0}'.format(subid))
            
            os.chdir(os.path.join(bids_root,subid,sesid,'func'))
            runidlist = list_dir('*.func.gii',subdir_only=False)
            
            for runid in runidlist:
                run_info = runid.split('_')
                key_name = [key.split('-')[0] for key in run_info]
                task_name = run_info[key_name.index('task')].split('-')[-1]
                run_num = run_info[key_name.index('run')].split('-')[-1]
                space_name = run_info[key_name.index('space')].split('-')[-1]
                hemi_name = run_info[key_name.index('hemi')].split('-')[-1][0]
                run_name = '{0}{1}'.format(task_name, run_num)
                
                run_dir = os.path.join(ses_dir,run_name)
                if not os.path.exists(run_dir):
                    os.mkdir(run_dir) 
                
                file_name = 'fmcpr.{0}.{1}h.gii'.format(space_name,
                                   hemi_name.lower())
                scr = os.path.join(bids_root,subid,sesid,'func', runid)
                dst = os.path.join(run_dir, file_name)
                
                os.symlink(scr,dst)
                
        print('{0} done'.format(subid))
    print('all done')
                
    