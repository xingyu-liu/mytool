#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate FSFAST data structure with given BIDS format preprocessed surface data
Author: Xingyu_Liu @ BNU
This is a temporary script file.
"""
import os
import glob
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(
            description=('generate FSFAST file structure with '
                         'given BIDS format preprocessed surface dataset'
                         'optional arguments help to generate files for'
                         'first level analysis'))
    parser.add_argument('bids_root',
                        type=str,
                        help=('the root folder of BIDS valid surface dataset '
                              '(sub-XXXXX folders should be found '
                              'at the top level in this folder)'))
    parser.add_argument('out_dir',
                        type=str,
                        help='the output path for FSFAST file structure')
    parser.add_argument('--fsrecon_dir',
                        default=None,
                        type=str,
                        help='the path of freesurfer recon folder')
    parser.add_argument('--fwhm',
                        default='0',
                        help='desired smoothing fwhm')
    parser.add_argument('--tr',
                        default=None,
                        help='TR of bold scan in ms')

    args = parser.parse_args()
    print(args)
    
    # core func
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
    
    def gii2nii_cmd(sspace,tspace,sval,tval,fwhm,hemi):
        cmd = ('mri_surf2surf '
               '--srcsubject {0} '
               '--trgsubject {1} '
               '--sval {2} '
               '--tval {3} '
               '--fwhm-trg {4} '
               '--hemi {5}'.format(sspace,tspace,sval,tval,fwhm,hemi))
        return cmd
        
    def meanval_cmd(vol_fpath,mask_fpath,meanval_path,waveform_path):
        cmd = ('meanval --i {0} --m {1} --o {2} --avgwf {3}'.format(
        vol_fpath,mask_fpath,meanval_path,waveform_path))
        
        return cmd
        
    
    def bids2fsfast(bids_root,out_dir,fsrecon_dir=None,fwhm=None,tr=None):
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)       
        
        os.chdir(bids_root)
        subidlist = list_dir('sub-*',subdir_only=True)
    
        for subid in subidlist:
            sub_dir = os.path.join(out_dir,subid)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir) 
            
            os.chdir(os.path.join(bids_root,subid))
            sesidlist = list_dir('ses-*',subdir_only=True)
    
            write_list2txt(os.path.join(sub_dir,'sessid'),sesidlist,
                           delimiter='\n')
    
            for sesid in sesidlist:
                ses_dir = os.path.join(sub_dir,sesid)
                if not os.path.exists(ses_dir):
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
                    space_name = run_info[
                            key_name.index('space')].split('-')[-1]
                    hemi_name = run_info[
                            key_name.index('hemi')].split('-')[-1][0].lower()
                    run_name = '{0}{1}'.format(task_name, run_num)
                    
                    run_dir = os.path.join(ses_dir,run_name)
                    if not os.path.exists(run_dir):
                        os.mkdir(run_dir) 
                    
                    file_name = '{0}.{1}h.gii'.format(space_name,hemi_name)
                    
                    scr = os.path.join(bids_root,subid,sesid,'func', runid)
                    dst = os.path.join(run_dir, file_name)
                    
                    if not os.path.exists(dst):
                        os.symlink(scr,dst)
                    
                    # shell command
                    # creat meanval and waveform file
                    mask_fname = '{0}space-T1w_desc-brain_mask.nii.gz'.format(
                            runid.split('space')[0])
                    mask_fpath = os.path.join(bids_root,subid,sesid,'func',
                                              mask_fname)
                    vol_fname = '{0}space-T1w_desc-preproc_bold.nii.gz'.format(
                            runid.split('space')[0])
                    vol_fpath = os.path.join(bids_root,subid,sesid,'func',
                                             vol_fname)                    
                    meanval_path = os.path.join(run_dir,'global.meanval.dat')
                    waveform_path = os.path.join(run_dir,'global.waveform.dat')
                    
                    cmd = meanval_cmd(vol_fpath, mask_fpath,
                                      meanval_path,waveform_path)
                    subprocess.call(cmd,shell=True)
                    
                    # gii 2 nii.gz 
                    if (fsrecon_dir is not None) and (tr is not None):
                        
                        cmd = 'export SUBJECTS_DIR={0}'.format(fsrecon_dir)
                        subprocess.call(cmd,shell=True)
                                               
                        if space_name == 'fsnative':
                            nii_fname = 'fmcpr.sm{0}.{1}h.nii.gz'.format(
                                    fwhm,hemi_name)
                            nii_fpath = os.path.join(run_dir,nii_fname)
                            cmd = gii2nii_cmd(subid,subid,dst,nii_fpath,
                                              fwhm,'{0}h'.format(hemi_name))
                        else:
                            nii_fname = 'fmcpr.sm{0}.{1}.{2}h.nii.gz'.format(
                                    fwhm,space_name,hemi_name)
                            nii_fpath = os.path.join(run_dir,nii_fname)
                            cmd = gii2nii_cmd(space_name,space_name,dst,
                                              nii_fpath,fwhm,
                                              '{0}h'.format(hemi_name))               
                        subprocess.call(cmd,shell=True)
                        
                        # add tr info
                        cmd = 'mri_convert {0} {0} -tr {1}'.format(
                                nii_fpath,tr)
                        subprocess.call(cmd,shell=True)
                        
                    else:
                        print('both fsrecon_dir and tr should be provided'
                              'in order to convert gii files to nifti files')
                                          
            print('{0} done'.format(subid))
        print('all done')
        
        
    # creat FSFAST data structure
    bids2fsfast(args.bids_root,args.out_dir,args.fsrecon_dir,args.fwhm,args.tr)
                
if __name__ == '__main__':
    main()