#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction analysis with CNN multivarient activation
Author: Xingyu Liu, Xiayu Chen, Taicheng Huang@ BNU
Reviewer:
"""

import argparse
import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from scipy.stats import pearsonr
from nipy.modalities.fmri.hemodynamic_models import compute_regressor
from dnnbrain.dnn import analyzer
from dnnbrain.dnn import io as dnn_io
from dnnbrain.brain import io as brain_io
from dnnbrain.dnn.models import dnn_truncate, TransferredNet, dnn_train_model


try:
    from sklearn import linear_model, model_selection, decomposition, svm
except ModuleNotFoundError:
    raise Exception('Please install sklearn in your workstation')


stim = ('/nfs/s2/userhome/liuxingyu/workingdir/dnn_face/dnn_activation/'
        'reconstruting_faces/stimuli_table/'
        'stimuli_table_reconface_sub-01_ses-01_faces001.csv')
net = 'alexnet'
layer = 'conv5'
axis = 'channel'
dnn_roi_filelist = '/nfs/s2/userhome/liuxingyu/Desktop/dnn_roi_filelist.csv'
pca = 10
unit_reduc = None
cvfold = 2
hrf = 'spm'
pred_model = 'glm'
response = ('/nfs/e3/natural_vision/ReconstructingFaceData/derivatives/'
            'fsfast/sub-01_ses-01/faces/001/fmcpr_sm5_rh.nii.gz')
#response = ('/nfs/s2/userhome/liuxingyu/Desktop/brain_roi_response.csv')
mask = ('/nfs/e3/natural_vision/ReconstructingFaceData/derivatives/'
        'fsfast/roi/face_roi_sm5_sub-01_rh_80.nii.gz')


#'spm','spm_time','spm_time_dispersion','canonical','canonical_derivative','fir'
#'spm: this is the hrf model used in spm'
#'spm_time: this is the spm model plus its time derivative (2 regressors)'
#'spm_time_dispersion: idem, plus dispersion derivative (3 regressors)'
#'canonical: this one corresponds to the Glover hrf'
#'canonical_derivative: the Glover hrf + time derivative (2 regressors)'
#'fir: finite impulse response basis, a set of delayed dirac models'

def db_mv_pred(axis=None,dnn_roi_filelist=None,mask=None,
               unit_reduction=None,pca=None,cvfold=2,pred_model,hrf=None,
               stim,net,layer,response,outdir,):    

    #%% Load response(y)
    def load_resp_csv(csv_path):        
        with open(csv_path,'r') as f:
            meta_data = [x.rstrip() for i, x in enumerate(f) if i<=2]
            resp_type = meta_data[0]
            tr = np.float(meta_data[1])
        resp_data = pd.read_csv(csv_path, skiprows=2)
        return resp_type, tr, list(resp_data.keys()), resp_data.values
    
    if response.endswith('csv'):
        assert mask is None, "Loading .csv response does not need a mask."
        # npic * ROIs
        resp_type,tr,roi_keys,resp = load_resp_csv(response)       
    elif response.endswith('nii') or response.endswith('nii.gz'):
        # Load brain images
        resp_raw, header = brain_io.load_brainimg(response)
        
        # get tr from nifti header
        if hrf is not None:
            assert header.get_xyzt_units()[-1] is not None, "TR was not provided in the brain imaging file header"
            if header.get_xyzt_units()[-1] in ['s','sec']:
                tr = header['pixdim'][4]
            elif header.get_xyzt_units()[-1] == 'ms':
                tr = header['pixdim'][4] / 1000
        
        # get masked resp data
        resp_raw_shape = np.shape(resp_raw)
        resp_raw = resp_raw.reshape(resp_raw_shape[0],-1)       
        if mask is not None:
            brain_mask, _ = brain_io.load_brainimg(mask, ismask=True)
            assert np.shape(brain_mask) == resp_raw_shape[1:], "Mask and brainimg should have the same geometry shape"
            brain_mask = brain_mask.reshape(-1)    
        else:
            brain_mask = np.zeros(resp_raw.shape[1])
            brain_mask[resp_raw.mean(0)!=0] = 1        
        resp = resp_raw[:,brain_mask!=0]
        
    else:
        raise Exception('Not support yet, please contact to the author for implementation.')

    brain_roi_size = resp.shape[1]    
    print('response data loaded')
    
    
    #%% Get CNN activation(x)
    netloader = dnn_io.NetLoader(net)
    imgcropsize = netloader.img_size     
    transform = transforms.Compose([transforms.Resize(imgcropsize),
                                    transforms.ToTensor()])                            
    picdataset = dnn_io.PicDataset(stim, transform=transform)
    assert 'stimID' in picdataset.csv_file.keys(), 'stimID must be provided in stimuli csv file'
    assert 'onset' in picdataset.csv_file.keys(), 'onset must be provided in stimuli csv file'
    assert 'duration' in picdataset.csv_file.keys(), 'duration must be provided in stimuli csv file'
 
    picdataloader = DataLoader(picdataset, batch_size=8, shuffle=False)
        # dnn_act: pic * channel * unit * unit
    
    # read dnn roi    
    chn_roi = None
    unit_roi = None
    if dnn_roi_filelist is not None:
        dnn_roi = pd.read_csv(dnn_roi_filelist)
        if 'channel' in dnn_roi.keys():
            chn_roi = dnn_roi['channel'].values
            chn_roi = np.asarray(chn_roi[~np.isnan(chn_roi)] - 1, dtype=np.int)
        if 'unit' in dnn_roi.keys():
            unit_roi = dnn_roi['unit'].values
            unit_roi = np.asarray(unit_roi[~np.isnan(unit_roi)] - 1, 
                                           dtype=np.int)
    # get dnn activation of dnn roi        
    dnn_act = analyzer.dnn_activation(
            picdataloader, net, layer, channel=list(chn_roi))
    dnn_act = dnn_act.reshape(dnn_act.shape[0], dnn_act.shape[1], -1)
    
    if unit_roi is not None:
        dnn_act = dnn_act[:,:,unit_roi]
           
    # unit dimention reduction
    if unit_reduc is not None:
        if unit_reduc == 'mean':
            dnn_act = dnn_act.mean(-1)[:,:,np.newaxis]
        elif unit_reduc == 'max':
            dnn_act = dnn_act.max(-1)[:,:,np.newaxis]
    
    n_stim = dnn_act.shape[0]
    n_chn = dnn_act.shape[1]
    n_unit = dnn_act.shape[2]
    
    print('dnn activation generated')
    

    #%% multivarient prediction analysis 
    # func
    def x_hrf(stim_csv_pd,x,hrf_model,fmri_frames,tr):
        '''convolve dnn_act with hrf and align with timeline of response
        
        parameters:
        ----------
            stim_csv_pd: pandas dataframe, with onset and duration keys.
            x: [n_event,n_sample]
                Onset, duration and x' 1st dim should have the same size.
            resp: total 1-d array
            tr: in sec
            
        '''
        x_hrfed = []
        for i in range(x.shape[1]):            
            exp_condition = [
                    stim_csv_pd['onset'],stim_csv_pd['duration'],x[:,i]]
            frametimes = np.arange(fmri_frames) * tr
            regressor,_ = compute_regressor(exp_condition,hrf_model,frametimes)            
            x_hrfed.append(regressor)
        
        x_hrfed = np.squeeze(np.asarray(x_hrfed)).transpose(1,0)
        return x_hrfed
    

    def dim2(x,axis=1):
        if np.ndim(x) == 1:
            return np.expand_dims(x,axis=axis)
        else:
            return x
        
    def x_pca(x,n_components):
        pca_m = decomposition.PCA(n_components)
        pc = pca_m.fit_transform(x)
        return pc,pca_m
        
    
    # prediction models
    def glm_r(x,y,cvfold):
        '''linear model using ordinary least squares
        
        parameters:
        -----------
        x: [n_samples, n_features]
        y: [n_samples, n_resp_variable]
        
        '''  
        
        # model score
        model = linear_model.LinearRegression()
        m_score = [model_selection.cross_val_score(
                model,x,y[:,y_i],scoring='explained_variance',cv=cvfold
                ) for y_i in range(y.shape[1])]
        m_score = dim2(np.asarray(m_score).mean(-1),axis=0)
        
        # output data
        model = linear_model.LinearRegression()
        model.fit(x, y)
        m_pred = model.predict(x)       
       
        return model, m_score, m_pred


    def lasso_r(x,y,cvfold):
        pass
    
    def svm_r(x,y,cvfold):
        pass
    
    def svm_c(x,y,cvfold):
        pass
    
    def lda_c(x,y,cvfold):
        pass
            



 
    def mv_model(x,y,pred_model,cvfold):

        if pred_model == 'glm':
            model, m_score, m_pred = glm_r(x,y,cvfold)
            return model, m_score, m_pred
        
        elif pred_model == 'lasso':
            lasso_r(x,y,cvfold)
            
        elif pred_model == 'svr':
            svm_r(x,y,cvfold)
                
        elif pred_model == 'svc':
            if response.endswith('nii') or response.endswith('nii.gz'):
                raise Exception('Classification is not supported with input as brain images.')
            svm_c(x,y,cvfold)
#        elif model == 'lda':
#            if response.endswith('nii') or response.endswith('nii.gz'):
#                raise Exception('Classification is not supported with input as brain images.')
#            model = lda.()
#            score_evl = 'accuracy'
        else:
            raise Exception('Please select lmr or lmc for univariate prediction analysis.')
            
        
        
    # mv main analysis
    if axis is None:                
        # pca on x
        if pca is None:
            x = dnn_act.reshape(n_stim,-1) # dnn_act: [n_stim, n_chn * n_unit]
        else: 
            x, pca_m = x_pca(dnn_act.reshape(dnn_act.shape[0],-1),pca)
        
        # hrf convolve, should be performed after pca
        if hrf is not None:
             x = x_hrf(picdataset.csv_file,x,hrf,resp.shape[0],tr)

            
        # prediction model
        mv_model(x=dnn_act,y=dim2(resp,1),pred_model=pred_model,cvfold=cvfold)
        
                   
    elif axis == 'channel':       
        for chn in range(dnn_act.shape[1]):
            # pca on x
            if pca is None:
                x = dnn_act.reshape(n_stim,-1) # dnn_act: [n_stim, n_chn * n_unit]
            else: 
                x, pca_m = x_pca(dnn_act.reshape(dnn_act.shape[0],-1),pca)
            
            # hrf convolve, should be performed after pca
            if hrf is not None:
                 x = x_hrf(picdataset.csv_file,x,hrf,resp.shape[0],tr)

        
    elif axis == 'column':
        for unit in range(dnn_act.shape[2]):
            # pca on x
            if pca is None:
                x = dnn_act.reshape(n_stim,-1) # dnn_act: [n_stim, n_chn * n_unit]
            else: 
                x, pca_m = x_pca(dnn_act.reshape(dnn_act.shape[0],-1),pca)
            
            # hrf convolve, should be performed after pca
            if hrf is not None:
                 x = x_hrf(picdataset.csv_file,x,hrf,resp.shape[0],tr)
        
    

     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   


    #%% ==================draft======================
    # Prepare model
    if model == 'glm':
        model = linear_model.LinearRegression()
        score_evl = 'explained_variance'
        
        
    elif model == 'lasso':
            model = linear_model.Lasso()
    elif model == 'lasso':
        if response.endswith('nii'):
            raise Exception('Classification is not supported with input as brain images.')
        model = linear_model.LogisticRegression()
        score_evl = 'accuracy'
    else:
        raise Exception('Please select lmr or lmc for univariate prediction analysis.')
    
    # Perform univariate prediction analysis
    if axis == 'channel':
        # nchannel * nvoxel
        acc_array = np.zeros((n_chn, brain_roi_size))
        accpos_array = np.zeros_like(acc_array)
        for i in range(brain_roi_size):
            print('voxel {}'.format(i+1))
            for nc in range(n_chn):
                scores_tmp = []
                for ux in range(n_unitx*n_unity):
                    scores_tmp.append(model_selection.cross_val_score(model, dnn_act[:,nc,ux][:,None], resp[:,i][:, None], scoring=score_evl, cv=cvfold))
                acc_array[nc,i] = np.max(scores_tmp)
                # Count coordinate from 1.
                accpos_array[nc,i] = np.argmax(scores_tmp)+1
    elif axis == 'column':
        # nunit * nvoxel
        acc_array = np.zeros((n_unitx*n_unity, brain_roi_size))
        accpos_array = np.zeros_like(acc_array)
        for i in range(brain_roi_size):
            print('voxel {}'.format(i+1))
            for ux in range(n_unitx*n_unity):
                scores_tmp = []
                for nc in range(n_chn):
                    scores_tmp.append(model_selection.cross_val_score(model, dnn_act[:,nc,ux][:,None], resp[:,i][:, None], scoring=score_evl, cv=cvfold))
                acc_array[ux,i] = np.max(scores_tmp)
                # Count coordinate from 1.
                accpos_array[ux,i] = np.argmax(scores_tmp)+1
    else:
        raise Exception('Please input channel or column in -axis')
            
    if response.endswith('nii'):
        acc_brainimg = np.zeros((n_chn, *resp_raw.shape[1:]))
        accpos_brainimg = np.zeros_like(acc_brainimg)
        for i, vx_idx in enumerate(brainact_idx):
            acc_brainimg[:, vx_idx[0], vx_idx[1], vx_idx[2]] = acc_array[:,i]
            accpos_brainimg[:, vx_idx[0], vx_idx[1], vx_idx[2]] = accpos_array[:,i]
            
    # Save files
    if not os.path.isdir(os.path.join(outdir, 'layer')):
        os.mkdir(os.path.join(outdir, 'layer'))
    if axis == 'channel':
        if not os.path.isdir(os.path.join(outdir, 'layer', 'channel')):
            os.mkdir(os.path.join(outdir, 'layer', 'channel'))
        if response.endswith('csv'):
            acc_pd = pd.DataFrame(acc_array, columns=roi_keys)
            accpos_pd = pd.DataFrame(accpos_array, columns=roi_keys)
            acc_pd.to_csv(os.path.join(outdir, 'layer', 'channel', 'accuracy.csv'), index=False)
            accpos_pd.to_csv(os.path.join(outdir, 'layer', 'channel', 'accuracy_pos.csv'), index=False)
        elif response.endswith('nii'):
            brain_io.save_brainimg(os.path.join(outdir, 'layer', 'channel', 'accuracy.'+brainimg_suffix, acc_brainimg, header))
        else:
            raise Exception('Not support yet, please contact to the author for implementation.')
    elif axis == 'column':
        if not os.path.isdir(os.path.join(outdir, 'layer', 'column')):
            os.mkdir(os.path.join(outdir, 'layer', 'column'))
        if response.endswith('csv'):
            acc_pd = pd.DataFrame(acc_array, columns=roi_keys)
            accpos_pd = pd.DataFrame(accpos_array, columns=roi_keys)
            acc_pd.to_csv(os.path.join(outdir, 'layer', 'column', 'accuracy.csv'), index=False)
            accpos_pd.to_csv(os.path.join(outdir, 'layer', 'column', 'accuracy_pos.csv'), index=False)
        elif response.endswith('nii'):
            brain_io.save_brainimg(os.path.join(outdir, 'layer', 'column', 'accuracy.'+brainimg_suffix, acc_brainimg, header))
        else:
            raise Exception('Not support yet, please contact to the author for implementation.')
    else:
        raise Exception('Please input channel or column in -axis')
            

    # ---Multivariate regression analysis---
    scores = []
    if model in ['glm', 'lasso']:
        # Extract brain activation
        picdataloader = DataLoader(picdataset, batch_size=8, shuffle=False)
        dnn_act = analyzer.dnn_activation(picdataloader, net, layer, channel)
        if dnn_act.ndim == 3:
            dnn_act = dnn_act[:, None, ...]
        # Reshape dnn_act and flatten its unit
        dnn_act = dnn_act.reshape((dnn_act.shape[0], dnn_act.shape[1]*dnn_act.shape[2]*dnn_act.shape[3]))
    
        if model == 'glm':
            model = linear_model.LinearRegression()
        elif model == 'lasso':
            model = linear_model.Lasso()
        else:
            raise Exception('Not support yet, please contact author for implementation.')
            
        if dnn_act.shape[0] < 10:
            comp_num = dnn_act.shape[0]
        else:
            comp_num = 10
        pca = decomposition.PCA(n_components=comp_num)
        # nsamples(pics)*nfeatures(units)
        for j in range(actnode_num):
            if (j%100 == 0) & (not roi):
                print('  Finish calculation on {} voxels'.format(j))
            # nsamples(pics)*nfeatures(voxels)
            # Decrease dimension using PCA
            dnn_act_pca = pca.fit_transform(dnn_act)
            # Cross validation
            cv = 2 if cvfold is None else cvfold
            scores_tmp = model_selection.cross_val_score(model, dnn_act_pca, response_list[:, j][:, None],
                                                         scoring='explained_variance', cv=cv)
            scores.append(np.mean(scores_tmp))
    
    elif model == 'nn':
        # process data
        pics = torch.tensor([pic.numpy() for pic, _ in picdataset])
        pics = pics.type(torch.float32)
        acts = torch.tensor(response_list, dtype=torch.float32)
    
        # truncate the pretrained neural network
        truncated_net = dnn_truncate(netloader.model, netloader.layer2indices[layer], layer)
        if 'fc' in layer:
            assert channel is None, 'FC layers have nothing to do with channels!'
            fc_in_num = list(truncated_net.modules())[-1].out_features
        elif 'conv' in layer:
            truncated_output = truncated_net(pics[0].unsqueeze(0))
            channel_num = list(truncated_net.modules())[-1].out_channels if channel is None else len(channel)
            channel_unit_num = truncated_output.shape[2:].numel()
            fc_in_num = channel_num * channel_unit_num
        else:
            raise ValueError("Wrong layer name!")
    
        if cvfold:
            # split data to cvfold folds
            # Each fold is then used once as a validation while the remaining folds form the training set.
            scores_list = []
            kf = model_selection.KFold(cvfold)
            for train_indices, test_indices in kf.split(pics):
                # train a new model
                train_pics = pics[train_indices]
                train_acts = acts[train_indices]
                dataset = TensorDataset(train_pics, train_acts)
                dataloader = DataLoader(dataset=dataset, batch_size=train_pics.shape[0], num_workers=1)
                transferred_net = TransferredNet(truncated_net, fc_in_num, train_acts.shape[1], channel)
                optimizer = torch.optim.Adam(transferred_net.fc.parameters(), lr=0.01)
                loss_func = nn.MSELoss()
                transferred_net = dnn_train_model(dataloader, transferred_net, loss_func, optimizer, 100)
    
                # test the trained model
                test_pics = pics[test_indices]
                test_acts = acts[test_indices]
                transferred_net.train(False)
                with torch.no_grad():
                    predicted_acts = transferred_net(test_pics)
    
                # calculate prediction score
                scores = []
                for x, y in zip(test_acts.numpy().T, predicted_acts.numpy().T):
                    r = pearsonr(x, y)[0]
                    if np.isnan(r):
                        r = 0
                    scores.append(np.power(r, 2))
                scores_list.append(scores)
            scores = np.nanmean(scores_list, 0)
    
        else:
            # use all data to train a model
            dataset = TensorDataset(pics, acts)
            dataloader = DataLoader(dataset=dataset, batch_size=pics.shape[0], num_workers=1)
            transferred_net = TransferredNet(truncated_net, fc_in_num, acts.shape[1], channel)
            optimizer = torch.optim.Adam(transferred_net.fc.parameters(), lr=0.01)
            loss_func = nn.MSELoss()
            transferred_net = dnn_train_model(dataloader, transferred_net, loss_func, optimizer, 100)
    
            # calculate prediction score
            transferred_net.train(False)
            with torch.no_grad():
                predicted_acts = transferred_net(pics)
            for x, y in zip(acts.numpy().T, predicted_acts.numpy().T):
                r = pearsonr(x, y)[0]
                if np.isnan(r):
                    r = 0
                scores.append(np.power(r, 2))
    
            # save net
            net_file = os.path.join(outdir, 'transferred_net.pkl')
            torch.save(transferred_net, net_file)
    
    else:
        raise Exception('Not support yet, please contact author for implementation.')
    
    # ---save prediction scores---
    if roi:
        score_df = pd.DataFrame({'ROI': roilabel, 'scores': scores})
        # Save behavior measurement into hardware
        score_df.to_csv(os.path.join(outdir, 'roi_score.csv'), index=False)
    else:
        # output image: channel*voxels
        out_brainimg = np.zeros((1, *brainimg_data.shape[1:]))
        for i, b_idx in enumerate(response_idx):
            out_brainimg[0, b_idx[0], b_idx[1], b_idx[2]] = scores[i]
        # Save image into hardware
        brain_io.save_brainimg(os.path.join(outdir, 'voxel_score.nii.gz'), out_brainimg, header)


