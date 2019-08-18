#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt



class Dnn_act:
    """ dnn activation
    
    Attibutes
    ---------
        
        data: array_like, 
            shape = [n_stim,n_chn,n_unit_in_row,n _unit_in_column]
    
    """
    
    def __init__(self, data, stim_per_cat):

        if data.ndim == 2 or data.ndim == 3:
            self.data = data
        elif data.ndim ==4:
            self.data = data.reshape([data.shape[0],data.shape[1],
                                      data.shape[2]*data.shape[3]])
        self.stim_num = np.shape(self.data)[0]
        self.stim_per_cat = stim_per_cat
        self.cat_num = int(self.stim_num / self.stim_per_cat)
        self.chn_num = np.shape(self.data)[1]
        
    def chn_act(self, top_n=5, relu=False, replace=False):
        
        if relu == True:
            self.data[self.data<0] = 0       
        unit_max_act = np.sort(self.data,-1)[:,:,-1*top_n:]       
        if replace == True:
            self.data = unit_max_act.mean(-1)
        else:
            return unit_max_act.mean(-1)
        

    def cat_mean_act(self):
        cur_shape = np.shape(self.data)
        new_shape = np.r_[self.cat_num,self.stim_per_cat,cur_shape[1:]]
        cat_data = self.data.reshape(new_shape)
        cat_mean = cat_data.mean(1)
        cat_std = cat_data.std(1)
        return cat_mean, cat_std


class Chn_score:
    """ channel value
    
    Attibutes
    ---------
        
        data: array_like, 
            shape = [n_cat,n_chn]
    
    """
    
    def __init__(self, data, chn_index=None):

        self.data = data
        self.cat_num = np.shape(self.data)[0]
        self.chn_num = np.shape(self.data)[1]
        if chn_index is None:
            chn_index = np.arange(self.chn_num) 
        self.chn_index = chn_index
    
    def sort_chn(self, chn_index):
        self.reset_chn_index()
        self.data = self.data[:, chn_index]
        self.chn_index = chn_index
        
    def reset_chn_index(self):
        self.data = self.data[:, np.argsort(self.chn_index)]
        self.chn_index = np.arange(self.chn_num) 
        

# glm
def generate_X(num_ft,samp_per_ft,target_ft=None):
    
    """generate dummy coding feature matrix
    
    """
    
    par = np.zeros([num_ft*samp_per_ft,num_ft])
    
    if target_ft != None:
        par[target_ft*samp_per_ft:
            (target_ft+1)*samp_per_ft, 
            target_ft] = num_ft - 1
        
        other_ft = np.delete(np.arange(num_ft), target_ft)
        for ft in other_ft:
            par[ft*samp_per_ft:(ft+1)*samp_per_ft, ft] = -1
            
    else:
        for ft in range(num_ft):
            par[ft*samp_per_ft:(ft+1)*samp_per_ft, ft] = 1
    return par


def glm_model(y,x,target_ft=None):  

    """build glm models

    Parameters
    ----------
        y: 2-d array, [n_samples, n_chns]   
        x: 2-d array, [n_samples, n_features]  
    
    """    
    
    t_pvalue = []  
    t_value = []
    
    if target_ft != None:
        for chn in range(y.shape[-1]):
            results = smf.OLS(y[:,chn],x).fit()
            t_pvalue.append(results.pvalues[target_ft+1])
            t_value.append(results.tvalues[target_ft+1])
            print('chn {0} done'.format(chn+1))
    else:
        for chn in range(y.shape[-1]):
            results = smf.OLS(y[:,chn],x).fit()
            t_pvalue.append(np.nan_to_num(results.pvalues))
            t_value.append(np.nan_to_num(results.tvalues))
            print('chn {0} done'.format(chn+1))
        
    t_pvalue = np.asarray(t_pvalue).T
    t_value = np.asarray(t_value).T
    
    return t_value,t_pvalue


# plot
def plot_chn_scatter(x,y,ref=None,cat_cls=None,label=None):
    
    """ 
    Parameters
    ---------
        
        x: channel index, shape = n_chn
        y: shape = [n_cat,n_chn]
        ref: highlight one specific cat
        cat_cls: summary categories into higher level class
    
    """   
    
    fig, ax = plt.subplots()
    
    if ref == None:
        if y.shape[0] <= 7:
            color = ['cornflowerblue','lightcoral','limegreen',
                     'gray', 'hotpink', 'blueviolet','gold']
            for cat in range(y.shape[0]):
                ax.scatter(x, y[cat,:], s=8, c=color[cat],label=label[cat])           
        else:
            color = np.random.random([3,y.shape[0]])
            for cat in range(y.shape[0]):
                ax.scatter(x, y[cat,:], s=8, c=color[:,cat],label=label[cat])

    elif ref != None and cat_cls == None:
        color = np.random.random([3,y.shape[0]])
        other_cat = np.delete(np.arange(y.shape[0]), ref)
        for cat in other_cat:
            ax.scatter(x, y[cat,:], s=8, c=color[:,cat])
        ax.scatter(x, y[ref,:], marker='o', s=30, c='', edgecolors='tab:red')
    elif ref != None and cat_cls != None:
        if len(cat_cls) <= 7:
            color = ['cornflowerblue','lightcoral','limegreen',
                     'gray', 'hotpink', 'blueviolet','gold']
        else:
            print('no more than 7 cat_cls')
        cat_cls_count = 0
        for cat_clsi in cat_cls:
            for cat in cat_clsi:
                ax.scatter(x, y[cat,:], s=8, c=color[cat_cls_count])
            cat_cls_count += 1
        ax.scatter(x, y[ref,:], marker='o', s=30, c='', edgecolors='tab:red')
    
    ax.legend()
    plt.show()