# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
    
def tSNR(data):
    """alculate the temporal signal-to-noise ratio (tSNR) for each vertex
    Parameters
    ----------          
 
        data: used to calculate tSNR, 
            shape = [n_vertice, m_timepoints].
    
    Returns
    -------
        data_tSNR: the tSNR of data, shape = [n_vertice, ].
   
    Notes
    -----
    The tSNR was defined as the ratio between the mean of a timeseries and 
    its SD for each vertex
    """
    
    data_mean = np.mean(data,axis=-1)
    data_std = np.std(data, axis=-1)
    data_tSNR = np.nan_to_num(data_mean / data_std)
    return data_tSNR


def hist2grp(data, labels, fig_size, title, bin_amount, density):
    
    plt.figure(figsize=(fig_size[0],fig_size[1]))
    colors = ['red', 'blue']
    plt.hist(data, bins=bin_amount, density=density, color=colors, 
             label=labels)
    plt.legend(prop={'size': 10})
    plt.title(title)
#    plt.show()
    
    
def within_between(data, start_point_array):
    pivot = np.concatenate((start_point_array, [np.shape(data)[0]]))
    
    within = np.triu(data[0:pivot[0],0:pivot[0]],k=1)
    within = np.reshape(within, np.size(within))
    zero_loc = np.where(within==0)[0][0:pivot[0]*(pivot[0]+1)//2]
    within = np.delete(within, zero_loc)
    
    between = np.array([])
    for i in np.arange(0, np.size(pivot)-1,1):
        #------append within------------
        within_x = np.triu(data[pivot[i]:pivot[i+1],pivot[i]:pivot[i+1]], k=1)
        within_x = np.reshape(within_x, np.size(within_x))
        length = pivot[i+1] - pivot[i]
        zero_loc = np.where(within_x==0)[0][0:length*(length+1)//2]
        within_x = np.delete(within_x, zero_loc)
        within = np.concatenate((within, within_x))
        #------appdend between----------
        between_x = data[0:pivot[i],pivot[i]:pivot[i+1]]
        between_x = np.reshape(between_x, np.size(between_x))
        between = np.concatenate((between, between_x))
    return within, between


def corr_matrix2graph(corr_matrix):
    import networkx as nx
    from sklearn.preprocessing import MinMaxScaler
    
    x = corr_matrix
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    #--------normalization done----------
    edges = [(i, j) for i in 
             range(np.shape(x)[0]) for j in range(i+1, np.shape(x)[0])]
    G = nx.Graph()
    G.add_nodes_from(np.arange(0,np.shape(x)[0],1))
    for i in np.arange(0, len(edges), 1):
        G.add_edge(edges[i][0], edges[i][1], weight=x[edges[i]])
    #--------draw graph----------------   
#    nx.draw(G, with_labels=True, font_weight='bold')   
    
    return G


def cluster(X, last_merge_number, cluster_number):
    from time import time
    from scipy.cluster import hierarchy
    
    t0 = time()
    Z = hierarchy.linkage(X, method='ward')
    print("%.2fs" % (time() - t0))
    
    plt.figure(figsize=(8, 12))
    hierarchy.dendrogram(Z, above_threshold_color='#bcbddc',
                         orientation='right')
    plt.show()
    
    #---------- show last x merge -------------------------
    plt.figure(figsize=(5, 8))
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.ylabel('sample index or (cluster size)')
    plt.xlabel('distance')
    hierarchy.dendrogram(Z, truncate_mode='lastp', 
                         p=last_merge_number, orientation='right', 
                         leaf_font_size=12., show_contracted=True)
    plt.show()
    
    #------------pick the n biggist cluster--------------------
    k = cluster_number
    X_cluster = hierarchy.fcluster(Z, k, criterion='maxclust')
    
    return X_cluster

    
def dendo_community(x):
    import community 
    
    G = corr_matrix2graph(x)    
    dendo = community.generate_dendrogram(G)
    dendo_community = np.array([dendo[0][key] for key in dendo[0].keys()])
    sort_index = np.argsort(dendo_community)
    sorted_x = x[sort_index,:]
    sorted_x = sorted_x[:, sort_index]
    
    return sorted_x, sort_index


def fdr_correction(p_value):
    """
    Parameters
    ----------         
        p_value: (n, ) array_like

    """    
    temp = p_value.argsort()
    ranks = np.empty(len(temp), int)
    ranks[temp] = np.arange(len(temp)) + 1
    fdr_Q = p_value * (np.size(p_value) / ranks)
    
    return fdr_Q
        

def cohen_d(pre,post):
    """alculate the cohen's d effect size of pre- and post-denoising fMRI data 
    Parameters
    ----------         
        pre: value of a certain measurement of pre-denoising fMRI data, 
            shape = [n_vertice, ].
 
        post: value of a certain measurement of post-denoising fMRI data, 
            shape = [n_vertice, ].
   
    Returns
    -------
        d: the cohen's d of pre and post, 
            shape = [n_vertice, ].
   
    Notes
    -----
        Cohen's d was calculated as the mean difference 
        between the pre- and post-denoising fMRI data divided by the pooled SD
    """
    npost = np.shape(post)[-1]
    npre = np.shape(pre)[-1]
    dof = npost + npre - 2
    d =  ((post.mean(-1) - pre.mean(-1)) / 
          np.sqrt(((npost-1)*np.var(post, axis=-1, ddof=1) + 
                   (npre-1)*np.var(pre, axis=-1, ddof=1)) / dof))
    d = np.nan_to_num(d)
    
    return d

    
def residual(X,y):
    from sklearn.linear_model import LinearRegression
    
    model = LinearRegression().fit(X, y)
    predict_y = model.predict(X)
    residual_y = y - predict_y
    
    return residual_y


def local_extreme(x,condition):
    derivative = np.zeros(np.shape(x))
    derivative[:,1:] = derivative[:,1:] - x[:,:-1]
    deri_change = np.zeros(np.shape(x))
    deri_change[:,1:-1] = derivative[:,1:-1]*derivative[:,2:]
    deri_change[deri_change>0] = 0
    deri_change[deri_change<0] = 1
    
    if condition == 'max':
        derivative[derivative>0] = 1
        derivative[derivative<0] = 0
        extreme_loc = derivative*deri_change
        
    elif condition == 'min':
        derivative[derivative>0] = 0
        derivative[derivative<0] = 1
        extreme_loc = derivative*deri_change        
     
    return extreme_loc


def roiing_volume(roi_annot,volume_ts,roix_regressed):
#    not_roi = np.where(roi_annot == 0)
#    volume_ts[not_roi[0],not_roi[1],not_roi[2],:] = 0
    roi_ts = []
    
    roi_label = np.asarray(np.unique(roi_annot), dtype=np.int)[1:]
    for i in roi_label:
        roi_i_loc = np.where(roi_annot==i)
        roi_i = []
        for j in range(len(roi_i_loc[0])):
            roi_i.append(volume_ts[roi_i_loc[0][j], roi_i_loc[1][j], 
                               roi_i_loc[2][j], :])
        roi_i = np.asarray(roi_i, dtype=np.float64)
        roi_ts.append(roi_i)
    
    if roix_regressed:
        roix_loc = np.where(roi_annot==roix_regressed)
        roix = roi_ts[roix_loc[0],roix_loc[1],roix_loc[2],:].mean(0)
        roix = roix.reshape(-1,1)
        roi_ts_xregressed = roi_ts
        for i in range(len(roi_ts)):
            for j in range(np.shape([roi_ts[i]])[0]):           
                observed_y = roi_ts[i][j,:].reshape(-1,1)
                roi_ts_xregressed[i] = residual(roix, observed_y)[:,0]
        roi_ts = roi_ts_xregressed
        
    return roi_ts



def roiing_volume_roi_mean(roi_annot,volume_ts):
    # roi_annot should always start with 1
    
    roi_label = np.asarray(np.unique(roi_annot), dtype=np.int)[1:]
    roi_ts = np.zeros([roi_label.max(),1,1,np.shape(volume_ts)[-1]])
    
    for i in roi_label:
        roi_i_loc = np.where(roi_annot==i)
        roi_i = volume_ts[roi_i_loc[0],roi_i_loc[1],roi_i_loc[2],:]
        roi_ts[i-1,0,0,:]= roi_i.mean(0)
    
    return roi_ts
