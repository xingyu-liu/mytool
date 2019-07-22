# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import stats

def isc(data1, data2):

    """calculate inter-subject correlation along the determined axis.
    
    Parameters
    ----------          
 
        data1: used to calculate functional connectivity, 
            shape = [n_samples, n_features].
        data2: used to calculate functional connectivity, 
            shape = [n_samples, n_features].
    
    Returns
    -------
        isc: point-to-point functional connectivity list of 
            data1 and data2, shape = [n_samples, ].
   
    Notes
    -----
        1. data1 and data2 should both be 2-dimensional.
        2. [n_samples, n_features] should be the same in data1 and data2.
        
    """
#    assert data1.ndim == 2 and data2.ndim == 2 and data1.shape == data2.shape, \
#        'data1 and data2 should have the same shape, and both should be 2-d array.\n \' \
#        Cannot calculate with shape {0}, {1}'.format(data1.shape, data2.shape)    
    
    data1 = np.nan_to_num(data1)
    data2 = np.nan_to_num(data2)
  
    z_data1 = np.nan_to_num(stats.zscore(data1,axis=-1))
    z_data2 = np.nan_to_num(stats.zscore(data2,axis=-1))
    corr = np.sum(z_data1*z_data2,axis=-1)/(np.size(data1,-1))
    
    return corr



def isfc(data1, data2):
    from scipy.spatial.distance import cdist

    """
    Cal functional connectivity between data1 and data2.
    Parameters
    ----------
        data1: used to calculate functional connectivity, 
            shape = [n_samples1, n_features].
        data2: used to calculate functional connectivity, 
            shape = [n_samples2, n_features].

    Returns
    -------
        isfc: functional connectivity map of data1 and data2, 
            shape = [n_samples1, n_samples2].

    Notes
    -----
        1. data1 and data2 should both be 2-dimensional.
        2. n_features should be the same in data1 and data2.
    """
    return 1 - cdist(data1, data2, metric='correlation')



def rdm_plot(data, label=None, fig_size=None, title=None, 
             delete_diag=None, show_value=None, colormap=None):
    
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm as mpl_cm

    if delete_diag is not None:
        data[range(np.size(data,0)),range(np.size(data,0))] = 0
    
    if fig_size is not None:
        fig = plt.figure(figsize=(fig_size[0],fig_size[1]))
    else:
        fig = plt.figure(figsize=(5,5))
    
    if title is not None:
        plt.title(title)
        
    if colormap is not None:
        cmap = mpl_cm.get_cmap(colormap)
    else:
        cmap = mpl_cm.coolwarm
    
    im = plt.imshow(data,cmap=cmap)
    
    plt.colorbar(im, fraction=0.02)

    
    if label is not None:
        # label them with the respective list entries
        plt.xticks(np.arange(np.size(data,1)), label,rotation=90)
        plt.yticks(np.arange(np.size(data,0)), label)
    
    
    if show_value is not None:
        valfmt = matplotlib.ticker.StrMethodFormatter('{x:.2f}')  
        for i in range(np.size(data,0)):
            for j in range(np.size(data,1)):
                plt.text(j, i, valfmt(data[i, j]), 
                         ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()


def plot_scatter_bar(matrix , x_bar=None, y_bar=None): 
    import matplotlib.pyplot as plt
    
    matrix = matrix.T
    scatter_data = np.where(matrix==1)
    x = scatter_data[0]
    y = scatter_data[1]

    # definitions for the axes
    plt.figure(figsize=(8, 8))
    left, width = 0.1, 0.8
    bottom, height = 0.1, 0.8
    spacing = 0.005
    if x_bar is not None:
        height = 0.65 
    if y_bar is not None:
        width = 0.65
    
    # start with a rectangular Figure
    rect_scatter = [left, bottom, width, height]  
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.set_xlim((0, matrix.shape[0]))
    ax_scatter.set_ylim((0, matrix.shape[-1]))  
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_scatter.scatter(x, y)

    if x_bar is not None:
        binwidth = 1       
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labelbottom=False)
        ax_histx.set_xlim(ax_scatter.get_xlim())
        ax_histx.hist(x, bins=range(0, matrix.shape[0],binwidth))    
    
    if y_bar is not None:
        binwidth = 1
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)
        ax_histy.set_ylim(ax_scatter.get_ylim())
        ax_histy.hist(y, bins=range(0, matrix.shape[-1],binwidth), 
                      orientation='horizontal')
     

 
    plt.show()
    
    
    
    
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
    import matplotlib.pyplot as plt
    
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
    from matplotlib import pyplot as plt
    from scipy.cluster import hierarchy
    
    t0 = time()
    Z = hierarchy.linkage(X, method='ward')
    print("%.2fs" % (time() - t0))
    
    plt.figure(figsize=(8, 12))
    dn = hierarchy.dendrogram(Z, above_threshold_color='#bcbddc',
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
    

def fdr_correction(x):
    temp = x.argsort()
    ranks = np.empty(len(temp), int)
    ranks[temp] = np.arange(len(temp)) + 1
    fdr_Q = x * (np.size(x) / ranks)
    
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
