# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def isc(data1, data2=None):

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

    if data2 is None:
        data2 = data1
    data1 = np.nan_to_num(data1)
    data2 = np.nan_to_num(data2)

    z_data1 = np.nan_to_num(stats.zscore(data1, axis=-1))
    z_data2 = np.nan_to_num(stats.zscore(data2, axis=-1))
    corr = np.sum(z_data1*z_data2, axis=-1)/(np.size(data1, -1))

    return corr


def isfc(data1, data2=None):
    from scipy.spatial.distance import cdist

    """Cal functional connectivity between data1 and data2.

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
    if data2 is None:
        data2 = data1

    corr = np.nan_to_num(1 - cdist(data1, data2, metric='correlation'))
    return corr


def rdm(data):
    """Cal representaion similarity matrix.

    Parameters
    ----------
        data1: used to calculate functional connectivity,
            shape = [n_samples, n_features].
    Returns
    -------
        rsm_value: upper triangle of rsm, without diagonal
    """

    rdm = 1 - isfc(data)[np.triu_indices(np.shape(data)[0], k=1)]
    return rdm


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

    data_mean = np.mean(data, axis=-1)
    data_std = np.std(data, axis=-1)
    data_tSNR = np.nan_to_num(data_mean / data_std)
    return data_tSNR


def d_prime(hits, misses, fas, crs):
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)[0, 0]
    half_fa = 0.5 / (fas + crs)[0, 0]
    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    hit_rate[hit_rate==1] = 1 - half_hit
    hit_rate[hit_rate==0] = half_hit 
    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    fa_rate[fa_rate==1] = 1 - half_fa
    fa_rate[fa_rate==0] = half_fa

    return stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)


def dice(x, y):
    """
    parameters:
    ----------
        x, y: 1-d array or list. x and y have the same size
    """
    dice_coef = []
    cat = np.unique(np.asarray([x, y]))
    for cat_i in cat:
        x_i = list(np.where(np.asarray(x) == cat_i)[0])
        y_i = list(np.where(np.asarray(y) == cat_i)[0])
        dice_i = (len(set(x_i) & set(y_i))) * 2 / (
                len(set(x_i)) + len(set(y_i)))
        dice_coef.append(dice_i)
    dice_coef = np.asarray(dice_coef).mean()
    return dice_coef


def sparseness(x, type='s', norm=False):
    """
    parameters:
    ----------
        x: [n_sitm] or [n_stim, n_cell], firing rate(activation) of each cell 
            to each stimulus
    """
    
    if np.ndim(x) == 1:
        x = x[:, np.newaxis]
        
    if norm is True:

        min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        x = min_max_scaler.fit_transform(x)
   
    n_stim = x.shape[0]

    # make sure any x > 0
    assert x.min() >= 0, 'x should all be positive'
    
    sparse_v = ((x.sum(0)/n_stim)**2) / (
            np.asarray([*map(lambda x: x**2, x)]).sum(0)/n_stim)
    # set sparse_v of cells that are always silent to 1
    sparse_v[x.sum(0) == 0] = 1
    
    if type == 's':
        sparse_v = (1 - sparse_v) / (1 - 1/n_stim)

    return sparse_v


def perct_agreement(x, y):
    """
    parameters:
    ----------
        x, y: 1-d array or list. x and y have the same size
    """
    agr_length = np.count_nonzero(np.asarray(x) == np.asarray(y))
    length = np.asarray(x).shape[0]
    return agr_length/length


def within_between(data, start_point_array):
    pivot = np.concatenate((start_point_array, [np.shape(data)[0]]))

    within = np.triu(data[0:pivot[0], 0:pivot[0]], k=1)
    within = np.reshape(within, np.size(within))
    zero_loc = np.where(within == 0)[0][0:pivot[0]*(pivot[0]+1)//2]
    within = np.delete(within, zero_loc)

    between = np.array([])
    for i in np.arange(0, np.size(pivot)-1, 1):
        # ------append within------------
        within_x = np.triu(data[pivot[i]:pivot[i+1], pivot[i]:pivot[i+1]], k=1)
        within_x = np.reshape(within_x, np.size(within_x))
        length = pivot[i+1] - pivot[i]
        zero_loc = np.where(within_x == 0)[0][0:length*(length+1)//2]
        within_x = np.delete(within_x, zero_loc)
        within = np.concatenate((within, within_x))
        # ------appdend between----------
        between_x = data[0:pivot[i], pivot[i]:pivot[i+1]]
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
    # --------normalization done----------
    edges = [(i, j) for i in
             range(np.shape(x)[0]) for j in range(i+1, np.shape(x)[0])]
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, np.shape(x)[0], 1))
    for i in np.arange(0, len(edges), 1):
        G.add_edge(edges[i][0], edges[i][1], weight=x[edges[i]])
    # --------draw graph----------------
    #  nx.draw(G, with_labels=True, font_weight='bold')

    return G


def rearrange_mat(x, rearrange_index):

    x_new = x[rearrange_index, :]
    x_new = x_new[:, rearrange_index]

    return x_new


def dendo_community(x):
    import community

    G = corr_matrix2graph(x)
    dendo = community.generate_dendrogram(G)
    dendo_community = np.array([dendo[0][key] for key in dendo[0].keys()])
    sort_index = np.argsort(dendo_community)

    sorted_x = rearrange_mat(x, sort_index)
    sorted_x = x[sort_index, :]
    sorted_x = sorted_x[:, sort_index]

    return sorted_x, sort_index


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

    # ---------- show last x merge -------------------------
    plt.figure(figsize=(5, 8))
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.ylabel('sample index or (cluster size)')
    plt.xlabel('distance')
    hierarchy.dendrogram(Z, truncate_mode='lastp',
                         p=last_merge_number, orientation='right',
                         leaf_font_size=12, show_contracted=True)
    plt.show()

    # ------------pick the n biggist cluster--------------------
    k = cluster_number
    X_cluster = hierarchy.fcluster(Z, k, criterion='maxclust')

    return X_cluster


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


def cohen_d(pre, post):
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
    pooled_sd = np.sqrt(((npost-1)*np.var(post, axis=-1, ddof=1) +
                         (npre-1)*np.var(pre, axis=-1, ddof=1)) / dof)
    d = (post.mean(-1) - pre.mean(-1)) / pooled_sd
    d = np.nan_to_num(d)

    return d
   

def residual(X, y):
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit(X, y)
    predict_y = model.predict(X)
    residual_y = y - predict_y

    return residual_y


def local_extreme(x, condition):
    derivative = np.zeros(np.shape(x))
    derivative[:, 1:] = derivative[:, 1:] - x[:, :-1]
    deri_change = np.zeros(np.shape(x))
    deri_change[:, 1:-1] = derivative[:, 1:-1]*derivative[:, 2:]
    deri_change[deri_change > 0] = 0
    deri_change[deri_change < 0] = 1

    if condition == 'max':
        derivative[derivative > 0] = 1
        derivative[derivative < 0] = 0
        extreme_loc = derivative*deri_change

    elif condition == 'min':
        derivative[derivative > 0] = 0
        derivative[derivative < 0] = 1
        extreme_loc = derivative*deri_change

    return extreme_loc


def rank(data, axis=0, order='descending'):
    if order == 'ascending':
        order = data.argsort(axis)
        ranks = order.argsort(axis)   

    elif order == 'descending':
        order = (data*-1).argsort(axis)
        ranks = order.argsort(axis)

    return ranks



def list_stats(x, method='mean', axis=None):
    if method == 'mean':
        return np.array([x[i].mean(axis) for i in range(len(x))])
    elif method == 'max':
        return np.array([x[i].max(axis) for i in range(len(x))])
    elif method == 'min':
        return np.array([x[i].min(axis) for i in range(len(x))])
    elif method == 'median':
        return np.array([np.median(x[i], axis) for i in range(len(x))])
    elif method == 'std':
        return np.array([x[i].std(axis) for i in range(len(x))])
    elif method == 'mode':
        return np.array([stats.mode(x[i])[0] for i in range(len(x))])

    # ignore nan
    elif method == 'nanmean':
        return np.array([np.nanmean(x[i], axis) for i in range(len(x))])
    elif method == 'nanmax':
        return np.array([np.nanmax(x[i], axis) for i in range(len(x))])
    elif method == 'nanmin':
        return np.array([np.nanmin(x[i], axis) for i in range(len(x))])
    elif method == 'nanmedian':
        return np.array([np.nanmedian(x[i], axis) for i in range(len(x))])
    elif method == 'nanstd':
        return np.array([np.nanstd(x[i], axis) for i in range(len(x))])
    elif method == 'nanmode':
        return np.array([stats.mode(x[i], nan_policy='omit')[0] for i 
                         in range(len(x))])
    # not stats
    elif method == 'size':
        return np.array([x[i].size for i in range(len(x))])
