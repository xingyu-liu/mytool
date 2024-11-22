# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %%
import numpy as np
from scipy import stats, ndimage
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from community import community_louvain
import copy
import matplotlib.pyplot as plt

# %%
def isc(data1, data2=None, rank_first=False):

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
        data2 = copy.deepcopy(data1)
    data1 = np.nan_to_num(data1)
    data2 = np.nan_to_num(data2)

    if rank_first:
        data1 = stats.rankdata(data1, axis=-1)
        data2 = stats.rankdata(data2, axis=-1)

    data1 = np.nan_to_num(stats.zscore(data1, axis=-1))
    data2 = np.nan_to_num(stats.zscore(data2, axis=-1))

    corr = np.sum(data1*data2, axis=-1) / np.sqrt(np.sum(data1**2, axis=-1)*np.sum(data2**2, axis=-1))

    return corr


def isfc(data1, data2=None, rank_first=False):

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
        data2 = copy.deepcopy(data1)
    data1 = np.nan_to_num(data1)
    data2 = np.nan_to_num(data2)

    if rank_first:
        data1 = stats.rankdata(data1, axis=-1)
        data2 = stats.rankdata(data2, axis=-1)

    dist = np.nan_to_num(1 - cdist(data1, data2, metric='correlation'))
    return dist


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


def normalize(x, norm_range=[0,1]):
    """ 
    if series is True, the last axis should be series 
    """
    dim = np.ndim(x)
    if dim == 1:
        x = x[..., None]
    min_max_scaler = MinMaxScaler(feature_range=norm_range)
    x = min_max_scaler.fit_transform(x)  
    if dim == 1:
        x = x[:, 0]
        
    return x

def IQR_replace_outlier(x, times=3, replace_policy='nan'):
    ''' 
    x shape: [n_sample, n_series]
    replace_policy: ['nan', 'IQR_bound', 'median']
    '''

    if np.ndim(x) ==  1:
        x = x[..., None]

    iqr = stats.iqr(x, axis=0)
    qu = stats.mstats.mquantiles(x, 0.75, axis=0)[0]
    ql = stats.mstats.mquantiles(x, 0.25, axis=0)[0]

    # upper bound
    bu = qu + times*iqr
    bl = ql - times*iqr
    
    # replace
    x_post = copy.deepcopy(x)
    if replace_policy == 'nan':
        x_post[x_post > bu] = np.nan
        x_post[x_post < bl] = np.nan
    elif replace_policy == 'IQR_bound':
        for si in range(x_post.shape[-1]):
            x_post[x_post[:,si] > bu[si], si] = bu[si]
            x_post[x_post[:,si] < bl[si], si] = bl[si]
    elif replace_policy == 'median':
        for si in range(x_post.shape[-1]):
            x_post[x_post[:,si] > bu[si], si] = np.median(x_post[:, si])
            x_post[x_post[:,si] < bl[si], si] = np.median(x_post[:, si])
    
    return x_post


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

    G = corr_matrix2graph(x)
    dendo = community_louvain.generate_dendrogram(G)
    dendo_community = np.array([dendo[0][key] for key in dendo[0].keys()])
    sort_index = np.argsort(dendo_community)

    sorted_x = rearrange_mat(x, sort_index)
    sorted_x = x[sort_index, :]
    sorted_x = sorted_x[:, sort_index]

    return sorted_x, sort_index


def cluster(X, cluster_number, label=None):
    from scipy.cluster import hierarchy

    Z = hierarchy.linkage(X, method='ward')

    plt.figure(figsize=(5, 8))
    hierarchy.dendrogram(Z, above_threshold_color='#bcbddc',labels=label,
                         orientation='right')
    plt.show()

    # # ---------- show last x merge -------------------------
    # plt.figure(figsize=(2, 3))
    # plt.title('Hierarchical Clustering Dendrogram (truncated)')
    # plt.ylabel('sample index or (cluster size)')
    # plt.xlabel('distance')
    # hierarchy.dendrogram(Z, truncate_mode='lastp',
    #                      p=last_merge_number, orientation='right',
    #                      leaf_font_size=12, show_contracted=True)
    # plt.show()

    # ------------pick the n biggist cluster--------------------
    X_cluster = hierarchy.fcluster(Z, cluster_number, criterion='maxclust')

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

    if np.ndim(X) == 1:
        X = X[..., None]
    if np.ndim(y) == 1:
        y = y[..., None]

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


def smooth_within_bounday(data, mask, sigma=1, mode='nearest'):

    data_smoothed = ndimage.gaussian_filter(data, sigma=sigma, mode=mode)
    normalization_mask =  ndimage.gaussian_filter((mask!=0).astype(float), sigma=sigma, mode=mode)
    normalization_mask[normalization_mask == 0] = 1
    data_smoothed /= normalization_mask
    return data_smoothed


def sparse2dense(sparse_data, mag_value, freq=100):
    '''
    Convert sparse data to dense data in the original space.

    Parameters:
    - sparse_data (ndarray): Sparse data of shape (n_sample, m_coordinates). 
        Make sure to set the coordinates type as integers if they are integers.
    - mag_value (ndarray): Magnitude values of shape (n_sample).
    - freq (int): Number of points to generate along each coordinate axis when the coordinates are not integers.
        Default is 100.

    Returns:
    - dense_data (ndarray): Dense data in the original space.
    - gridsmesh (list): List of meshgrid arrays representing the coordinate grids.

    '''

    # remove the offset of the coordinates
    sparse_data = sparse_data - sparse_data.min(axis=0)
    
    grids = []
    for i in range(sparse_data.shape[1]):
        if sparse_data.dtype == 'int':
            grids.append(np.arange(sparse_data[:, i].min(), sparse_data[:, i].max()+1))
        else:
            grids.append(np.linspace(sparse_data[:, i].min(), sparse_data[:, i].max(), freq))

    # Put sparse data back to the original space
    dense_data = np.ones(np.concatenate([i.shape for i in grids])) * np.nan
    if sparse_data.dtype == 'int':
        dense_data[tuple(sparse_data.T)] = mag_value
    else:
        dense_data[tuple([np.argmin(np.abs(grids[i][:, np.newaxis] - sparse_data[:, i]), axis=0) \
                        for i in range(sparse_data.shape[1])])] = mag_value
    gridsmesh = np.meshgrid(*grids)

    return dense_data, gridsmesh



def partial_corr(C):
    
    from scipy import linalg
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.

    Author: Fabian Pedregosa-Izquierdo, f@bianp.net
    Testing: Valentina Borghesani, valentinaborghesani@gmail.com
    
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable


    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr

# compute effective dimention given PCA explained variance
def effective_dim(explained_variances, method='n2'):
    """
    Compute the effective dimensionality of a dataset based on the explained variances of its principal components.

    :param explained_variances: Array of explained variances for each principal component.
    :return: Effective dimensionality.
    """
    # Convert explained variances to proportions
    proportions = explained_variances / np.sum(explained_variances)
    if method == 'n1':
        pass
    elif method == 'n2':
        effective_dim = proportions.sum()**2 / np.sum(proportions**2)

    return effective_dim


def crop_to_non_nan_region(data):
    """
    Crops the input array to the smallest region that contains all non-NaN values.
    
    Parameters:
        data (np.ndarray): Input array of any dimension.
        
    Returns:
        cropped_array (np.ndarray): Cropped array.
        slices (tuple): Slices that can be used to crop other arrays to the same region.
    """
    valid_loc = np.where(~np.isnan(data))
    slices = tuple(slice(np.min(dim), np.max(dim) + 1) for dim in valid_loc)

    return data[slices], slices


def partial_corr(corr_mat):
    '''
    Calculate the partial correlation matrix from the correlation matrix.
    
    Parameters:
    corr_mat (numpy.ndarray): Correlation matrix.
    
    Returns:
    numpy.ndarray: Partial correlation matrix.
    '''
    # Calculate the inverse of the correlation matrix
    inv_corr_mat = np.linalg.inv(corr_mat)
    
    # Initialize the partial correlation matrix
    p_corr_mat = np.zeros_like(inv_corr_mat)
    
    # Calculate the partial correlations
    for i in range(inv_corr_mat.shape[0]):
        for j in range(inv_corr_mat.shape[1]):
            if i != j:
                # Off-diagonal elements: calculate partial correlations
                p_corr_mat[i, j] = -inv_corr_mat[i, j] / np.sqrt(inv_corr_mat[i, i] * inv_corr_mat[j, j])
            else:
                # Diagonal elements should be 1
                p_corr_mat[i, j] = 1
                
    return p_corr_mat


def get_p_for_r(r, n):
    """Calculate p-values for Pearson correlation coefficients.
    
    Parameters
    ----------
    r : float or numpy.ndarray
        Correlation coefficient(s). Can be a single value or 1D array.
    n : int
        Sample size used to calculate the correlation(s). Must be > 2.
        
    Returns
    -------
    float or numpy.ndarray
        P-value(s) corresponding to the correlation coefficient(s).
        Returns same type as input (single value or array).
        
    Notes
    -----
    Uses Student's t-distribution to compute two-tailed p-values.
    Formula: t = r * sqrt(n-2) / sqrt(1-r^2)
    """
    # Input validation
    if n <= 2:
        raise ValueError("Sample size (n) must be greater than 2")
        
    # Convert single value to array if needed
    r_is_single = np.isscalar(r)
    r = np.asarray(r)
    
    # Calculate p-values using t-distribution
    t_stat = np.abs(r) * np.sqrt(n-2) / np.sqrt(1-r**2)
    p = stats.t.sf(t_stat, n-2) * 2
    
    # Return same format as input
    return p[0] if r_is_single else p
