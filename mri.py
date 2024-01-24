#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:25:50 2020

@author: liuxingyu
"""

import numpy as np
from nibabel.cifti2 import cifti2
from scipy.signal import fftconvolve
from scipy import ndimage
# from nipy.modalities.fmri.hemodynamic_models import spm_hrf
import copy

# %% 
def smooth_3dmri(func_data, func_mask, sigma=1, mode='reflect'):
    # add a new axis if the data is 3d
    if np.ndim(func_data) == 3:
        func_data = func_data[..., np.newaxis]
    
    # apply mask
    func_data[func_mask==0, :] = 0

    # smooth considering the boundary effect
    data_smoothed = ndimage.gaussian_filter(func_data, sigma=(sigma, sigma, sigma, 0), mode=mode)
    normalization_mask =  ndimage.gaussian_filter((func_mask!=0).astype(float), sigma=sigma, mode=mode)
    normalization_mask[normalization_mask == 0] = 1
    data_smoothed /= normalization_mask[..., np.newaxis]

    return np.squeeze(data_smoothed)


def roiing_volume(roi_annot, data, method='nanmean', key=None):

    if key is not None:
        roi_key = key
    else:
        roi_key = np.asarray(np.unique(roi_annot), dtype=np.int)
        roi_key = roi_key[roi_key!=0]
    
    roi_data = []
    
    for i in roi_key:
        # ignore nan
        if method == 'nanmean':
            roi_data.append(np.nanmean(data[roi_annot==i], 0))
        elif method == 'nanmedian':
            roi_data.append(np.nanmedian(data[roi_annot==i], 0))  
        elif method == 'nanstd':
            roi_data.append(np.nanstd(data[roi_annot==i], 0))        
        elif method == 'nanmax':
            roi_data.append(np.nanmax(data[roi_annot==i], 0))
        elif method == 'nanmin':
            roi_data.append(np.nanmin(data[roi_annot==i], 0))
        elif method == 'nansize':
            roi_data.append(np.sum(~np.isnan(data[roi_annot==i])))
    
    roi_data = np.asarray(roi_data)
    return roi_key, roi_data


def get_n_ring_neighbor(faces, n=1, ordinal=False, mask=None):
    """ copy from freeroi by Xiayu CHEN
    get n ring neighbor from faces array

    Parameters
    ----------
    faces : numpy array
        the array of shape [n_triangles, 3]
    n : integer
        specify which ring should be got
    ordinal : bool
        True: get the n_th ring neighbor
        False: get the n ring neighbor
    mask : 1-D numpy array
        specify a area where the ROI is
        non-ROI element's value is zero

    Returns
    -------
    lists
        each index of the list represents a vertex number
        each element is a set which includes neighbors of corresponding vertex
    """

    from scipy import sparse

    def mesh_edges(faces):
        """
        Returns sparse matrix with edges as an adjacency matrix
        Parameters
        ----------
        faces : array of shape [n_triangles x 3]
            The mesh faces
        Returns
        -------
        edges : sparse matrix
            The adjacency matrix
        """
        npoints = np.max(faces) + 1
        nfaces = len(faces)
        a, b, c = faces.T
        edges = sparse.coo_matrix((np.ones(nfaces), (a, b)),
                                  shape=(npoints, npoints))
        edges = edges + sparse.coo_matrix((np.ones(nfaces), (b, c)),
                                          shape=(npoints, npoints))
        edges = edges + sparse.coo_matrix((np.ones(nfaces), (c, a)),
                                          shape=(npoints, npoints))
        edges = edges + edges.T
        edges = edges.tocoo()
        return edges

    n_vtx = np.max(faces) + 1  # get the number of vertices
    if mask is not None and np.nonzero(mask)[0].shape[0] == n_vtx:
        # In this case, the mask covers all vertices.
        # So the program reset it as a None to save the computational cost.
        mask = None

    # find 1_ring neighbors' id for each vertex
    coo_w = mesh_edges(faces)
    csr_w = coo_w.tocsr()
    if mask is None:
        vtx_iter = range(n_vtx)
        n_ring_neighbors = [
                csr_w.indices[csr_w.indptr[i]:csr_w.indptr[i+1]] for i
                in vtx_iter]
        n_ring_neighbors = [set(i) for i in n_ring_neighbors]
    else:
        mask_id = np.nonzero(mask)[0]
        vtx_iter = mask_id
        n_ring_neighbors = [
                set(csr_w.indices[csr_w.indptr[i]:csr_w.indptr[i+1]])
                if mask[i] != 0 else set() for i in range(n_vtx)]
        for vtx in vtx_iter:
            neighbor_set = n_ring_neighbors[vtx]
            neighbor_iter = list(neighbor_set)
            for i in neighbor_iter:
                if mask[i] == 0:
                    neighbor_set.discard(i)

    if n > 1:
        # find n_ring neighbors
        one_ring_neighbors = [i.copy() for i in n_ring_neighbors]
        n_th_ring_neighbors = [i.copy() for i in n_ring_neighbors]
        # if n>1, go to get more neighbors
        for i in range(n-1):
            for neighbor_set in n_th_ring_neighbors:
                neighbor_set_tmp = neighbor_set.copy()
                for v_id in neighbor_set_tmp:
                    neighbor_set.update(one_ring_neighbors[v_id])

            if i == 0:
                for v_id in vtx_iter:
                    n_th_ring_neighbors[v_id].remove(v_id)

            for v_id in vtx_iter:
                # get the (i+2)_th ring neighbors
                n_th_ring_neighbors[v_id] -= n_ring_neighbors[v_id]
                # get the (i+2) ring neighbors
                n_ring_neighbors[v_id] |= n_th_ring_neighbors[v_id]
    elif n == 1:
        n_th_ring_neighbors = n_ring_neighbors
    else:
        raise RuntimeError(
                'The number of rings should be equal or greater than 1!')

    if ordinal:
        return n_th_ring_neighbors
    else:
        return n_ring_neighbors
    
    
class CiftiReader(object):
    """ copy from freeroi by Xiayu CHEN """

    def __init__(self, file_path):
        self.full_data = cifti2.load(file_path)

    @property
    def header(self):
        return self.full_data.header

    @property
    def brain_structures(self):
        return [_.brain_structure for _ in self.header.get_index_map(1).brain_models]

    @property
    def label_info(self):
        """
        Get label information from label tables
        Return:
        ------
        label_info[list]:
            Each element is a dict about corresponding map's label information.
            Each dict's content is shown as below:
                key[list]: a list of integers which are data values of the map
                label[list]: a list of label names
                rgba[ndarray]: shape=(n_label, 4)
                    The four elements in the second dimension are
                    red, green, blue, and alpha color components for label (between 0 and 1).
        """
        label_info = []
        for named_map in self.header.get_index_map(0).named_maps:
            label_dict = {'key': [], 'label': [], 'rgba': []}
            for k, v in named_map.label_table.items():
                label_dict['key'].append(k)
                label_dict['label'].append(v.label)
                label_dict['rgba'].append(v.rgba)
            label_dict['rgba'] = np.asarray(label_dict['rgba'])
            label_info.append(label_dict)

        return label_info

    @property
    def volume(self):
        return self.header.get_index_map(1).volume

    def brain_models(self, structures=None):
        """
        get brain model from cifti file
        Parameter:
        ---------
        structures: list of str
            Each structure corresponds to a brain model.
            If None, get all brain models.
        Return:
        ------
            brain_models: list of Cifti2BrainModel
        """
        brain_models = list(self.header.get_index_map(1).brain_models)
        if structures is not None:
            if not isinstance(structures, list):
                raise TypeError("The parameter 'structures' must be a list")
            brain_models = [brain_models[self.brain_structures.index(s)] for s in structures]
        return brain_models

    def map_names(self, rows=None):
        """
        get map names
        Parameters:
        ----------
        rows: sequence of integer
            Specify which map names should be got.
            If None, get all map names
        Return:
        ------
        map_names: list of str
        """
        named_maps = list(self.header.get_index_map(0).named_maps)
        if named_maps:
            if rows is None:
                map_names = [named_map.map_name for named_map in named_maps]
            else:
                map_names = [named_maps[i].map_name for i in rows]
        else:
            map_names = []
        return map_names

    def label_tables(self, rows=None):
        """
        get label tables
        Parameters:
        ----------
        rows: sequence of integer
            Specify which label tables should be got.
            If None, get all label tables.
        Return:
        ------
        label_tables: list of Cifti2LableTable
        """
        named_maps = list(self.header.get_index_map(0).named_maps)
        if named_maps:
            if rows is None:
                label_tables = [named_map.label_table for named_map in named_maps]
            else:
                label_tables = [named_maps[i].label_table for i in rows]
        else:
            label_tables = []
        return label_tables

    def get_data(self, structure=None, zeroize=False):
        """
        get data from cifti file
        Parameters:
        ----------
        structure: str
            One structure corresponds to one brain model.
            specify which brain structure's data should be extracted
            If None, get all structures, meanwhile ignore parameter 'zeroize'.
        zeroize: bool
            If true, get data after filling zeros for the missing vertices/voxels.
        Return:
        ------
        data: numpy array
            If zeroize doesn't take effect, the data's shape is (map_num, index_num).
            If zeroize takes effect and brain model type is SURFACE, the data's shape is (map_num, vertex_num).
            If zeroize takes effect and brain model type is VOXELS, the data's shape is (map_num, i_max, j_max, k_max).
        map_shape: tuple
            the shape of the map.
            If brain model type is SURFACE, the shape is (vertex_num,).
            If brain model type is VOXELS, the shape is (i_max, j_max, k_max).
            Only returned when 'structure' is not None and zeroize is False.
        index2v: list
            index2v[cifti_data_index] == map_vertex/map_voxel
            Only returned when 'structure' is not None and zeroize is False.
        """

        _data = np.array(self.full_data.get_fdata())
        if structure is not None:
            brain_model = self.brain_models([structure])[0]
            offset = brain_model.index_offset
            count = brain_model.index_count

            if zeroize:
                if brain_model.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
                    n_vtx = brain_model.surface_number_of_vertices
                    data = np.zeros((_data.shape[0], n_vtx), _data.dtype)
                    data[:, list(brain_model.vertex_indices)] = _data[:, offset:offset+count]
                elif brain_model.model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                    # This function have not been verified visually.
                    vol_shape = self.header.get_index_map(1).volume.volume_dimensions
                    data_shape = (_data.shape[0],) + vol_shape
                    data_ijk = np.array(list(brain_model.voxel_indices_ijk))
                    data = np.zeros(data_shape, _data.dtype)
                    data[:, data_ijk[:, 0], data_ijk[:, 1], data_ijk[:, 2]] = _data[:, offset:offset+count]
                else:
                    raise RuntimeError("The function can't support the brain model: {}".format(brain_model.model_type))
                return data
            else:
                if brain_model.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
                    map_shape = (brain_model.surface_number_of_vertices,)
                    index2v = list(brain_model.vertex_indices)
                elif brain_model.model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                    # This function have not been verified visually.
                    map_shape = self.header.get_index_map(1).volume.volume_dimensions
                    index2v = list(brain_model.voxel_indices_ijk)
                else:
                    raise RuntimeError("The function can't support the brain model: {}".format(brain_model.model_type))
                return _data[:, offset:offset+count], map_shape, index2v
        else:
            return _data

def save2cifti(file_path, data, brain_models, map_names=None, volume=None, label_tables=None):
    """ copy from freeroi by Xiayu CHEN
    Save data as a cifti file
    If you just want to simply save pure data without extra information,
    you can just supply the first three parameters.
    NOTE!!!!!!
        The result is a Nifti2Image instead of Cifti2Image, when nibabel-2.2.1 is used.
        Nibabel-2.3.0 can support for Cifti2Image indeed.
        And the header will be regard as Nifti2Header when loading cifti file by nibabel earlier than 2.3.0.
    Parameters:
    ----------
    file_path: str
        the output filename
    data: numpy array
        An array with shape (maps, values), each row is a map.
    brain_models: sequence of Cifti2BrainModel
        Each brain model is a specification of a part of the data.
        We can always get them from another cifti file header.
    map_names: sequence of str
        The sequence's indices correspond to data's row indices and label_tables.
        And its elements are maps' names.
    volume: Cifti2Volume
        The volume contains some information about subcortical voxels,
        such as volume dimensions and transformation matrix.
        If your data doesn't contain any subcortical voxel, set the parameter as None.
    label_tables: sequence of Cifti2LableTable
        Cifti2LableTable is a mapper to map label number to Cifti2Label.
        Cifti2Lable is a specification of the label, including rgba, label name and label number.
        If your data is a label data, it would be useful.
    """
    if file_path.endswith('.dlabel.nii'):
        assert label_tables is not None
        idx_type0 = 'CIFTI_INDEX_TYPE_LABELS'
    elif file_path.endswith('.dscalar.nii'):
        idx_type0 = 'CIFTI_INDEX_TYPE_SCALARS'
    elif file_path.endswith('.dtseries.nii'):
        idx_type0 = 'CIFTI_INDEX_TYPE_SERIES'
    else:
        raise TypeError('Unsupported File Format')

    # CIFTI_INDEX_TYPE_SCALARS always corresponds to Cifti2Image.header.get_index_map(0),
    # and this index_map always contains some scalar information, such as named_maps.
    # We can get label_table and map_name and metadata from named_map.
    if idx_type0 != 'CIFTI_INDEX_TYPE_SERIES':
        mat_idx_map0 = cifti2.Cifti2MatrixIndicesMap([0], idx_type0)

        if map_names is None:
            map_names = [None] * data.shape[0]
        else:
            assert data.shape[0] == len(map_names), "Map_names are mismatched with the data"

        if label_tables is None:
            label_tables = [None] * data.shape[0]
        else:
            assert data.shape[0] == len(label_tables), "Label_tables are mismatched with the data"

        for mn, lbt in zip(map_names, label_tables):
            named_map = cifti2.Cifti2NamedMap(mn, label_table=lbt)
            mat_idx_map0.append(named_map)
    else:
        mat_idx_map0 = cifti2.Cifti2MatrixIndicesMap([0], idx_type0, \
                number_of_series_points=data.shape[0], series_exponent=1, \
                    series_start=0, series_step=1, series_unit='SECOND')

    # CIFTI_INDEX_TYPE_BRAIN_MODELS always corresponds to Cifti2Image.header.get_index_map(1),
    # and this index_map always contains some brain_structure information, such as brain_models and volume.
    mat_idx_map1 = cifti2.Cifti2MatrixIndicesMap([1], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
    for bm in brain_models:
        mat_idx_map1.append(bm)
    if volume is not None:
        mat_idx_map1.append(volume)

    matrix = cifti2.Cifti2Matrix()
    matrix.append(mat_idx_map0)
    matrix.append(mat_idx_map1)
    header = cifti2.Cifti2Header(matrix)
    img = cifti2.Cifti2Image(data, header)

    cifti2.save(img, file_path)


def save_fslr_map(df, save_col_name, mask, bm, save_path, scale='roi'):
    '''
    save_col_name: the column name of the data in the df that is desired to
        be saved.
    mask: np.array data
    bm: corresponding brain model data of the mask
    scale: df must have the column indicating the scale. For scale='roi', 
        the named column is 'roi_mask'.
    '''

    if scale == 'roi':
        data2save = np.full(mask.shape, np.nan)
        for _, roi in enumerate(df['roi_mask'].unique()):
            data2save[mask==roi] = df.loc[df['roi_mask']==roi, save_col_name]

        save2cifti(save_path, data2save[None,...], bm)


def convolve_hrf(X, onsets, durations, n_vol, tr, ops=100):
    """
    Convolve each X's column iteratively with HRF and align with the timeline of BOLD signal
    Parameters
    ----------
    X : array
        Shape = n_event or [n_event, n_condition]
    onsets : array_like
        In sec. size = n_event
    durations : array_like
        In sec. size = n_event
    n_vol : int
        The number of volumes of BOLD signal
    tr : float
        Repeat time in second
    ops : int
        Oversampling number per second, must be one of the (10, 100, 1000)

    Returns
    -------
    X_hrfed : array
        The result after convolution and alignment
        shape = n_vol or [n_event, n_vol]
    """

    if np.ndim(X) == 1:
        X = X[..., None]
    assert np.ndim(X) == 2, 'X must be a 1D or 2D array'

    assert X.shape[0] == onsets.shape[0] and X.shape[0] == durations.shape[0], \
        'The length of onsets and durations should be matched with the number of events.'
    assert ops in (10, 100, 1000), 'Oversampling rate must be one of the (10, 100, 1000)!'

    # unify the precision
    decimals = int(np.log10(ops))
    onsets = np.round(np.asarray(onsets), decimals=decimals)
    durations = np.round(np.asarray(durations), decimals=decimals)
    tr = np.round(tr, decimals=decimals)
    
    assert onsets.min() >= 0, 'The onsets must be non-negative'
    if onsets.min() > 0:
        # The earliest event's onset is later than the start point of response.
        # We supplement it with zero-value event to align with the response.
        X = np.insert(X, 0, np.zeros(X.shape[1]), 0)
        durations = np.insert(durations, 0, onsets.min(), 0)
        onsets = np.insert(onsets, 0, 0, 0)

    # compute volume acquisition timing
    vol_t = (np.arange(n_vol) * tr * ops).astype(int)  

    # # generate hrf kernel
    hrf = np.array(spm_hrf(tr, oversampling=tr*ops))
    hrf = hrf[..., None]
    # hrf = np.load('mytool/hrf.npy') 

    # do convolution in batches for trade-off between speed and memory
    batch_size = int(100000 / ops)
    bat_indices = np.arange(0, X.shape[-1], batch_size)
    bat_indices = np.r_[bat_indices, X.shape[-1]]

    X_tc_hrfed = []
    for idx, bat_idx in enumerate(bat_indices[:-1]):
        X_bat = X[:, bat_idx:bat_indices[idx+1]]
        # generate X raw time course
        X_tc = np.zeros((vol_t.max(), X_bat.shape[-1]), dtype=np.float32)
        for i, onset in enumerate(onsets):
            onset_start = int(onset * ops)
            onset_end = int(onset_start + durations[i] * ops)
            X_tc[onset_start:onset_end, :] = X_bat[i, :]

        # convolve X raw time course with hrf kernal
        fftconvolve(X_tc[:10], hrf[:2])  # to fix weird bug (probably some setting change) caused by running spm_hrf
        X_tc_hrfed.append(fftconvolve(X_tc, hrf))
        # print('hrf convolution: sample {0} to {1} finished'.format(bat_idx+1, bat_indices[idx+1]))

    X_tc_hrfed = np.concatenate(X_tc_hrfed)

    # downsample to volume frame
    X_hrfed = X_tc_hrfed[vol_t, :]

    return X_hrfed
