#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 23:25:50 2020

@author: liuxingyu
"""
# %%
import os
import numpy as np
from nibabel.cifti2 import cifti2
import scipy
# from nipy.modalities.fmri.hemodynamic_models import spm_hrf
import nibabel as nib
import subprocess
import mytool
from typing import Union, Tuple
from gdist import compute_gdist
from typing import Optional

# %%
# io
def determine_mri_cifti_type(f_path):
    """
    Determine if a file path corresponds to a CIFTI file format.
    
    Args:
        f_path (str): Path to the file
        
    Returns:
        bool: True if the file is a CIFTI format, False otherwise
    """
    cifti_extensions = ('.dscalar.nii', '.dlabel.nii', '.dtseries.nii')
    return any(f_path.endswith(ext) for ext in cifti_extensions)


def load_mri_data(f_path, bs=None, zeroize=False):
    '''
    f_path: file path
    bs: brain structures, only for cifti file. Should be a list of strings.
    '''
    
    f_name = os.path.basename(f_path)

    if f_name.endswith('.nii.gz'):
        data_dict = nib.load(f_path).get_fdata()
        affine = nib.load(f_path).affine
        header = nib.load(f_path).header

        return data_dict, affine, header

    elif f_name.endswith('.gii'):
        data_dict = nib.load(f_path).agg_data()

        return data_dict, None, None

    elif f_name.endswith('.annot'):
        data_dict = nib.freesurfer.read_annot(f_path)

        return data_dict, None, None
    
    elif f_name.endswith('.dtseries.nii') or f_name.endswith('.dscalar.nii') or f_name.endswith('.dlabel.nii'):
        data_dict = {}
        data_dict['zeroize'] = zeroize
        reader = CiftiReader(f_path)

        # data = reader.get_data()
        # if bs is not None:
        #     data_dict['bs'] = bs
        #     data_dict['bm'] = reader.brain_models(data_dict['bs'])

        #     data_concat = []
        #     for struci in data_dict['bs']:
        #         brain_model = reader.brain_models([struci])[0]
        #         offset = brain_model.index_offset
        #         count = brain_model.index_count
        #         data_concat.append(data[:, offset:offset+count])
        #     data = np.concatenate(data_concat, axis=-1)
        
        # else:
        #     data_dict['bs'] = reader.brain_structures
        #     data_dict['bm'] = reader.brain_models()
        
        if bs is None:
            data_dict['bs'] = reader.brain_structures
            data_dict['bm'] = reader.brain_models()
        else:
            data_dict['bs'] = bs
            data_dict['bm'] = reader.brain_models(data_dict['bs'])

        data = []
        for struci in data_dict['bs']:
            datai = reader.get_data(struci, zeroize=zeroize)
            if not zeroize:
                datai = datai[0]
            data.append(datai)
        data = np.concatenate(data, axis=-1)

        data_dict['data'] = data.T
        data_dict['volume'] = reader.volume

        return data_dict

def save_mri_data(data, f_path, affine=None, header=None, ref_f=None, print_f=False):

    f_name = os.path.basename(f_path)

    # if f_name doesn't end with .nii.gz, .gii, .dtseries.nii, .dscalar.nii, raise error
    if not f_name.endswith(('.nii.gz', '.gii', '.dtseries.nii', '.dscalar.nii', '.dlabel.nii')):
        raise ValueError('f_path should end with .nii.gz, .gii, .dtseries.nii, .dscalar.nii, .dlabel.nii')

    # get ref info
    if ref_f is not None and os.path.basename(ref_f).endswith('.nii.gz'):
        affine = nib.load(ref_f).affine
        header = nib.load(ref_f).header

    # save data
    if f_name.endswith('.nii.gz'):
        img = nib.Nifti1Image(data, affine, header=header)
        nib.save(img, f_path)

    elif f_name.endswith('.gii'):
        img = nib.gifti.GiftiImage()
        img.add_gifti_data_array(nib.gifti.GiftiDataArray(data=data.astype(np.float32), intent='NIFTI_INTENT_NONE'))
        img.to_filename(f_path)
    
    elif f_name.endswith('.dtseries.nii') or f_name.endswith('.dscalar.nii') or f_name.endswith('.dlabel.nii'):
        # if data is not a dict, raise error
        if not isinstance(data, dict):
            raise ValueError('data should be a dict, including data, bm, (volume)')
        
        if data['zeroize']:
            raise ValueError('zeroize should be False when saving CIFTI file, otherwise the brain model will be incorrect')

        # # make sure bm starts from 0 and concatenated correctly
        index_offset = 0
        for bm_id in range(len(data['bm'])):
            data['bm'][bm_id].index_offset = index_offset

            index_offset += data['bm'][bm_id].index_count
        if 'volume' not in data.keys():
            data['volume'] = None

        save2cifti(f_path, data['data'], data['bm'], volume=data['volume'])

    if print_f:
        print(f'data saved to {f_path}')
    

def save_img_roiwise(value, key, atlas_data, save_f, atlas_data_f=None):
    '''
    atlas_data: 1d or 2d array, [n_roi, n_maps]
    roi_mask: specify the roi_mask to save the value in the volume
    value: the value to save in the volume. shape: [n_roi, n_map], should match the order of roi_mask
    save_f: output volume (nii.gz) /surface file (func.gii)
    '''

    atlas_data_sq = np.squeeze(atlas_data)

    if atlas_data_f is not None:
        _, affine, header = load_mri_data(atlas_data_f)
    else:
        affine, header = np.eye(4), None

    if np.ndim(value) == 1:
        value = value[:, np.newaxis]

    # put value back to the volume
    sdata = np.ones(list(atlas_data_sq.shape) + [value.shape[1]]) * np.nan
    for i, roi_maski in enumerate(key):
        sdata[atlas_data_sq==roi_maski] = value[i]

    # save the volume
    save_mri_data(sdata, save_f, affine, header=header)
    

def save_img_inmask(mask_data, mask_data_f, value, save_f):
    '''
    mask_data: 3d numpy array
    value: the value to save in the volume. shape: [n_voxel, n_map]
    vol_f: output volume file
    '''

    mask_data_sq = np.squeeze(mask_data)

    if np.ndim(mask_data_sq) == 3:
        data_type='vol'
        if not save_f.endswith('.nii.gz'):
            raise ValueError('vol output file should be in nii.gz format')
    elif np.ndim(mask_data_sq) == 1:
        data_type='surf'
        if not save_f.endswith('.func.gii'):
            raise ValueError('surf output file should be in func.gii format')

    if mask_data_f is not None:
        _, affine, header = load_mri_data(mask_data_f)
    else:
        affine, header = np.eye(4), None

    if np.ndim(value) == 1:
        value = value[:, np.newaxis]

    # put value back to the volume
    sdata = np.ones(list(mask_data.shape) + [value.shape[1]]) * np.nan
    sdata[mask_data!=0] = value

    # save the volume
    save_mri_data(sdata, save_f, affine, header=header)


def vol2surf(vol_f, surf_f, hemi, fs_dir, fs_sub):
    # surf_f in 'func.gii' format if visualization using wb_view

    env = os.environ.copy()
    env['SUBJECTS_DIR'] = fs_dir

    cmd = f'mri_vol2surf --mov {vol_f} --regheader {fs_sub} --hemi {hemi}' + \
            f' --o {surf_f} --projfrac 0.5'
    subprocess.run(f'zsh -c "source ~/.zshrc && {cmd}"', shell=True, env=env)

def save_fslr_map(df, save_col_name, mask, bm, save_path, scale='roi'):
    '''
    save_col_name: the column name of the data in the df that is desired to
        be saved.
    mask: np.array data
    bm: corresponding brain model data of the mask
    scale: df must have the column indicating the scale. For scale='roi', 
        the named column is 'key'.
    '''

    if scale == 'roi':
        data2save = np.full(mask.shape, np.nan)
        for _, roi in enumerate(df['key'].unique()):
            data2save[mask==roi] = df.loc[df['key']==roi, save_col_name]

        save2cifti(save_path, data2save[None,...], bm)

        
# %% 
def spatial_smooth_3d(input_data: np.ndarray, 
                     mask: np.ndarray = None, 
                     sigma: Union[float, Tuple[float, float, float]] = 1.0,
                     mode: str = 'reflect',
                     preserve_mask: bool = False) -> np.ndarray:
    """Applies 3D spatial smoothing to MRI data with optional masking.

    This function performs Gaussian smoothing on 3D or 4D MRI data while properly 
    handling boundary effects at mask edges. For 4D data, smoothing is applied to 
    each 3D volume independently.

    Parameters
    ----------
    input_data : np.ndarray
        Input data to smooth. Can be 3D or 4D array. If 4D, the last dimension
        represents different volumes/timepoints.
    mask : np.ndarray, optional
        Binary mask specifying regions to smooth. Should be same shape as input_data
        excluding the last dimension for 4D data. If None, all voxels are smoothed.
    sigma : float or tuple of float, optional
        Standard deviation for Gaussian kernel. Can be:
        - Single float: Same sigma applied to all spatial dimensions
        - Tuple of 3 floats: Different sigma for each spatial dimension (x,y,z)
        Default is 1.0
    mode : str, optional
        Determines how edges are handled. Options: 'reflect' (default), 'constant',
        'nearest', 'mirror', 'wrap'. See scipy.ndimage.gaussian_filter for details.
    preserve_mask : bool, optional
        If True, values in masked-out regions (mask==0) will retain their original 
        values instead of being set to 0. Default is False.

    Returns
    -------
    np.ndarray
        Smoothed data with same shape as input_data

    Raises
    ------
    ValueError
        If input shapes are incompatible or parameters are invalid
    TypeError
        If input types are incorrect

    Notes
    -----
    The smoothing operation is normalized at mask boundaries to prevent edge 
    artifacts. This is done by dividing the smoothed data by a smoothed version 
    of the mask.
    """
    # Input validation
    if not isinstance(input_data, np.ndarray):
        raise TypeError("input_data must be a numpy array")
    
    ndim = input_data.ndim
    if ndim not in (3, 4):
        raise ValueError("input_data must be 3D or 4D array")

    # Validate mode parameter
    valid_modes = {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}")

    # Handle sigma specification
    if isinstance(sigma, (int, float)):
        sigma = (float(sigma),) * 3
    elif isinstance(sigma, (tuple, list)) and len(sigma) == 3:
        sigma = tuple(float(s) for s in sigma)
    else:
        raise ValueError("sigma must be a float or tuple of 3 floats")
    
    if any(s <= 0 for s in sigma):
        raise ValueError("sigma values must be positive")

    # Prepare data
    working_data = input_data.copy()
    if ndim == 3:
        working_data = working_data[..., np.newaxis]
    
    # Validate and prepare mask
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be a numpy array")
        if mask.shape != working_data.shape[:-1]:
            raise ValueError("mask shape must match input_data spatial dimensions")
        mask = mask.astype(bool)
    else:
        # Only create mask if needed
        mask = np.ones(working_data.shape[:-1], dtype=bool) if preserve_mask else None

    # Store original values for masked regions if needed
    masked_values = None
    if preserve_mask and mask is not None:
        masked_indices = ~mask
        masked_values = working_data[masked_indices]

    # Apply mask if needed
    if mask is not None and not preserve_mask:
        working_data = working_data.copy()  # Ensure we don't modify the input
        working_data[~mask] = 0

    # Prepare smoothing parameters
    smooth_sigma = sigma + (0,)  # Add 0 for last dimension
    
    try:
        # Smooth data
        data_smoothed = scipy.ndimage.gaussian_filter(
            working_data, 
            sigma=smooth_sigma,
            mode=mode
        )
        
        # Apply normalization only if we have a mask and it's not all ones
        if mask is not None and not np.all(mask):
            # Calculate normalization factor
            norm_mask = scipy.ndimage.gaussian_filter(
                mask.astype(float), 
                sigma=sigma,
                mode=mode
            )
        
            # Avoid division by zero
            norm_mask[norm_mask < 1e-10] = 1
            
            # Apply normalization
            data_smoothed /= norm_mask[..., np.newaxis]
        
        # Restore original values in masked regions if requested
        if preserve_mask and masked_values is not None:
            data_smoothed[masked_indices] = masked_values
                
    except Exception as e:
        raise RuntimeError(f"Smoothing operation failed: {str(e)}")

    # Return data in original dimensionality
    if ndim == 3:
        data_smoothed = data_smoothed[..., 0]

    return data_smoothed


def roi_describe(data, atlas_data, method='nanmean', key=None, skip_key0=True):
    """
    Get the ROI (Region of Interest) data based on the atlas data.

    Parameters:
    - data: ndarray
        The input data array. If it is a series, the last dimension represents maps.
    - atlas_data: ndarray
        The atlas data array, which should have the same shape as the data or one less dimension if data is a series.
    - method: str, optional (default='nanmean')
        The method to use for summarizing the ROI data. Options are:
        'nanmean', 'nanmedian', 'nanstd', 'nanmax', 'nanmin', 'nansize'.
    - key: array-like, optional
        The key of the ROI. If None, the unique values of the atlas data are used.
    - skip_key0: bool, optional (default=True)
        Whether to skip the key with value 0.

    Returns:
    - tuple: (roi_data, key)
        roi_data: ndarray
            The summarized ROI data.
        key: array-like
            The keys corresponding to the ROIs.
    """

    # Ensure the dimensions of data and atlas_data are compatible
    if np.ndim(data) != np.ndim(atlas_data):
        if (np.ndim(data) != np.ndim(atlas_data) + 1) or (data.shape[:-1] != atlas_data.shape):
            raise ValueError('The data should have the same shape as the atlas data '
                             'or one less dimension if data is a series.')

    atlas_data = atlas_data.astype(int)
    
    if key is None:
        key = np.unique(atlas_data)
        if skip_key0:
            key = key[key != 0]

    methods = {
        'nanmean': np.nanmean,
        'nanmedian': np.nanmedian,
        'nanstd': np.nanstd,
        'nanmax': np.nanmax,
        'nanmin': np.nanmin,
        'nancount': lambda x, axis: np.sum(~np.isnan(x), axis)
    }

    if method not in methods:
        raise ValueError(f"Invalid method '{method}'. Valid methods are: {list(methods.keys())}")

    roi_data = []
    for i in key:
        data_i = data[atlas_data == i]
        if data_i.size == 0:
            roi_data_i = np.full(data.shape[-1], np.nan)
        else:
            roi_data_i = methods[method](data_i, axis=0)
        roi_data.append(roi_data_i)

    roi_data = np.asarray(roi_data)
    return roi_data, key


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


# def convolve_hrf(X, onsets, durations, n_vol, tr, ops=100):
#     """
#     Convolve each X's column iteratively with HRF and align with the timeline of BOLD signal
#     Parameters
#     ----------
#     X : array
#         Shape = n_event or [n_event, n_condition]
#     onsets : array_like
#         In sec. size = n_event
#     durations : array_like
#         In sec. size = n_event
#     n_vol : int
#         The number of volumes of BOLD signal
#     tr : float
#         Repeat time in second
#     ops : int
#         Oversampling number per second, must be one of the (10, 100, 1000)

#     Returns
#     -------
#     X_hrfed : array
#         The result after convolution and alignment
#         shape = n_vol or [n_event, n_vol]
#     """

#     if np.ndim(X) == 1:
#         X = X[..., None]
#     assert np.ndim(X) == 2, 'X must be a 1D or 2D array'

#     assert X.shape[0] == onsets.shape[0] and X.shape[0] == durations.shape[0], \
#         'The length of onsets and durations should be matched with the number of events.'
#     assert ops in (10, 100, 1000), 'Oversampling rate must be one of the (10, 100, 1000)!'

#     # unify the precision
#     decimals = int(np.log10(ops))
#     onsets = np.round(np.asarray(onsets), decimals=decimals)
#     durations = np.round(np.asarray(durations), decimals=decimals)
#     tr = np.round(tr, decimals=decimals)
    
#     assert onsets.min() >= 0, 'The onsets must be non-negative'
#     if onsets.min() > 0:
#         # The earliest event's onset is later than the start point of response.
#         # We supplement it with zero-value event to align with the response.
#         X = np.insert(X, 0, np.zeros(X.shape[1]), 0)
#         durations = np.insert(durations, 0, onsets.min(), 0)
#         onsets = np.insert(onsets, 0, 0, 0)

#     # compute volume acquisition timing
#     vol_t = (np.arange(n_vol) * tr * ops).astype(int)  

#     # # generate hrf kernel
#     hrf = np.array(spm_hrf(tr, oversampling=tr*ops))
#     hrf = hrf[..., None]
#     # hrf = np.load('mytool/hrf.npy') 

#     # do convolution in batches for trade-off between speed and memory
#     batch_size = int(100000 / ops)
#     bat_indices = np.arange(0, X.shape[-1], batch_size)
#     bat_indices = np.r_[bat_indices, X.shape[-1]]

#     X_tc_hrfed = []
#     for idx, bat_idx in enumerate(bat_indices[:-1]):
#         X_bat = X[:, bat_idx:bat_indices[idx+1]]
#         # generate X raw time course
#         X_tc = np.zeros((vol_t.max(), X_bat.shape[-1]), dtype=np.float32)
#         for i, onset in enumerate(onsets):
#             onset_start = int(onset * ops)
#             onset_end = int(onset_start + durations[i] * ops)
#             X_tc[onset_start:onset_end, :] = X_bat[i, :]

#         # convolve X raw time course with hrf kernal
#         fftconvolve(X_tc[:10], hrf[:2])  # to fix weird bug (probably some setting change) caused by running spm_hrf
#         X_tc_hrfed.append(fftconvolve(X_tc, hrf))
#         # print('hrf convolution: sample {0} to {1} finished'.format(bat_idx+1, bat_indices[idx+1]))

#     X_tc_hrfed = np.concatenate(X_tc_hrfed)

#     # downsample to volume frame
#     X_hrfed = X_tc_hrfed[vol_t, :]

#     return X_hrfed

def transform_to_another_space(source_f, ref_f, xfm_list, output_f, 
                               interpolate_method='NearestNeighbor', load_output=False):
    '''Transform dense 3D/4D brain data from one space to another using ANTs.

    Parameters
    ----------
    source_f : str
        Path to the input image file to be transformed
    ref_f : str
        Path to the reference image that defines the output space
    xfm_list : list
        List of transformation files. Each element can be either:
        - str: path to transformation file
        - list/tuple: (path, bool) where bool indicates whether to invert transform
    output_f : str
        Path where the transformed image will be saved
    load_output : bool, optional
        If True, loads and returns the transformed data, by default False

    Returns
    -------
    numpy.ndarray, optional
        Transformed image data if load_output=True

    Notes
    -----
    Uses ANTs' antsApplyTransforms tool to perform the transformation. The function
    assumes ANTs is installed and accessible in the system path.
    '''

    # stitch the xfm_list into a single string
    xfm_str = [f'-t {i} ' for i in xfm_list]
    xfm_str = ''.join(xfm_str)[:-1].replace("'", "")  

    # Apply transformation
    cmd = f'antsApplyTransforms -d 3 -e 3 -i {source_f} -r {ref_f} -o {output_f} {xfm_str} -n {interpolate_method}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Transformation failed: {result.stderr}")

    # Load and return transformed data if requested
    if load_output:
        try:
            data_transformed = load_mri_data(output_f)[0]
            return data_transformed
        except Exception as e:
            raise RuntimeError(f"Failed to load transformed data: {str(e)}")

def transform_sparse_to_another_space(data_sparse, 
                                      dense_mask_current, dense_mask_current_f, dense_mask_ref, dense_mask_ref_f, 
                                      xfm_list, output_f):
    """Transform sparse ROI data from one space to another.
    
    Args:
        data_sparse (np.ndarray): ROI data in 2D space [n_voxel, n_feature] or 1D [n_voxel]
        dense_mask_current (np.ndarray): Dense mask in 3D space
        dense_mask_current_f (str): Path to dense mask file in source space
        dense_mask_ref (np.ndarray): Dense mask in 3D space [n_voxel]
        dense_mask_ref_f (str): Path to dense mask file in target space
        xfm_list (list): List of transformation matrix files and whether to apply inverse transformation
            e.g. [[xfm_path1, 1], [xfm_path2, 0]], where 1 means apply inverse transformation, 0 means apply forward transformation
        temp_f (str): Path for temporary file storage
    
    Returns:
        np.ndarray: Transformed data in target space [n_voxel, n_feature] or [n_voxel]
    """
    # Input validation
    if not isinstance(data_sparse, np.ndarray):
        raise ValueError("data_sparse must be a numpy array")

    # Handle input dimensions
    is_2d = data_sparse.ndim == 2
    if not is_2d:
        if data_sparse.ndim != 1:
            raise ValueError("data_sparse must be 1D or 2D array")
        data_sparse = data_sparse[..., np.newaxis]

    # Convert sparse to dense 3D representation
    data_dense, _ = mytool.core.sparse2dense(data_sparse, dense_mask=dense_mask_current)

    temp_f = output_f.replace('.nii.gz', '_temp.nii.gz')
    save_mri_data(data_dense, temp_f, ref_f=dense_mask_current_f)

    # Transform dense data
    data_transformed = transform_to_another_space(
        source_f=temp_f, 
        ref_f=dense_mask_ref_f,
        xfm_list=xfm_list,
        output_f=output_f, 
        interpolate_method='NearestNeighbor', load_output=True)

    # Convert back to sparse representation
    data_transformed_sparse = data_transformed[dense_mask_ref != 0, :]

    # Return appropriate dimensionality
    if not is_2d:
        data_transformed_sparse = data_transformed_sparse[:, 0]

    return data_transformed_sparse

def spin_permutation_voxel(coords, max_iterations=50):
    """
    Performs spin permutation on voxels in n-dimensional space, preserving spatial frequency.
    The function performs a shift and rotation, but rotations are constrained to n_dim × 2 orientations
    (positive or negative along each dimension).
    
    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape [n_voxel, n_dim] representing voxel coordinates in n-dimensional space
    max_iterations : int, optional
        Maximum number of iterations to attempt matching all voxels, by default 50
    
    Returns
    -------
    numpy.ndarray
        Array of shape [n_voxel] representing new voxel IDs after permutation
    
    Raises
    ------
    ValueError
        If unable to assign all voxels or if resulting permutation is invalid
    """
    n_voxel, n_dim = coords.shape
    coords = coords.astype(int)  # Ensure integer coordinates
    
    # Step 1: Initial transformation
    # Random shift by selecting a new origin
    origin_idx = np.random.randint(0, n_voxel)
    shift_vector = coords[0] - coords[origin_idx]
    shifted_coords = coords + shift_vector
    
    # Random rotation via dimension permutation and sign flips
    dim_permutation = np.random.permutation(n_dim)
    flip_signs = np.random.choice([-1, 1], size=n_dim)
    transformed_coords = shifted_coords[:, dim_permutation] * flip_signs
    
    # Step 2: Initialize tracking sets and mappings
    original_pos_set = {tuple(coord) for coord in coords}
    id_mapping = {}
    remaining_indices = set(range(n_voxel))
    remaining_positions = original_pos_set.copy()
    
    # Step 3: Iterative matching process
    for iteration in range(max_iterations):
        if not remaining_indices:
            break
            
        progress_made = False
        
        # Try direct matches first
        transformed_pos_set = {tuple(coord) for coord in transformed_coords}
        intersection_pos = original_pos_set.intersection(transformed_pos_set)
        
        # Process all possible direct matches
        newly_matched = set()
        for idx in remaining_indices:
            trans_pos = tuple(transformed_coords[idx])
            if trans_pos in intersection_pos and trans_pos in remaining_positions:
                # Find matching original position
                for j, coord in enumerate(coords):
                    if tuple(coord) == trans_pos and j not in id_mapping.values():
                        id_mapping[idx] = j
                        newly_matched.add(idx)
                        remaining_positions.remove(trans_pos)
                        progress_made = True
                        break
        
        remaining_indices -= newly_matched
        
        # If no direct matches found, try a shift
        if not progress_made and remaining_indices:
            remaining_pos_array = np.array(list(remaining_positions))
            remaining_idx_array = np.array(list(remaining_indices))
            remaining_transformed = transformed_coords[remaining_idx_array]
            
            # Find best shift by trying each remaining position as target
            best_shift = None
            max_matches = 0
            
            # Sample a subset of positions to try as shift targets (for efficiency)
            n_samples = min(len(remaining_positions), 10)
            sample_indices = np.random.choice(len(remaining_pos_array), n_samples, replace=False)
            sample_targets = remaining_pos_array[sample_indices]
            
            # Convert remaining_positions to numpy array for faster lookups
            remaining_pos_set = set(map(tuple, remaining_pos_array))
            
            # Use vectorized operations for shift calculations
            base_positions = remaining_transformed[:min(5, len(remaining_transformed))]
            for base_pos in base_positions:
                # Calculate all shift vectors at once
                shift_vectors = sample_targets - base_pos
                
                # Apply each shift vector to all remaining points
                for shift_vector in shift_vectors:
                    shifted = remaining_transformed + shift_vector
                    
                    # Vectorized matching count
                    matches = sum(tuple(coord) in remaining_pos_set for coord in shifted)
                    
                    if matches > max_matches:
                        max_matches = matches
                        best_shift = shift_vector

            # Apply best shift if found
            if best_shift is not None and max_matches > 0:
                transformed_coords[remaining_idx_array] += best_shift
                progress_made = True
        
        if not progress_made:
            break
    
    # Validation
    if len(id_mapping) != n_voxel:
        missing = set(range(n_voxel)) - set(id_mapping.keys())
        print(f"Failed to assign all voxels. Missing {len(missing)} indices: {missing}")
        return None
    
    # Create final permutation array
    permutation = np.array([id_mapping[i] for i in range(n_voxel)])
    
    if len(np.unique(permutation)) != n_voxel:
        duplicates = [i for i in permutation if list(permutation).count(i) > 1]
        print(f"Invalid permutation: contains duplicate IDs: {set(duplicates)}")
        return None
    
    return permutation

# %%
def get_geodesic_dist(surf_f, source_node, target_node):
    # surf_f or faces and vertices must be provided

    if surf_f is not None:
        if surf_f.endswith('.gii'):
            surfp = nib.load(surf_f).agg_data()
        else:
            surfp = nib.freesurfer.read_geometry(surf_f)
        vertices, faces = surfp[0], surfp[1]

    vertices = vertices.astype(np.float64)
    faces = faces.astype(np.int32)

    dists = []
    for nodei in source_node:
        dist = compute_gdist(vertices, 
                            faces, 
                            np.array([nodei], dtype=np.int32),
                            np.array(target_node, dtype=np.int32))
        dists.append(dist)

    return dists


# %%
def func_concate_prep(
    func_data: np.ndarray,
    func_mask: Optional[np.ndarray] = None,
    detrend: bool = True,
    sigma: float = 0.0,
    zscore: bool = True,
    set_unchanged_to_nan: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Prepare functional data by applying preprocessing steps including masking, smoothing, detrending, and normalization.

    Parameters
    ----------
    func_data : np.ndarray
        Functional data with time as the last dimension. Must be 3D or 4D array.
    func_mask : np.ndarray, optional
        Binary mask for the functional data. Must have same spatial dimensions as func_data.
        If None, all voxels are used (default).
    detrend : bool, default=True
        Whether to remove linear trends from the time series.
    sigma : float, default=0.0
        Standard deviation for Gaussian smoothing kernel in voxel units.
        If 0, no smoothing is applied.
    zscore : bool, default=True
        Whether to z-score normalize the time series.
    set_unchanged_to_nan : bool, default=False
        Whether to set voxels with constant values to NaN.

    Returns
    -------
    func_data_masked : np.ndarray, shape = (N, T) if 4D input, (N,) if 3D input
        Preprocessed data for non-zero mask voxels.
        N is number of non-zero voxels and T is number of timepoints.
    func_mask_nonzero: np.ndarray, shape = (N,)
        Boolean mask indicating non-zero voxels.

    Raises
    ------
    ValueError
        If func_data is not 3D or 4D array
        If func_mask shape doesn't match func_data spatial dimensions
        If sigma is negative
    
    Notes
    -----
    Processing steps are applied in this order:
    1. Input validation
    2. Expand dimensions if 3D input
    3. Apply mask
    4. Spatial smoothing (if sigma > 0)
    5. Extract non-zero voxels
    6. Detrending (if detrend=True)
    7. Z-score normalization (if zscore=True)
    8. Set constant voxels to NaN (if set_unchanged_to_nan=True)

    Examples
    --------
    >>> data = np.random.rand(64, 64, 32, 100)  # 4D functional data
    >>> mask = np.ones((64, 64, 32))  # 3D binary mask
    >>> processed_data, nonzero_mask = func_concate_prep(data, mask, sigma=1.5)
    """
    # # Input validation
    # if not (3 <= func_data.ndim <= 4):
    #     raise ValueError("func_data must be 3D or 4D array, got shape {func_data.shape}")
    
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")

    # Ensure 4D data
    ndim = func_data.ndim
    func_data = np.asarray(func_data)
    if ndim == 3 or ndim == 1:
        func_data = func_data[..., np.newaxis]
    
    # Create or validate mask
    if func_mask is None:
        func_mask = np.ones(func_data.shape[:-1], dtype=bool)
    else:
        func_mask = np.asarray(func_mask)
        if func_mask.shape != func_data.shape[:-1]:
            raise ValueError(
                f"func_mask shape {func_mask.shape} must match func_data spatial "
                f"dimensions {func_data.shape[:-1]}"
            )

    # Apply spatial smoothing if requested
    if sigma > 0:
        func_data = mytool.mri.spatial_smooth_3d(
            func_data, 
            mask=func_mask,
            sigma=sigma,
            mode='nearest'
        )

    # Extract non-zero voxels
    func_data_masked = func_data[func_mask != 0, :]

    # Apply detrending
    if detrend:
        func_data_masked = scipy.signal.detrend(func_data_masked, axis=-1)

    # Apply z-score normalization
    if zscore:
        with np.errstate(invalid='ignore'):  # Ignore warning for constant voxels
            func_data_masked = scipy.stats.zscore(func_data_masked, axis=-1, nan_policy='omit')

    # Set constant voxels to NaN
    if set_unchanged_to_nan:
        with np.errstate(invalid='ignore'):
            std_vals = np.nanstd(func_data_masked, axis=-1)
            func_data_masked[std_vals == 0, :] = np.nan

    if ndim == 3 or ndim == 1:
        func_data_masked = func_data_masked[..., 0]

    return func_data_masked, func_mask[func_mask != 0]


def hemilize_atlas(atlas_data, roiinfo, hemi):
    '''
    atlas_data: the atlas data in 3D
    roiinfo: the roi info of the atlas. DataFrame with columns 'hemi' and 'key'
    hemi: the hemisphere to be set to 1

    return: the atlas data with the specified hemisphere being 1, 
            the other hemisphere being 2, 
            and the rest being 0
    '''

    atlas_hemi = roiinfo[roiinfo['hemi']==hemi].reset_index(drop=True)
    atlas_data_hemi = np.zeros(atlas_data.shape)
    atlas_data_hemi[atlas_data!=0] = -1
    atlas_data_hemi[np.isin(atlas_data, atlas_hemi['key'].values)] = 1
    atlas_data_hemi = atlas_data_hemi.astype(int)
    
    return atlas_data_hemi
