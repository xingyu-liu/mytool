# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly
import pandas as pd

# %%

def plotly_to_mpl(cmap_plotly):
    colors = []
    for color in cmap_plotly:
        # Convert 'rgb(r, g, b)' to (r/255, g/255, b/255)
        color = color.replace('rgb(', '').replace(')', '')
        r, g, b = map(int, color.split(','))
        colors.append((r / 255, g / 255, b / 255))

    cmap_plt = LinearSegmentedColormap.from_list('', colors)

    return cmap_plt


# get_rgba
def get_rgba(data, cmap='jet', clip_pct=[0.05, 0.95], nonzero=False):

    if nonzero is True:
        norm = matplotlib.colors.Normalize(vmin=np.quantile(data[data!=0], clip_pct[0]), vmax=np.quantile(data[data!=0], clip_pct[1]), clip=True)
    else:
        norm = matplotlib.colors.Normalize(vmin=np.quantile(data, clip_pct[0]), vmax=np.quantile(data, clip_pct[1]), clip=True)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

    color = np.asarray(mapper.to_rgba(data.reshape(-1), bytes=True))/255
    color = color.reshape(np.r_[data.shape,4])
    
    return color


def cmyk_to_rgb(cmyk, cmyk_scale=1, rgb_scale=1):
    rgb = [rgb_scale * (1 - cmyk[i] / float(cmyk_scale)) * (1 - cmyk[-1] / float(cmyk_scale)) for i in range(3)]
    return np.array(rgb)


def blend_cmap(x, y, cmap1='Purples', cmap2='Greens', norm_xy=True):
    if norm_xy:
        # Normalize x and y to the range [0, 1]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # Get colors from the two color maps
    colors1 = np.array(plt.get_cmap(cmap1)(x))
    colors2 = np.array(plt.get_cmap(cmap2)(y))

    # Blend the colors by averaging the RGB components
    blended_colors = (colors1[:, :3] + colors2[:, :3]) / 2
    blended_colors = np.clip(blended_colors, 0.001, 0.999)
    blended_cmap = LinearSegmentedColormap.from_list('blended', blended_colors, N=len(blended_colors))

    return blended_colors, blended_cmap

# 
def get_cmap_atlas(atlas, hemi='rh', info_f=None):
    if atlas not in ['MMP', 'RSN7nw', 'RSN17nw', 'mesulam']:
        raise ValueError('atlas should be either MMP or RSN7nw')
    
    if atlas == 'MMP':
        if info_f is None:
            info_f = '/home/star/github/atlas/MMP/atlas-MMP_roiinfo.txt'
        atlas_info = pd.read_csv(info_f, sep='\t')

        colors = [tuple(int(i)/255 for i in j[1:-1].split()) for j in 
            atlas_info.loc[atlas_info['hemi']==hemi, 'color'].values]
        
    if atlas in ['RSN7nw', 'RSN17nw']:
        if info_f is None:
            info_f = f'/home/star/github/atlas/RSN/atlas-{atlas}_roiinfo.txt'
        atlas_info = pd.read_csv(info_f, sep='\t')
    
        colors = [tuple(int(i)/255 for i in j[1:-1].split()) for j in 
            atlas_info['color'].values]
    
    if atlas == 'mesulam':
        if info_f is None:
            info_f = '/home/star/github/atlas/mesulam/atlas-mesulam_roiinfo.txt'
        atlas_info = pd.read_csv(info_f, sep='\t')
        colors = [tuple(int(i)/255 for i in j[1:-1].split()) for j in 
            atlas_info['color'].values]

    # repalce the above with np.clip
    colors = np.array([tuple(np.clip(i, 0.001, 0.999) for i in j) for j in colors])

    cmap = plt.cm.colors.ListedColormap(colors)

    return cmap, colors, atlas_info


def get_symetric_vbound(data, percentiles=[2, 98]):
    vbound = np.max(np.abs([np.nanpercentile(data, percentiles)]))
    return [-vbound, vbound]

# plot 3d using plt
def scatter3d_plt(data_vec, loc, hemi, loc4each=False, mask_underlay=False,
                   nrows=1, ncols=1, cmap='magma', perspective='default', 
                   ms=5, vmin=None, vmax=None, vcenter=None, colorbar=True, figsize=None):
    '''
    Plot 3D scatter plots of data vectors at specified locations.

    Parameters:
    -----------
    data_vec : list of 1d arrays
        The data values to be plotted for each point
    loc : array-like
        If loc4each=False: shape=(3, n_features) for shared locations
        If loc4each=True: shape=(n_samples, 3, n_features) for individual locations
    hemi : str
        Hemisphere ('lh' or 'rh') for view orientation
    loc4each : bool
        Whether each data vector has its own set of locations
    mask_underlay : bool
        Whether to plot an underlay mask in silver
    nrows, ncols : int
        Grid layout for multiple plots
    cmap : str
        Colormap name for data visualization
    perspective : str or list
        View perspective ('default', 'axial', 'coronal', 'sagittal' or [elev, azim])
    ms : float
        Marker size for scatter points
    vmin, vmax : float
        Value range for color mapping
    vcenter : float
        Center value for diverging colormaps
    colorbar : bool
        Whether to show the colorbar
    figsize : tuple
        Figure size in inches (width, height)
    '''
    if figsize is None: 
        figsize = (ncols*2, nrows*1.8)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                            subplot_kw={'projection': '3d'})
    if nrows == 1 and ncols == 1: 
        axes = np.array([axes])

    for i, ax in enumerate(axes.flatten()):
        if i >= len(data_vec):
            ax.axis('off')
            continue
        
        datai = data_vec[i]
        loci = loc if not loc4each else loc[i]
        
        # Plot the underlay mask
        if mask_underlay:
            ax.scatter(loci[0], loci[1], loci[2], c='silver', edgecolor='none', 
                      s=ms, alpha=0.3, depthshade=False)
        
        # Compute value range for color mapping
        if vmax is None:
            vmaxi = np.nanpercentile(datai, 98)
        else:
            vmaxi = vmax
        if vmin is None:
            vmini = np.nanpercentile(datai, 2)
        else:
            vmini = vmin
        if vcenter is not None:
            vrange = np.nanmax(np.abs([vcenter - vmini, vmaxi - vcenter]))
            vmaxi = vcenter + vrange
            vmini = vcenter - vrange
        
        # Plot scalar data
        scatter = ax.scatter(loci[0], loci[1], loci[2], c=datai, cmap=cmap, 
                           edgecolor='none', vmin=vmini, vmax=vmaxi, s=ms, 
                           alpha=1, depthshade=False)
        
        if colorbar:
            plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                        pad=0.1, shrink=0.8)
        
        # Set view and eye
        adjust_3dmri_plot(fig, np.array([ax]), hemi, perspective=perspective, 
                         loc=loci, loc4each=False, crop=True, square=True)

    fig.subplots_adjust(wspace=0, hspace=0.1)

    return fig, axes


def adjust_3dmri_plot(fig, axes, hemi, perspective='default', grid=False, 
                      loc=None, loc4each=False, ax_visible=True, crop=True, square=True):
    
    for i, ax in enumerate(axes.flatten()):

        # set view and eye
        if perspective == 'default':
           ax.view_init(elev=30, azim=-60 if hemi == 'lh' else -120)
        elif perspective == 'axial':
            ax.view_init(elev=90 if hemi == 'lh' else -90, azim=270 if hemi == 'lh' else -270)
        elif perspective == 'coronal':
            ax.view_init(elev=0, azim=270 if hemi == 'lh' else 90)
        elif perspective == 'sagittal':
            ax.view_init(elev=0, azim=180 if hemi == 'lh' else 0)
        elif type(perspective) == list:
            ax.view_init(elev=perspective[0], azim=perspective[1])
        
        if loc is not None:
            loci = loc if not loc4each else loc[i]

            if crop:
                # set x, y, z limit to the valid region
                ax.set_xlim([np.nanmin(loci[0])-1, np.nanmax(loci[0])+1]) 
                ax.set_ylim([np.nanmin(loci[1])-1, np.nanmax(loci[1])+1])
                ax.set_zlim([np.nanmin(loci[2])-1, np.nanmax(loci[2])+1])
            
            if square:
                # set x, y,z scale according to their range
                ax.set_box_aspect([np.ptp(i[~np.isnan(i)]+2) for i in loci])

            if not grid:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

        if ax_visible:
            ax.axis('on')
        else:
            ax.axis('off')
            continue

    fig.subplots_adjust(wspace=0, hspace=0.1)

    return fig, axes
