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

# %% 
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
