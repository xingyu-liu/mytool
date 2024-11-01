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
