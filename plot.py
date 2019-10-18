# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def rdm(data, label=None, fig_size=None, title=None, vmin=None, vmax=None,
        delete_diag=None, show_value=None, colormap=None):

    if delete_diag is not None:
        data[range(np.size(data, 0)), range(np.size(data, 0))] = 0

    if fig_size is not None:
        fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
    else:
        fig = plt.figure(figsize=(5, 5))

    if title is not None:
        plt.title(title)

    if colormap is not None:
        cmap = plt.cm.get_cmap(colormap)
    else:
        cmap = plt.cm.coolwarm

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.02)

    if label is not None:
        # label them with the respective list entries
        plt.xticks(np.arange(np.size(data, 1)), label, rotation=90)
        plt.yticks(np.arange(np.size(data, 0)), label)

    if show_value is not None:
        valfmt = matplotlib.ticker.StrMethodFormatter('{x:.0f}')
        for i in range(np.size(data, 0)):
            for j in range(np.size(data, 1)):
                plt.text(j, i, valfmt(data[i, j]),
                         ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()


def sub_plot(x, nrows, ncols, sharex=False, sharey=False, 
             plot_type='im', vmin=None, vmax=None,
             row_label=None, title=None, cmap='coolwarm',
             show=True, subplot_kw=None, plot_type_kw=None):
    """

    Parameters
    ----------
        x[list]: list of image_array
    """

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            sharex=sharex, sharey=sharey,
                            subplot_kw=subplot_kw,
                            figsize=[10, 6])

    for i, ax in enumerate(axs.flat[:len(x)]):
        if row_label is not None:
            if np.mod(i, ncols) == 0:
                ax.set_ylabel(row_label[i//ncols])
        
        if plot_type == 'im':
            ax.imshow(x[i], cmap=cmap, vmin=vmin, vmax=vmax, **plot_type_kw)
        elif plot_type == 'plot':
            ax.plot(x[i], vmin=vmin, vmax=vmax, **plot_type_kw)
        elif plot_type == 'hist':
            ax.hist(x[i], **plot_type_kw)
        elif plot_type == 'scatter':
            ax.scatter(x[i][:, 0], x[i][:, 1], **plot_type_kw)
        elif plot_type == 'semilogy':
            ax.semilogy(x[i])
            
        if title is not None:
            ax.set_title(str(title[i]))
#        plt.colorbar(im, ax=ax, fraction=0.02)
    plt.tight_layout()
    if show is True:
        plt.show()


def scatter_bar(matrix, x_bar=None, y_bar=None):
    """

    Parameters
    ----------
        matrix: binary
    """

    matrix = matrix.T
    scatter_data = np.where(matrix == 1)
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
        ax_histx.hist(x, bins=range(0, matrix.shape[0], binwidth))

    if y_bar is not None:
        binwidth = 1
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)
        ax_histy.set_ylim(ax_scatter.get_ylim())
        ax_histy.hist(y, bins=range(0, matrix.shape[-1], binwidth),
                      orientation='horizontal')

    plt.show()


def gradient_color_lines(data, x=None, linestyle='-', label=None, ref=None,
                         coloarmap='viridis', marker='o', markersize=3):

    """

    Parameters
    ----------
        data: list of array-like x and y, only y if x is provided
    """
    fig, ax = plt.subplots()

#    cmap = plt.cm.get_cmap(coloarmap)
#    color_norm = plt.Normalize(0,len(data))
#    color = cmap(color_norm(range(len(data))))

    sns.set_palette(coloarmap, n_colors=len(data))

    if x is None:
        for series in data:
            ax.plot(series[0], series[1], linestyle=linestyle,
                    marker=marker, markersize=markersize)
        if ref is not None:
            ax.plot(data[ref][0], data[ref][1], linestyle=linestyle,
                    c='tab:red', marker=marker, markersize=markersize)
    else:
        for series in data:
            ax.plot(x, series, linestyle=linestyle,
                    marker=marker, markersize=markersize)
        if ref is not None:
            ax.plot(x, data[ref], linestyle=linestyle, c='tab:red',
                    marker=marker, markersize=markersize)

    if label is not None:
        ax.legend(label)

    plt.show()


def gradient_color_hist(data, bin_num=10, fit=False, show_range=None,
                        density=False, label=None,
                        coloarmap='viridis', histtype='bar',
                        rug=True, hist=False):

    """

    Parameters
    ----------
        data: list of 1-d data
        weights : the same shape with data or str 'percentage'
    """

    fig, ax = plt.subplots()
    sns.set_palette(coloarmap, n_colors=len(data))

    if fit is not True:
        ax.hist(data, bins=bin_num, range=show_range,
                density=density, histtype=histtype)
    else:
        for series in data:
            sns.distplot(series, ax=ax,
                         rug=rug, hist=hist, bins=bin_num)

    if label is not None:
        ax.legend(label)
    plt.show()
