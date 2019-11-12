#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
# import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns


class Dnn_act:
    """ dnn activation

    Attibutes
    ---------

        data: array_like,
            shape = [n_stim,n_chn,n_unit_in_row,n_unit_in_column]

    """

    def __init__(self, data, stim_per_cat):

        if data.ndim == 2 or data.ndim == 3:
            self.data = data
        elif data.ndim == 4:
            self.data = data.reshape([data.shape[0], data.shape[1], -1])
        else:
            print('the shape of data has to be 2/3/4-d')
        self.stim_num = np.shape(self.data)[0]
        self.stim_per_cat = stim_per_cat
        self.cat_num = int(self.stim_num / self.stim_per_cat)
        self.chn_num = np.shape(self.data)[1]
        self.relued = False

    def relu(self):
        self.data[self.data < 0] = 0
        self.relued = True

    def chn_act(self, top_n=5, replace=False):
        unit_max_act = np.sort(self.data, -1)[:, :, -1*top_n:]
        if replace is True:
            self.data = unit_max_act.mean(-1)
        else:
            return unit_max_act.mean(-1)

    def cat_mean_act(self):
        cur_shape = np.shape(self.data)
        new_shape = np.r_[self.cat_num, self.stim_per_cat, cur_shape[1:]]
        cat_data = self.data.reshape(new_shape)
        return cat_data.mean(1), cat_data.std(1)


class Chn_score:
    """ channel value

    Attibutes
    ---------

        data: array_like,
            shape = [n_cat,n_chn]

    """

    def __init__(self, data, chn_index=None):

        self.data = data
        self.cat_num = np.shape(self.data)[0]
        self.chn_num = np.shape(self.data)[1]
        if chn_index is None:
            chn_index = np.arange(self.chn_num)
        self.chn_index = chn_index

    def sort_chn(self, chn_index):
        self.reset_chn_index()
        self.data = self.data[:, chn_index]
        self.chn_index = chn_index

    def reset_chn_index(self):
        self.data = self.data[:, np.argsort(self.chn_index)]
        self.chn_index = np.arange(self.chn_num)


# glm
def generate_X(num_ft, samp_per_ft, target_ft=None):

    """generate dummy coding feature matrix

    """

    par = np.zeros([num_ft*samp_per_ft, num_ft])

    if target_ft is not None:
        par[target_ft*samp_per_ft:(target_ft+1)*samp_per_ft,
            target_ft] = num_ft - 1
        other_ft = np.delete(np.arange(num_ft), target_ft)
        for ft in other_ft:
            par[ft*samp_per_ft:(ft+1)*samp_per_ft, ft] = -1
    else:
        for ft in range(num_ft):
            par[ft*samp_per_ft:(ft+1)*samp_per_ft, ft] = 1
    return par


def glm_model(y, x, target_ft=None):

    """build glm models

    Parameters
    ----------
        y: 2-d array, [n_samples, n_chns]
        x: 2-d array, [n_samples, n_features]

    """

    t_pvalue = []
    t_value = []

    if target_ft is not None:
        for chn in range(y.shape[-1]):
            results = smf.OLS(y[:, chn], x).fit()
            t_pvalue.append(results.pvalues[target_ft+1])
            t_value.append(results.tvalues[target_ft+1])
            print('chn {0} done'.format(chn+1))
    else:
        for chn in range(y.shape[-1]):
            results = smf.OLS(y[:, chn], x).fit()
            t_pvalue.append(np.nan_to_num(results.pvalues))
            t_value.append(np.nan_to_num(results.tvalues))
            print('chn {0} done'.format(chn+1))

    t_pvalue = np.asarray(t_pvalue).T
    t_value = np.asarray(t_value).T

    return t_value, t_pvalue


# plot
def plot_chn_cat(data, ref=None, line_cls=None, label=None,
                 linestyle='-', markersize=1, refmarkersize=10):

    """
    Parameters
    ---------

        data: shape = [n_point,n_line]
        ref: highlight one line,type=int
        cat_cls: summary line into higher level class

    """

    fig, ax = plt.subplots()

    color = ['limegreen', 'orange', 'cornflowerblue', 'silver', 'lightcoral',
             'hotpink', 'blueviolet', 'gold']

    if ref is None:
        im_data = data
    else:
        im_data = np.delete(data, ref, -1)

    if line_cls is None:
        if im_data.shape[-1] <= 8:
            ax.set_color_cycle(color)
            ax.plot(im_data, linestyle=linestyle, marker='o', ms=markersize)
        else:
            ax.plot(im_data, linestyle=linestyle, marker='o', ms=markersize)
    else:
        if len(line_cls) > 8:
            print('no more than 7 cat_cls')
        for i, line in enumerate(line_cls):
            ax.plot(im_data[:, line], c=color[i],
                    linestyle=linestyle, marker='o', ms=markersize)

    if ref is not None:
        ax.plot(data[:, ref], linestyle=linestyle, marker='o',
                ms=refmarkersize,
                markeredgecolor='tab:red', markerfacecolor='None')

    if label is not None:
        ax.legend(label)
    plt.show()


def plot_lines_layers(data, x=None, label=None,colormap='rainbow', 
                      subplot_kw=None,
                      plot_kw={'linestyle':'-', 'marker':'o', 'markersize':3}):
    """

    Parameters
    ----------
        data: list of array-like x and y
            len(data) = 8, first 5 is conv layer, last 3 is fc layer
    """
    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw=subplot_kw)

    if len(data) <= 6 or len(data) > 7:
        sns.set_palette(colormap, n_colors=len(data))
        for series in data:
            if x is None:
                ax.plot(series[0], series[1], **plot_kw)
            else:
                ax.plot(x, series, **plot_kw)
    elif len(data) == 7:
        cmap = plt.cm.get_cmap('Blues')
        color_norm = plt.Normalize(0, 7)
        color_conv = cmap(color_norm(range(7)))[-5:, :]

        cmap = plt.cm.get_cmap('Oranges')
        color_norm = plt.Normalize(0, 4)
        color_fc = cmap(color_norm(range(4)))[-3:-1, :]

        color = np.r_[color_conv, color_fc]

        for i, series in enumerate(data):
            if x is None:
                ax.plot(series[0], series[1], color=color[i], **plot_kw)
            else:
                ax.plot(x, series, color=color[i], **plot_kw)
    
#    ax.invert_xaxis()

    ax.spines['top'].set_visible(False) 
#    ax.spines['bottom'].set_visible(False)
#    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    
    if label is not None:
        ax.legend(label)
    plt.show()


def plot_hist_layers(data, bin_num=20, fit=False, show_range=None,
                     density=False, label=None, colormap='rainbow',
                     histtype='bar', rug=True, hist=False, subplot_kw=None):
    """

    Parameters
    ----------
        data: list of 1-d data
        weights : the same shape with data or str 'percentage'
    """
    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw=subplot_kw)

    if len(data) <= 6 or len(data) > 7:
        sns.set_palette(colormap, n_colors=len(data))
        if fit is not True:
            ax.hist(data, bins=bin_num, range=show_range,
                    density=density, histtype=histtype)
        else:
            for series in data:
                sns.distplot(series, ax=ax, rug=rug, hist=hist, bins=bin_num)

    elif len(data) == 7:
        cmap = plt.cm.get_cmap('Blues')
        color_norm = plt.Normalize(0, 7)
        color_conv = cmap(color_norm(range(7)))[-5:, :]

        cmap = plt.cm.get_cmap('Oranges')
        color_norm = plt.Normalize(0, 4)
        color_fc = cmap(color_norm(range(4)))[-3:-1, :]

        color = np.r_[color_conv, color_fc]

        if fit is not True:
            ax.hist(data, bins=bin_num, range=show_range, color=color,
                    density=density, histtype=histtype)
        else:
            if show_range is not None:
                ax.set_xlim(show_range[0], show_range[1])
            for i, series in enumerate(data):
                sns.distplot(series, ax=ax, color=list(color[i]),
                             rug=rug, hist=hist, bins=bin_num)
#    ax.invert_xaxis()

    ax.spines['top'].set_visible(False) 
#    ax.spines['bottom'].set_visible(False)
#    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if label is not None:
        ax.legend(label)

    plt.show()
