# -*- coding: utf-8 -*-
"""
Simple seismic plotting functions.

:copyright: 2016 Agile Geoscience
:license: Apache 2.0
"""
import numpy as np
import matplotlib.font_manager as fm
from matplotlib import cm
from matplotlib.colors import makeMappingArray
import utils


def watermark_seismic(ax, cfg):
    """
    Add semi-transparent text to the seismic.
    """
    text = cfg['watermark_text']
    xn = cfg['watermark_cols']
    yn = cfg['watermark_rows']

    font = fm.FontProperties()
    font.set_weight(cfg['watermark_weight'])
    font.set_style(cfg['watermark_style'])
    font.set_family(cfg['watermark_family'])
    style = {'size': cfg['watermark_size'],
             'color': cfg['watermark_colour'],
             'alpha': cfg['watermark_alpha'],
             'fontproperties': font
             }
    alignment = {'rotation': cfg['watermark_angle'],
                 'horizontalalignment': 'left',
                 'verticalalignment': 'baseline'
                 }
    params = dict(style, **alignment)

    # Axis ranges.
    xr = ax.get_xticks()
    yr = ax.get_yticks()
    aspect = xr.size / yr.size
    yn = yn or (xn / aspect)
    yn += 1

    # Positions for even lines.
    xpos = np.linspace(xr[0], xr[-1], xn)[:-1]
    ypos = np.linspace(yr[0], yr[-1], yn)[:-1]

    # Intervals.
    xiv = xpos[1] - xpos[0]
    yiv = ypos[1] - ypos[0]

    # Adjust the y positions.
    ypos += yiv / 2

    # Place everything.
    c = False
    for y in ypos:
        for x in xpos:
            if c:
                xi = x + xiv / 2
            else:
                xi = x
            ax.text(xi, y, text, clip_box=ax.clipbox, clip_on=True, **params)
        c = not c

    return ax


def plot_header(head_ax, s, fs, cfg, version=''):
    """
    Plot EBCDIC or ASCII header.
    """
    font = fm.FontProperties()
    font.set_family('monospace')
    font.set_size(fs-1)
    head_ax.axis([0, 40, 41, 0])
    head_ax.text(1, 1,
                 s,
                 ha='left', va='top',
                 fontproperties=font)
    head_ax.set_xticks([])
    head_ax.set_yticks([])
    if cfg['credit']:
        head_ax.text(40, 42,
                     'plot by github.com/agile-geoscience/seisplot v{}'.format(version),
                     size=fs, color='lightgray',
                     ha='right', va='top'
                     )
    return head_ax


def plot_trace_info(trhead_ax, blurb, fs=10):
    """
    Plot trace header info.
    """
    font = fm.FontProperties()
    font.set_family('monospace')
    trhead_ax.axis([0, 40, 0, 40])
    trhead_ax.set_yticks([])
    trhead_ax.set_xticks([])
    trhead_ax.text(20, 20, blurb,
                   ha='center', va='center',
                   rotation=0, fontsize=fs-4,
                   fontproperties=font)
    return trhead_ax


def plot_histogram(ax, data, tickfmt, cfg):
    """
    Plot a histogram of amplitude values.
    """
    fs = cfg['fontsize']
    percentile = cfg['percentile']
    data = np.array(data)
    datamax = np.amax(data)
    datamin = np.amin(data)
    datamean = np.nanmean(data)
    largest = max(datamax, abs(datamin))
    clip_val = np.percentile(data, percentile)
    ax.patch.set_alpha(0.0)
    color = utils.rgb_to_hex(cfg['highlight_colour'])

    y, x, _ = ax.hist(np.ravel(data), bins=int(100.0 / (clip_val / largest)),
                      alpha=0.6, color=color, lw=0)

    ax.set_xlim(-clip_val, clip_val)
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs-4)
    ax.set_xlabel('amplitude', fontsize=fs - 4)
    ax.xaxis.set_label_coords(0.5, -0.12)
    ax.set_ylim([0, y.max()])
    ax.set_yticks(np.linspace(0, y.max(), 6))
    ax.set_ylabel('percentage of samples', fontsize=fs-4)
    ax.tick_params(axis='x', pad=25)
    ax.xaxis.labelpad = 25

    ticks = ax.get_yticks().tolist()  # ytick labels
    percentages = 100*np.array(ticks)/data.size
    labels = []
    for label in percentages:
        labels.append("{:.2f}".format(label))
    ax.set_yticklabels(labels, fontsize=fs - 4)
    ax.xaxis.set_major_formatter(tickfmt)
    if datamax < 1:
        statstring = "\nMinimum: {:.4f}\nMean:    {:.4f}\nMaximum: {:.4f}".format(datamin, datamean, datamax)
        statstring += "\n{:.1f} percentile: {:.4f}".format(percentile, clip_val)
    elif datamax < 10:
        statstring = "\nMinimum: {:.2f}\nMean:    {:.2f}\nMaximum: {:.2f}".format(datamin, datamean, datamax)
        statstring += "\n{:.1f} percentile: {:.2f}".format(percentile, clip_val)
    elif datamax < 100:
        statstring = "\nMinimum: {:.1f}\nMean:    {:.1f}\nMaximum: {:.1f}".format(datamin, datamean, datamax)
        statstring += "\n{:.1f} percentile: {:.1f}".format(percentile, clip_val)
    else:
        statstring = "\nMinimum: {:.0f}\nMean:    {:.0f}\nMaximum: {:.0f}".format(datamin, datamean, datamax)
        statstring += "\n{:.1f} percentile: ±{:.0f}".format(percentile, clip_val)

    ax.text(.98, .95, 'AMPLITUDE HISTOGRAM',
            horizontalalignment='right',
            verticalalignment='top',
            fontweight='bold',
            color=utils.rgb_to_hex(cfg['highlight_colour']),
            transform=ax.transAxes, fontsize=fs-3)
    ax.text(.98, .95, statstring,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes, fontsize=fs-3)
    ax.set_alpha(0)

    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-')
        line.set_alpha(0.2)

    return ax


def plot_colourbar(clr_ax, cmap, data=None, mima=False, plusminus=False, zorder=1):
    """
    Puts a colorbar under / behind histogram
    """
    seisbar = cm.get_cmap(cmap)
    ncolours = 32
    seis_array = makeMappingArray(ncolours, seisbar)
    color_arr = seis_array[np.newaxis, :]
    color_arr = color_arr[:, :, :-1]
    colour_roll = np.rollaxis(color_arr, 1)
    colour_roll2 = np.rollaxis(colour_roll, 1)
    clr_ax.imshow(colour_roll2, extent=[0, ncolours, 0, 1], aspect='auto', zorder=zorder)
    clr_ax.set_yticks([])
    clr_ax.set_xticks([])
    ma, mi = np.amax(data), np.amin(data)
    if mima:
        clr_ax.text(0.95, 0.5, '{:3.0f}'.format(mi),
                    transform=clr_ax.transAxes,
                    horizontalalignment='center', verticalalignment='top',
                    fontsize=8)
        clr_ax.text(0.05, 0.5, '{:3.0f}'.format(ma), transform=clr_ax.transAxes,
                    horizontalalignment='center',
                    fontsize=8)
    if plusminus:
        clr_ax.text(0.95, 0.5, "+",
                    transform=clr_ax.transAxes,
                    ha='right', color='w',
                    va='center', fontsize=16)
        clr_ax.text(0.05, 0.5, "-",
                    transform=clr_ax.transAxes, color='k',
                    ha='left', va='center', fontsize=16)

    return clr_ax


def plot_title(ax, text, fs, cfg):
    """
    Add a title.
    """
    color = utils.rgb_to_hex(cfg['highlight_colour'])
    x = 1.0 if cfg['sidelabel'] == 'right' else 0.0
    ax.text(x, 0.0, text, size=fs,
            ha=cfg['sidelabel'],
            va='bottom',
            color=color,
            )
    ax.axis('off')
    return ax


def plot_subtitle(ax, text, fs, cfg):
    """
    Add a title.
    """
    color = utils.rgb_to_hex(cfg['highlight_colour'])
    x = 1.0 if cfg['sidelabel'] == 'right' else 0.0
    ax.text(x, -0.15, text, size=fs,
            ha=cfg['sidelabel'],
            va='top',
            color=color,
            )
    ax.axis('off')
    return ax
