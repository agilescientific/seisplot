#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Simple seismic plotter.

:copyright: 2016 Agile Geoscience
:license: Apache 2.0
"""
# Import standard libraries.
import argparse
import os
import time

# Import 3rd party.
import yaml
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
from matplotlib import cm
from matplotlib.colors import makeMappingArray

from obspy.segy.segy import readSEGY

# Import our stuff.
from notice import Notice
from utils import add_subplot_axes


#####################################################################
#
# VARIOUS PLOTTING FUNCTIONS
#
#####################################################################

def wiggle_plot(ax, data, tbase, ntraces,
                skip=1,
                perc=98.0,
                gain=1.0,
                rgb=(0, 0, 0),
                alpha=0.5,
                lw=0.2):
    """
    Plots wiggle traces of seismic data. Skip=1, every trace, skip=2, every
    second trace, etc.
    """
    rgba = list(rgb) + [alpha]
    sc = np.percentile(data, perc)  # Normalization factor
    wigdata = data[:, ::skip]
    xpos = np.arange(ntraces)[::skip]

    for x, trace in zip(xpos, np.transpose(wigdata)):
        # Compute high resolution trace for aesthetics.
        amp = gain * trace / sc + x
        hypertime = np.linspace(tbase[0], tbase[-1], (10 * tbase.size - 1) + 1)
        hyperamp = np.interp(hypertime, tbase, amp)

        # Plot the line, then the fill.
        ax.plot(hyperamp, hypertime, 'k', lw=lw)
        ax.fill_betweenx(hypertime, hyperamp, x,
                         where=hyperamp > x,
                         facecolor=rgba,
                         lw=0,
                         )
    return ax


def decorate_seismic(ax, ntraces, tickfmt, cfg):
    """
    Add various things to the seismic plot.
    """
    fs = cfg['fontsize']
    # ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlim([0, ntraces])
    ax.set_ylabel('Two-way time [ms]', fontsize=fs - 2)
    ax.set_xlabel('Trace no.', fontsize=fs - 2, horizontalalignment='center')
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs - 2)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs - 2)
    ax.xaxis.set_major_formatter(tickfmt)
    ax.yaxis.set_major_formatter(tickfmt)
    if cfg['seis_grid'] == 'on':
        pass
        # ax.grid()
    if cfg['seis_grid'] == 'off':
        pass
        # ax.grid('off')
    return ax


def plot_colourbar(fig, ax, im, data, mima=False, fs=10):
    """
    Put a small colourbar right in the seismic data area.
    """
    ma, mi = np.amax(data), np.amin(data)
    colorbar_ax = add_subplot_axes(ax, [0.975, 0.025, 0.01, 0.1])
    fig.colorbar(im, cax=colorbar_ax)
    if mima:
        colorbar_ax.text(0.5, -0.1, '%3.0f' % mi,
                         transform=colorbar_ax.transAxes,
                         horizontalalignment='center', verticalalignment='top',
                         fontsize=fs - 3)
        colorbar_ax.text(0.5, 1.1, '%3.0f' % ma,
                         transform=colorbar_ax.transAxes,
                         horizontalalignment='center',
                         fontsize=fs - 3)
    else:
        colorbar_ax.text(0.5, 0.9, "+",
                         transform=colorbar_ax.transAxes,
                         ha='center', color='white',
                         va='center', fontsize=12)
        colorbar_ax.text(0.5, 0.15, "-",
                         transform=colorbar_ax.transAxes, color='k',
                         ha='center', va='center', fontsize=16)
    colorbar_ax.set_axis_off()
    return fig


def plot_spectrum(spec_ax, data, dt, tickfmt, trace=10, fs=10):
    """
    Plot a power spectrum.
    w is window length for smoothing filter
    """
    S = abs(fft(data[:, trace]))
    faxis = np.fft.fftfreq(len(data[:, 1]), d=dt / 10 ** 6)
    x = faxis[:len(faxis) // 4]
    y = np.sqrt(S[0:len(faxis) // 4])
    spec_ax.patch.set_alpha(0.5)
    spec_ax.fill_between(x, y, 0, lw=0, facecolor='k', alpha=0.5)
    spec_ax.set_xlabel('frequency [Hz]', fontsize=fs - 4)
    spec_ax.xaxis.set_label_coords(0.5, -0.12)
    spec_ax.set_xlim([0, x.max()])
    spec_ax.set_xticklabels(spec_ax.get_xticks(), fontsize=fs - 4)
    spec_ax.set_yticklabels(spec_ax.get_yticks(), fontsize=fs - 4)
    spec_ax.set_ylabel('amplitude', fontsize=fs - 4)
    spec_ax.text(.98, .9, 'Amplitude spectrum',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=spec_ax.transAxes, fontsize=fs - 3)
    spec_ax.yaxis.set_major_formatter(tickfmt)
    spec_ax.xaxis.set_major_formatter(tickfmt)
    spec_ax.grid('on')

    return spec_ax


def chunk(string, width=80):
    """
    Chunks a long string into lines `width` characters long, default 80 chars.
    If the string is not a multiple of 80 chars, the end is padded with spaces.
    """
    lines = int(np.ceil(len(string) / width))
    result = ''
    for i in range(lines):
        line = string[i*width:i*width+width]
        result += line + (width-len(line))*' ' + '\n'
    return result


def plot_header(head_ax, s, fs):
    """
    Plot EBCIDIC header.
    """
    font = fm.FontProperties()
    font.set_family('monospace')
    font.set_size(fs-2)
    head_ax.axis([0, 40, 0, 41])
    head_ax.text(1, 40 - 1,
                 chunk(s),
                 ha='left', va='top',
                 fontproperties=font)
    head_ax.set_xticks([])
    head_ax.set_yticks([])

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
    trhead_ax.text(20, 20, blurb, ha='center', va='center', rotation=0, fontsize=fs-4, fontproperties=font)
    return trhead_ax


def plot_histogram(hist_ax, data, tickfmt, fs=10, zorder=2):
    """
    Plot a histogram of amplitude values.
    """
    largest = max(np.amax(data), abs(np.amin(data)))
    clip_val = np.percentile(data, 99.0)
    hist_ax.patch.set_alpha(0.0)
    y, x, _ = hist_ax.hist(np.ravel(data), bins=int(100.0 / (clip_val / largest)), 
                           alpha=1.0, color='#777777', lw=0, zorder=2)

    hist_ax.set_xlim(-clip_val, clip_val)
    hist_ax.set_xticklabels(hist_ax.get_xticks(), fontsize=fs - 4)
    hist_ax.set_xlabel('amplitude', fontsize=fs - 4)
    hist_ax.xaxis.set_label_coords(0.5, -0.12)
    hist_ax.set_ylim([0, y.max()])
    hist_ax.set_yticks(np.linspace(0, y.max(), 6))
    a = hist_ax.get_yticks().tolist()  # ytick labels
    a[0], a[1], a[2], a[3], a[4], a[5] = '0', '20', '40', '60', '80', '100'
    hist_ax.set_yticklabels(a, fontsize=fs - 4)
    hist_ax.xaxis.set_major_formatter(tickfmt)
    hist_ax.text(.98, .9, 'Histogram',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=hist_ax.transAxes, fontsize=fs - 3)
    hist_ax.set_alpha(0)
    hist_ax.grid()
    return hist_ax


def plot_hrz_colorbar(clr_ax, cmap, data=None, mima=False, plusminus=False, zorder=1):
    """
    Puts a horizontal colorbar under / behind histogram
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
    # clr_ax.set_axis_off()

    return clr_ax


def plot_title(title_ax, title, fs):
    """
    Add a title.
    """
    title_ax.text(1.0, 0.0, title, size=fs,
                  horizontalalignment='right',
                  verticalalignment='bottom')
    title_ax.axis('off')
    return title_ax


def main(target, cfg):
    """
    Puts everything together.
    """
    t0 = time.time()

    # Read the file.
    section = readSEGY(target, unpack_headers=True)

    # Calculate some things.
    # NB Getting nsamples and dt from the first trace assumes that all
    # traces are the same length, which is not a safe assumption in SEGY v2.
    nsamples = section.traces[0].header.number_of_samples_in_this_trace
    dt = section.traces[0].header.sample_interval_in_ms_for_this_trace
    ntraces = len(section.traces)
    tbase = 0.001 * np.arange(0, nsamples * dt, dt)
    tstart = 0
    tend = np.amax(tbase)
    wsd = ntraces / cfg['tpi']

    # Make the data array.
    data = np.vstack([t.data for t in section.traces]).T

    # Collect some other data. Use a for loop because there are several.
    elev, esp, ens, tsq = [], [], [], []
    for i, trace in enumerate(section.traces):
        elev.append(trace.header.receiver_group_elevation)
        esp.append(trace.header.energy_source_point_number)
        ens.append(trace.header.ensemble_number)
        tsq.append(trace.header.trace_sequence_number_within_line)

    clip_val = np.percentile(data, 99.0)

    # Notify user of parameters
    Notice.info("n_traces   {}".format(ntraces))
    Notice.info("n_samples  {}".format(nsamples))
    Notice.info("dt         {}".format(dt))
    Notice.info("t_start    {}".format(tstart))
    Notice.info("t_end      {}".format(tend))
    Notice.info("max_val    {}".format(np.amax(data)))
    Notice.info("min_val    {}".format(np.amin(data)))
    Notice.info("clip_val   {}".format(clip_val))

    t1 = time.time()
    Notice.ok("Read data successfully in {:.1f} s".format(t1-t0))

    #####################################################################
    #
    # MAKE PLOT
    #
    #####################################################################
    Notice.hr_header("Plotting")

    ##################################
    # Plot size parameters
    # Some constants
    wsl = 6  # Width of sidelabel
    mih = 10  # Minimum plot height
    fhh = 5  # File header height
    m = 0.5  # margin in inches

    # Margins, CSS like
    mt, mb, ml, mr = m, m, 2 * m, 2 * m
    mm = mr / 2  # padded margin between seismic and label

    # Width is determined by tpi, plus a constant for the sidelabel, plus 1 in
    w = ml + wsd + wsl + mr + mm

    # Height is given by ips, but with a minimum of 8 inches, plus 1 in
    h = max(mih, cfg['ips'] * (np.amax(tbase) - np.amin(tbase)) / 1000 + mt + mb)

    # More settings
    ssl = (ml + wsd + mm) / w  # Start of side label (ratio)
    fs = cfg['fontsize']

    Notice.info("Width of plot   {} in".format(w))
    Notice.info("Height of plot  {} in".format(h))

    ##################################
    # Make the figure.
    fig = plt.figure(figsize=(w, h), facecolor='w')
    ax = fig.add_axes([ml / w, mb / h, wsd / w, (h - mb - mt) / h])

    # make parasitic axes for labeling CDP number
    par1 = ax.twiny()
    par1.spines["top"].set_position(("axes", 1.0))
    tickfmt = mtick.FormatStrFormatter('%.0f')
    par1.plot(ens, np.zeros_like(ens))
    par1.set_xlabel("CDP number", fontsize=fs-2)
    par1.set_xticklabels(par1.get_xticks(), fontsize=fs-2)
    par1.xaxis.set_major_formatter(tickfmt)

    # Plot title
    title_ax = fig.add_axes([ssl, 1-mt/h, wsl/w, mt/(2*h)])
    title_ax = plot_title(title_ax, target, fs=fs)

    # Plot text header.
    s = section.textual_file_header.decode()
    start = (h - mt - fhh) / h
    head_ax = fig.add_axes([ssl, start, wsl/w, fhh/h])
    head_ax = plot_header(head_ax, s, fs=fs-1)

    # Params for histogram plot
    pady, padx = 0.1, 0.25 * wsl / w
    cstrip = 0.025  # color_strip height
    charty = 0.125  # height of chart
    xhist = (ssl + padx)
    whist = (1 - ssl - (mr/w)) - 2 * padx

    if cfg['display'].lower() == 'varden':
        im = ax.imshow(data,
                       cmap=cfg['cmap'],
                       clim=[-clip_val, clip_val],
                       extent=[0, ntraces, tbase[-1], tbase[0]],
                       aspect='auto'
                       )

    elif cfg['display'].lower in ['vd', 'varden', 'variable']:
        ax = wiggle_plot(ax,
                         data,
                         tbase,
                         ntraces,
                         skip=cfg['skip'],
                         gain=cfg['gain'],
                         rgb=cfg['colour'],
                         alpha=cfg['opacity'],
                         lw=cfg['lineweight']
                         )
        ax.set_ylim(ax.get_ylim()[::-1])

        # Plot colourbar under histogram
        clrbar_ax = fig.add_axes([xhist, 1.5 * mb/h + charty + pady - cstrip, whist, cstrip])
        plot_hrz_colorbar(clrbar_ax, cmap=cfg['cmap'])

    elif cfg['display'].lower() == 'both':
        # variable density goes on first
        im = ax.imshow(data,
                       cmap=cfg['cmap'],
                       clim=[-clip_val, clip_val],
                       extent=[0, ntraces, tbase[-1], tbase[0]],
                       aspect='auto'
                       )
        # Plot colourbar under histogram
        clrbar_ax = fig.add_axes([xhist, 1.5 * mb/h + charty + pady - cstrip, whist, cstrip])
        plot_hrz_colorbar(clrbar_ax, cmap=cfg['cmap'])

        # wiggle plots go on top
        ax = wiggle_plot(ax,
                         data,
                         tbase,
                         ntraces,
                         skip=cfg['skip'],
                         gain=cfg['gain'],
                         rgb=cfg['colour'],
                         alpha=cfg['opacity'],
                         lw=cfg['lineweight']
                         )
        # ax.set_ylim(ax.get_ylim()[::-1])

    else:
        Notice.fail("You need to specify the type of display: wiggle or vd")

    # Annotations
    ax = decorate_seismic(ax, ntraces, tickfmt, cfg)

    # Plot histogram.
    hist_ax = fig.add_axes([xhist, 1.5 * mb/h + charty + pady, whist, charty])
    hist_ax = plot_histogram(hist_ax, data, tickfmt, fs)

    # Plot spectrum.
    spec_ax = fig.add_axes([xhist, 1.5 * mb/h, whist, charty])
    spec_ax = plot_spectrum(spec_ax, data, dt, tickfmt, fs)

    t2 = time.time()
    Notice.ok("Built plot in {:.1f} s".format(t2-t1))

    #####################################################################
    #
    # SAVE FILE
    #
    #####################################################################
    Notice.hr_header("Saving")
    s = "Saved image file {} in {:.1f} s"
    if cfg['outfile']:
        fig.savefig(cfg['outfile'])
        t3 = time.time()
        Notice.ok(s.format(cfg['outfile'], t3-t2))
    else:
        stem, _ = os.path.splitext(target)
        fig.savefig(stem)
        t3 = time.time()
        Notice.ok(s.format(stem+'.png', t3-t2))

    return


if __name__ == "__main__":

    Notice.title()
    parser = argparse.ArgumentParser(description='Plot a SEGY file.')
    parser.add_argument("-c", "--config",
                        metavar="config file",
                        type=argparse.FileType('r'),
                        default="config.yaml",
                        nargs="?",
                        help="The name of a YAML config file.")
    parser.add_argument('filename',
                        metavar='SEGY file',
                        type=str,
                        nargs='?',
                        help='The path to a SEGY file.')
    parser.add_argument('-o', '--out',
                        metavar='Output file',
                        type=str,
                        nargs='?',
                        default='',
                        help='The path to an output file.')
    args = parser.parse_args()
    target = args.filename
    with args.config as f:
        cfg = yaml.load(f)
    Notice.hr_header("Initializing")
    Notice.info("Filename   {}".format(target))
    Notice.info("Config     {}".format(args.config.name))
    cfg['outfile'] = args.out

    # Go and do things.
    main(target, cfg)

    # We're back! And we're done...
    Notice.hr_header("Done")
