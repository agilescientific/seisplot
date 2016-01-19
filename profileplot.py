#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Simple seismic plotter.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
# Import standard libraries.
import argparse
import os

# Import 3rd party.
import yaml
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
from obspy.segy.segy import readSEGY

# Import our stuff.
from notice import Notice


def add_subplot_axes(ax, rect, axisbg='w'):
    """
    Facilitates the addition of a small subplot within another plot.

    From: http://stackoverflow.com/questions/17458580/
    embedding-small-plots-inside-subplots-in-matplotlib

    License: CC-BY-SA

    Args:
        ax (axis): A matplotlib axis.
        rect (list): A rect specifying [left pos, bot pos, width, height]
    Returns:
        axis: The sub-axis in the specified position.
    """
    def axis_to_fig(axis):
        fig = axis.figure

        def transform(coord):
            a = axis.transAxes.transform(coord)
            return fig.transFigure.inverted().transform(a)

        return transform

    fig = plt.gcf()
    left, bottom, width, height = rect
    trans = axis_to_fig(ax)
    x1, y1 = trans((left, bottom))
    x2, y2 = trans((left + width, bottom + height))
    subax = fig.add_axes([x1, y1, x2 - x1, y2 - y1])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2] ** 0.5
    y_labelsize *= rect[3] ** 0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def make_patch_spines_invisible(ax):
    """
    Removes spines from patches.
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax

def decorate_seismic(ax, fs=10):
    """
    Add various things to the seismic plot.
    """
    ax.set_ylabel('Two-way time [ms]', fontsize=fs-2)
    ax.set_xlabel('Trace no.', fontsize=fs - 2)
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs - 2)
    #par1.set_xticklabels(par1.get_xticks(), fontsize=fs - 2)
    ax.set_yticklabels(ax.get_xticks(), fontsize=fs - 2)
    ax.grid()
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
        colorbar_ax.text(0.5, 1.1, '%3.0f' % ma, transform=colorbar_ax.transAxes,
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


def plot_spectrum(spec_ax, data, dt, fs=10):
    """
    Plot a power spectrum.
    """
    S = abs(fft(data[:, 1]))
    faxis = np.fft.fftfreq(len(data[:, 1]), d=dt/10**6)

    spec_ax.patch.set_alpha(0.5)
    spec_ax.plot(faxis[:len(faxis)//4], np.log10(S[0:len(faxis)//4]), 'k', lw=2, alpha=0.5)
    spec_ax.set_xlabel('frequency [Hz]', fontsize=fs - 4)
    spec_ax.set_xticklabels(spec_ax.get_xticks(), fontsize=fs - 4)
    spec_ax.set_yticklabels(spec_ax.get_yticks(), fontsize=fs - 4)
    spec_ax.set_ylabel('power [dB]', fontsize=fs - 2)
    spec_ax.set_title('Power spectrum', fontsize=fs - 2)
    spec_ax.grid('on')

    return spec_ax


def plot_header(head_ax, s, fs=10):
    """
    Plot EBCIDIC header.
    """
    font = fm.FontProperties()
    font.set_family('monospace')
    eblist = []
    for i in range(0, 3200, 80):
        eblist.append(s[i:i + 80])
    head_ax.axis([0, 40, 0, 40])
    buff = 1.0
    for i, line in enumerate(eblist):
        head_ax.text(buff, 40 - i * 0.95 - buff, line[4:], ha='left', va='top',
                     rotation=0, fontsize=fs-4, fontproperties=font)
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

# Do a histogram
def plot_histogram(hist_ax, data, fs=10):
    largest = max(np.amax(data), abs(np.amin(data)))
    clip_val = np.percentile(data, 99.0)
    hist_ax.patch.set_alpha(0.5)
    hist_ax.hist(np.ravel(data), bins=int(50.0 / (clip_val / largest)), alpha=0.5, color=['black'])
    hist_ax.set_xlim(-clip_val, clip_val)
    hist_ax.set_xticklabels(hist_ax.get_xticks(), fontsize=fs - 4)
    hist_ax.set_xlabel('amplitude', fontsize=fs - 4)
    hist_ax.set_ylim(hist_ax.get_ylim()[0], hist_ax.get_ylim()[1]),
    hist_ax.set_yticks([])
    hist_ax.set_title('Histogram', fontsize=fs - 3)
    return hist_ax


def main(target, cfg):
    """
    Puts everything together.
    """

    # Read the file.
    section = readSEGY(target, unpack_headers=True)

    elev, esp, ens = [], [], []  # energy source point number
    for i, trace in enumerate(section.traces):
        nsamples = trace.header.number_of_samples_in_this_trace
        dt = trace.header.sample_interval_in_ms_for_this_trace
        elev.append(trace.header.receiver_group_elevation)
        esp.append(trace.header.energy_source_point_number)
        ens.append(trace.header.ensemble_number)

    ntraces = len(section.traces)
    tbase = np.arange(0, nsamples * dt / 1000.0, dt)
    tstart = 0
    tend = np.amax(tbase)
    aspect = float(ntraces) / float(nsamples)

    # Build the data container
    data = np.zeros((nsamples, ntraces))
    for i, trace in enumerate(section.traces):
        data[:, i] = trace.data

    line_extents = {'first_trace': 1,
                    'last_trace': ntraces,
                    'start_time': tstart,
                    'end_time': tend
                    }

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
    Notice.ok("Read data successfully")

    #####################################################################
    # 
    # MAKE PLOT
    # 
    #####################################################################
    Notice.hr_header("Plotting")

    w = ntraces / cfg['tpi']
    h = cfg['ips'] * (tend - tstart)/1000
    
    fig = plt.figure(figsize=(w, h), facecolor='w')
    ax = fig.add_axes([0.05, 0.1, 0.7, 0.8])
    par1 = ax.twiny()
    par2 = ax.twiny()

    # Offset the top spine of par2.  The ticks and label have already been
    # placed on the top by twiny above.
    fig.subplots_adjust(top=0.75)
    par2.spines["top"].set_position(("axes", 1.2))

    # Having been created by twiny, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    par2 = make_patch_spines_invisible(par2)

    # Second, show the right spine.
    par2.spines["top"].set_visible(True)

    # p2 = par1.plot(ens, np.zeros_like(ens))

    fs = cfg['fontsize']
    par1.set_xlabel("CDP number", fontsize=fs-2)
    par2.set_xlabel("source elevation", fontsize=fs-2)

    im = ax.imshow(data, cmap=cm.gray, origin='upper',
                   vmin=-clip_val,
                   vmax=clip_val,
                   extent=(line_extents['first_trace'],
                           line_extents['last_trace'],
                           line_extents['end_time'],
                           line_extents['start_time']),
                   aspect = aspect
                   )

    # Decorate the seismic axes
    ax = decorate_seismic(ax, fs)

    # Plot colorbar.
    fig = plot_colourbar(fig, ax, im, data, fs)

    # Plot text header.
    s = str(section.textual_file_header)[2:-1]
    head_ax = fig.add_axes([0.760, 0.55, 0.225, 0.35], axisbg='w')
    head_ax = plot_header(head_ax, s, fs)

    # Plot blurb.
    # blurb = "select trace header info goes here \n and here \n and here"
    # trhead_ax = fig.add_axes([0.760, 0.37, 0.225, 0.165], axisbg='w')
    # trhead_ax = plot_trace_info(trhead_ax, blurb, fs)

    # Plot histogram.
    hist_ax = fig.add_axes([0.79, 0.25, 0.18, 0.10], axisbg='w')
    hist_ax = plot_histogram(hist_ax, data, fs)

    # Plot spectrum.
    spec_ax = fig.add_axes([0.79, 0.1, 0.18, 0.1], axisbg='w')
    spec_ax = plot_spectrum(spec_ax, data, dt, fs)

    # Save figure.
    stem, _ = os.path.splitext(target)
    fig.savefig(stem)
    Notice.ok("Saved image file {}".format(stem+'.png'))
    Notice.hr_header("Done")


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
    args = parser.parse_args()
    target = args.filename
    with args.config as f:
        cfg = yaml.load(f)
    Notice.hr_header("Initializing")
    Notice.info("Filename   {}".format(target))
    Notice.info("Config     {}".format(args.config.name))
    main(target, cfg)
