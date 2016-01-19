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
from utils import add_subplot_axes
from utils import make_patch_spines_invisible


#####################################################################
#
# VARIOUS PLOTTING FUNCTIONS
#
#####################################################################

def decorate_seismic(ax, fs=10):
    """
    Add various things to the seismic plot.
    """
    ax.set_ylabel('Time samples', fontsize=fs)
    ax.set_xlabel('Trace number', fontsize=fs)
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs)
    # par1.set_xticklabels(par1.get_xticks(), fontsize=fs - 2)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs)
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
    font.set_size(fs-2)
    eblist = []
    for i in range(0, 3200, 80):
        eblist.append(s[i:i + 80])
    head_ax.axis([0, 40, 0, 40])
    buff = 1.0
    for i, line in enumerate(eblist):
        head_ax.text(buff, 40 - i * 0.95 - buff, line[4:], ha='left', va='top',
                     rotation=0, fontproperties=font)
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


def plot_histogram(hist_ax, data, fs=10):
    """
    Plot a histogram of amplitude values.
    """
    largest = max(np.amax(data), abs(np.amin(data)))
    clip_val = np.percentile(data, 99.0)
    hist_ax.patch.set_alpha(0.5)
    hist_ax.hist(np.ravel(data), bins=int(50.0 / (clip_val / largest)), alpha=0.5, color=['black'], linewidth=0)
    hist_ax.set_xlim(-clip_val, clip_val)
    hist_ax.set_xticklabels(hist_ax.get_xticks(), fontsize=fs-4)
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

    nsamples = section.traces[0].header.number_of_samples_in_this_trace
    dt = section.traces[0].header.sample_interval_in_ms_for_this_trace
    ntraces = len(section.traces)
    tbase = np.linspace(0, ((nsamples-1) * (dt / 1000.0)), nsamples)
    tstart = 0
    tend = np.amax(tbase)

    # Build the data container
    elev, esp, ens = [], [], []  # energy source point number
    data = np.zeros((nsamples, ntraces))
    for i, trace in enumerate(section.traces):
        data[:, i] = trace.data
        elev.append(trace.header.receiver_group_elevation)
        esp.append(trace.header.energy_source_point_number)
        ens.append(trace.header.ensemble_number)

    line_extents = {'first_trace': 1,
                    'last_trace': ntraces,
                    'start_time': tstart,
                    'end_time': tend
                    }

    clip_val = np.percentile(data, 99.0)
    print(line_extents)
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

    ##################################
    # Plot size parameters
    # Some constants
    wsl = 6  # Width of sidelabel
    mih = 8  # Minimum plot height
    fhh = 5  # File header height

    # Width and height of data area
    wd = ntraces / cfg['tpi']
    hd = cfg['ips'] * (tend - tstart)/1000

    aspect = hd / wd

    print(tstart, tend, cfg['ips'], wd, hd)

    # Width is determined by tpi, plus a constant for the sidelabel, plus 1 in
    w = wsl + wd + 1
    # Height is given by ips, but with a minimum of 6 inches, plus 1 in
    h = max(mih, hd) + 1

    print(w, h)

    # One inch in height and width
    oih = 1/h
    oiw = 1/w

    # Margins, CSS like
    m = 0.55
    mt, mr, mb, ml = m*oih, 2*m*oiw, m*oih, 2*m*oiw

    # Position of divider between seismic and sidelabel
    col2 = 1 - (wsl / w)
    wsd = col2 - 0.02  # Width of seismic data
    ssl = col2 + 0.02  # Start of sidelabel

    # Set the fontsize
    fs = cfg['fontsize']

    ##################################
    # Make the figure.
    fig = plt.figure(figsize=(w, h), facecolor='w')
    ax = fig.add_axes([ml, mb, wsd, (1-mb-mt)])
    #par1 = ax.twiny()
    #par2 = ax.twiny()

    # Offset the top spine of par2.  The ticks and label have already been
    # placed on the top by twiny above.
    #fig.subplots_adjust(top=0.75)
    #par2.spines["top"].set_position(("axes", 1.2))

    # Having been created by twiny, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the
    # patch and spines invisible.
    #par2 = make_patch_spines_invisible(par2)

    # Second, show the right spine.
    #par2.spines["top"].set_visible(True)

    # p2 = par1.plot(ens, np.zeros_like(ens))

    #par1.set_xlabel("CDP number", fontsize=fs-2)
    #par2.set_xlabel("source elevation", fontsize=fs-2)

    im = ax.imshow(data,
                   cmap=cm.gray,
                   origin='upper',
                   vmin=-clip_val,
                   vmax=clip_val,
                   aspect=aspect
                   )

    # Decorate the seismic axes
    ax = decorate_seismic(ax, fs)

    # Plot colorbar.
    fig = plot_colourbar(fig, ax, im, data, fs)

    # Plot text header.
    s = str(section.textual_file_header)[2:-1]
    head_ax = fig.add_axes([ssl, 1-(mb + fhh*oih), 1-(ml+ssl), fhh*oih])
    head_ax = plot_header(head_ax, s, fs)

    # Plot blurb.
    # blurb = "select trace header info goes here \n and here \n and here"
    # trhead_ax = fig.add_axes([0.760, 0.37, 0.225, 0.165], axisbg='w')
    # trhead_ax = plot_trace_info(trhead_ax, blurb, fs)

    # Plot histogram.
    xhist = ssl + 0.02
    whist = 1 - (ml+ssl) - 2*0.02
    hist_ax = fig.add_axes([xhist, 0.25, whist, 0.1])
    hist_ax = plot_histogram(hist_ax, data, fs)

    # Plot spectrum.
    spec_ax = fig.add_axes([xhist, 0.1, whist, 0.1])
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
