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
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
from matplotlib import cm
from matplotlib.colors import makeMappingArray
from matplotlib.font_manager import FontProperties
from PIL import Image

from obspy.segy.segy import readSEGY

# Import our stuff.
from notice import Notice
import utils

#####################################################################
#
# VARIOUS PLOTTING FUNCTIONS
#
#####################################################################


def wiggle_plot(ax, data, tbase, ntraces,
                skip=1,
                perc=99.0,
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
    ax.set_xlim([0, ntraces])
    ax.set_ylabel('Two-way time [ms]', fontsize=fs - 2)
    ax.set_xlabel('Trace no.', fontsize=fs - 2, horizontalalignment='center')
    ax.set_xticklabels(ax.get_xticks(), fontsize=fs - 2)
    ax.set_yticklabels(ax.get_yticks(), fontsize=fs - 2)
    ax.xaxis.set_major_formatter(tickfmt)
    ax.yaxis.set_major_formatter(tickfmt)

    return ax


def watermark_seismic(ax, text, size, colour, xn, yn=None):
    """
    Add semi-transparent text to the seismic.
    """
    font = FontProperties()
    font.set_weight('bold')
    font.set_family('sans-serif')
    style = {'size': size,
             'color': colour,
             'alpha': 0.5,
             'fontproperties': font
             }
    alignment = {'rotation': 33,
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


def get_spectrum(signal, fs):
    windowed = signal * np.blackman(len(signal))
    a = abs(np.fft.rfft(windowed))
    f = np.fft.rfftfreq(len(signal), 1/fs)

    db = 20 * np.log10(a)
    sig = db - np.amax(db) + 20
    indices = ((sig[1:] >= 0) & (sig[:-1] < 0)).nonzero()
    crossings = [z - sig[z] / (sig[z+1] - sig[z]) for z in indices]
    mi, ma = np.amin(crossings), np.amax(crossings)
    x = np.arange(0, len(f))  # for back-interpolation
    f_min = np.interp(mi, x, f)
    f_max = np.interp(ma, x, f)

    return f, a, f_min, f_max


def plot_spectrum(spec_ax, data, dt, tickfmt, ntraces=10, fontsize=10):
    """
    Plot a power spectrum.
    w is window length for smoothing filter
    """

    trace_indices = utils.get_trace_indices(data.shape[1],
                                            ntraces,
                                            random=True)
    fs = 1/(dt/10**6)

    specs, peaks, mis, mas = [], [], [], []
    for ti in trace_indices:
        trace = data[:, ti]
        f, amp, fmi, fma = get_spectrum(trace, fs)

        peak = f[np.argmax(amp)]

        specs.append(amp)
        peaks.append(peak)
        mis.append(fmi)
        mas.append(fma)

    spec = np.mean(np.dstack(specs), axis=-1)
    spec = np.squeeze(spec)
    db = 20 * np.log10(amp)
    db = db - np.amax(db)
    f_peak = np.mean(peaks)
    f_min = np.amin(mis)
    f_max = np.amax(mas)

    statstring = "\nMin: {:.2f} Hz\nPeak: {:.2f} Hz\nMax: {:.2f}"
    stats = statstring.format(f_min, f_peak, f_max)

    spec_ax.plot(f, db, lw=0)  # Plot invisible line to get the min
    y_min = spec_ax.get_yticks()[0]
    spec_ax.fill_between(f, y_min, db, lw=0, facecolor='k', alpha=0.5)
    spec_ax.set_xlabel('frequency [Hz]', fontsize=fontsize - 4)
    spec_ax.xaxis.set_label_coords(0.5, -0.12)
    spec_ax.set_xlim([0, np.amax(f)])
    spec_ax.set_xticklabels(spec_ax.get_xticks(), fontsize=fontsize - 4)
    spec_ax.set_yticklabels(spec_ax.get_yticks(), fontsize=fontsize - 4)
    spec_ax.set_ylabel('power [dB]', fontsize=fontsize - 4)
    spec_ax.text(.98, .95, 'AMPLITUDE SPECTRUM'+stats,
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=spec_ax.transAxes, fontsize=fontsize - 3)
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
    Plot EBCDIC or ASCII header.
    """
    font = fm.FontProperties()
    font.set_family('monospace')
    font.set_size(fs-1)
    head_ax.axis([0, 40, 41, 0])
    head_ax.text(1, 1,
                 chunk(s),
                 ha='left', va='top',
                 fontproperties=font)
    head_ax.set_xticks([])
    head_ax.set_yticks([])
    head_ax.text(40, 42,
                 'plot by github.com/agile-geoscience/seisplot',
                 size=fs, color='lightgray',
                 ha=cfg['sidelabel'], va='top'
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


def plot_histogram(hist_ax, data, tickfmt, percentile=99.0, fs=10):
    """
    Plot a histogram of amplitude values.
    """
    datamax = np.amax(data)
    datamin = np.amin(data)
    largest = max(datamax, abs(datamin))
    clip_val = np.percentile(data, percentile)
    hist_ax.patch.set_alpha(0.0)
    y, x, _ = hist_ax.hist(np.ravel(data), bins=int(100.0 / (clip_val / largest)), 
                           alpha=1.0, color='#777777', lw=0)

    hist_ax.set_xlim(-clip_val, clip_val)
    hist_ax.set_xticklabels(hist_ax.get_xticks(), fontsize=fs - 4)
    hist_ax.set_xlabel('amplitude', fontsize=fs - 4)
    hist_ax.xaxis.set_label_coords(0.5, -0.12)
    hist_ax.set_ylim([0, y.max()])
    hist_ax.set_yticks(np.linspace(0, y.max(), 6))
    hist_ax.set_ylabel('percentage of samples', fontsize=fs - 4)
    hist_ax.tick_params(axis='x', pad=25)
    hist_ax.xaxis.labelpad = 25

    ticks = hist_ax.get_yticks().tolist()  # ytick labels
    percentages = 100*np.array(ticks)/data.size
    labels = []
    for label in percentages:
        labels.append("{:.2f}".format(label))
    hist_ax.set_yticklabels(labels, fontsize=fs - 4)
    hist_ax.xaxis.set_major_formatter(tickfmt)
    if datamax < 1:
        statstring = "\nMinimum: {:.4f}\nMaximum: {:.4f}".format(datamin, datamax)
    elif datamax < 10:
        statstring = "\nMinimum: {:.2f}\nMaximum: {:.4f}".format(datamin, datamax)
    elif datamax < 100:
        statstring = "\nMinimum: {:.1f}\nMaximum: {:.4f}".format(datamin, datamax)
    else:
        statstring = "\nMinimum: {:.0f}\nMaximum: {:.4f}".format(datamin, datamax)

    statstring += "\nClip percentile: {:.1f}".format(percentile)

    hist_ax.text(.98, .95, 'AMPLITUDE HISTOGRAM'+statstring,
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=hist_ax.transAxes, fontsize=fs - 3)
    hist_ax.set_alpha(0)
    hist_ax.grid()
    return hist_ax


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


def plot_title(title_ax, title, fs):
    """
    Add a title.
    """
    title_ax.text(1.0, 0.0, title, size=fs,
                  ha=cfg['sidelabel'],
                  va='bottom')
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

    # Make the data array.
    data = np.vstack([t.data for t in section.traces]).T

    # Collect some other data. Use a for loop because there are several.
    elev, esp, ens, tsq = [], [], [], []
    for i, trace in enumerate(section.traces):
        elev.append(trace.header.receiver_group_elevation)
        esp.append(trace.header.energy_source_point_number)
        ens.append(trace.header.ensemble_number)
        tsq.append(trace.header.trace_sequence_number_within_line)

    clip_val = np.percentile(data, cfg['percentile'])

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
    Notice.ok("Read data in {:.1f} s".format(t1-t0))

    #####################################################################
    #
    # MAKE PLOT
    #
    #####################################################################
    Notice.hr_header("Plotting")

    ##################################
    # Plot size parameters
    # Some constants
    fs = cfg['fontsize']
    wsl = 6  # Width of sidelabel, inches
    mih = 12  # Minimum plot height, inches
    fhh = 5  # File header box height, inches
    m = 0.5  # basic unit of margins, inches

    # Margins, CSS like: top, right, bottom, left.
    mt, mr, mb, ml  = m,  2 * m, m, 2 * m
    mm = m  # padded margin between seismic and label

    # Width is determined by seismic width, plus sidelabel, plus margins.
    seismic_width = ntraces / cfg['tpi']
    w = ml + seismic_width + mm + wsl + mr  # inches

    # Height is given by ips, but with a minimum of mih inches
    seismic_height = cfg['ips'] * (tbase[-1] - tbase[0]) / 1000
    h_reqd =  mb + seismic_height + mt  # inches
    h = max(mih, h_reqd)

    # Calculate where to start sidelabel and seismic data.
    # Depends on whether sidelabel is on the left or right.
    if cfg['sidelabel'] == 'right':
        ssl = (ml + seismic_width + mm) / w  # Start of side label (ratio)
        seismic_left = ml / w
    else:
        ssl = ml / w
        seismic_left = (ml + wsl + mm) / w

    adj = max(0, h - h_reqd) / 2
    seismic_bottom = (mb / h) + adj / h
    seismic_width_fraction = seismic_width / w
    seismic_height_fraction = seismic_height / h

    # Publish some notices so user knows plot size.
    Notice.info("Width of plot   {} in".format(w))
    Notice.info("Height of plot  {} in".format(h))

    ##################################
    # Make the figure.
    fig = plt.figure(figsize=(w, h), facecolor='w')

    # Add the main seismic axis.
    ax = fig.add_axes([seismic_left,
                       seismic_bottom,
                       seismic_width_fraction,
                       seismic_height_fraction
                       ])

    # make parasitic axes for labeling CDP number
    par1 = ax.twiny()
    par1.spines["top"].set_position(("axes", 1.0))
    tickfmt = mtick.FormatStrFormatter('%.0f')
    par1.plot(ens, np.zeros_like(ens))
    par1.set_xlabel("CDP number", fontsize=fs-2)
    par1.set_xticklabels(par1.get_xticks(), fontsize=fs-2)
    par1.xaxis.set_major_formatter(tickfmt)

    # Plot title
    title_ax = fig.add_axes([ssl, 1-mt/h, wsl/w, mt/(h)])
    title_ax = plot_title(title_ax, target, fs=1.5*fs)

    # Plot text header.
    s = section.textual_file_header.decode()
    start = (h - 1.5*mt - fhh) / h
    head_ax = fig.add_axes([ssl, start, wsl/w, fhh/h])
    head_ax = plot_header(head_ax, s, fs=fs-1)

    # Plot histogram.
    # Params for histogram plot
    pady = 0.75 / h  # 0.75 inch
    padx = 0.75 / w   # 0.75 inch
    cstrip = 0.3/h   # color_strip height = 0.3 in
    charth = 1.5/h   # height of charts = 1.5 in
    chartw = wsl/w - mr/w - padx  # or ml/w for left-hand sidelabel -- same thing
    chartx = (ssl + padx)
    histy = 1.5 * mb/h + charth + pady
    # Plot colourbar under histogram
    clrbar_ax = fig.add_axes([chartx, histy - cstrip, chartw, cstrip])
    clrbar_ax = plot_colourbar(clrbar_ax, cmap=cfg['cmap'])
    # Plot histogram itself
    hist_ax = fig.add_axes([chartx, histy, chartw, charth])
    hist_ax = plot_histogram(hist_ax, data, tickfmt, percentile=cfg['percentile'], fs=fs)

    # Plot spectrum.
    specy = 1.5 * mb/h
    spec_ax = fig.add_axes([chartx, specy, chartw, charth])
    spec_ax = plot_spectrum(spec_ax, data, dt, tickfmt, ntraces=20, fontsize=fs)

    # Plot seismic data.
    if cfg['display'].lower() in ['vd', 'varden', 'variable']:
        im = ax.imshow(data,
                       cmap=cfg['cmap'],
                       clim=[-clip_val, clip_val],
                       extent=[0, ntraces, tbase[-1], tbase[0]],
                       aspect='auto'
                       )

    elif cfg['display'].lower() == 'wiggle':
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

    elif cfg['display'].lower() == 'both':
        # variable density goes on first
        im = ax.imshow(data,
                       cmap=cfg['cmap'],
                       clim=[-clip_val, clip_val],
                       extent=[0, ntraces, tbase[-1], tbase[0]],
                       aspect='auto'
                       )

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

    # Seismic axis annotations.
    ax = decorate_seismic(ax, ntraces, tickfmt, cfg)

    # Watermark.
    if cfg['watermark_text']:
        text = cfg['watermark_text']
        size = cfg['watermark_size']
        colour = cfg['watermark_colour']
        xn = cfg['watermark_cols']
        yn = cfg['watermark_rows']
        ax = watermark_seismic(ax, text, size, colour, xn, yn)

    t2 = time.time()
    Notice.ok("Built plot in {:.1f} s".format(t2-t1))

    #####################################################################
    #
    # SAVE FILE
    #
    #####################################################################
    Notice.hr_header("Saving")

    if cfg['stain_paper'] or cfg['coffee_rings'] or cfg['distort'] or cfg['scribble']:
        stupid = True
    else:
        stupid = False

    s = "Saved image file {} in {:.1f} s"
    if cfg['outfile']:

        if os.path.isfile(cfg['outfile']):
            outfile = cfg['outfile']
        else:  # is directory
            stem, ext = os.path.splitext(os.path.split(target)[1])
            outfile = os.path.join(cfg['outfile'], stem + '.png') 

        stem, _ = os.path.splitext(outfile)  # Needed for stupidity.
        fig.savefig(outfile)
        t3 = time.time()
        Notice.ok(s.format(outfile, t3-t2))
    else:  # Do the default: save a PNG in the same dir as the target.
        stem, _ = os.path.splitext(target)
        fig.savefig(stem)
        t3 = time.time()
        Notice.ok(s.format(stem+'.png', t3-t2))

    if stupid:
        fig.savefig(stem + ".stupid.png")
    else:
        return

    #####################################################################
    #
    # SAVE STUPID FILE
    #
    #####################################################################
    Notice.hr_header("Applying the stupidity")

    stupid_image = Image.open(stem + ".stupid.png")
    if cfg['stain_paper']:
        utils.stain_paper(stupid_image)
    utils.add_rings(stupid_image, cfg['coffee_rings'])
    if cfg['scribble']:
        utils.add_scribble(stupid_image)
    stupid_image.save(stem + ".stupid.png")

    s = "Saved stupid file stupid.png in {:.1f} s"
    t4 = time.time()
    Notice.ok(s.format(t4-t3))

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
    parser.add_argument('-R', '--recursive',
                        action='store_true',
                        help='Descend into subdirectories.')
    args = parser.parse_args()
    target = args.filename
    with args.config as f:
        cfg = yaml.load(f)
    Notice.hr_header("Initializing")
    Notice.info("Config     {}".format(args.config.name))

    # Fill in 'missing' fields in cfg.
    defaults = {'sidelabel': 'right',
                'tpi': 10,
                'ips': 1,
                'skip': 2,
                'display': 'vd',
                'gain': 1.0,
                'percentile': 99.0,
                'colour': [0, 0, 0],
                'opacity': 1.0,
                'lineweight': 0.2,
                'cmap': 'Greys',
                'fontsize': 10,
                'watermark_text': '',  # None by default
                'watermark_size': 14,
                'watermark_colour': 'white',
                'watermark_rotation': 33,
                'watermark_cols': 6,
                'watermark_rows': 0,  # automatic
                'stain_paper': None,
                'coffee_rings': 0,
                'distort': False,
                'scribble': False,
                }

    for k, v in defaults.items():
        if cfg.get(k) is None:
            cfg[k] = v

    cfg['outfile'] = args.out

    # Gather files to work on, then go and do them.
    if os.path.isfile(target):
        Notice.hr_header("Processing file: {}".format(target))
        main(target, cfg)
        Notice.hr_header("Done")
    elif os.path.isdir(target):
        if args.recursive:
            Notice.info("Looking for SEGY files in {} and its subdirectories".format(target))
            for target in utils.walk(target, "\\.se?gy$"):
                Notice.hr_header("Processing file: {}".format(target))
                main(target, cfg)
        else:
            Notice.info("Finding SEGY files in {}".format(target))
            for target in utils.listdir(target, "\\.se?gy$"):
                Notice.hr_header("Processing file: {}".format(target))
                main(target, cfg)
        Notice.hr_header("Done")
    else:
        Notice.fail("Not a file or directory.")
