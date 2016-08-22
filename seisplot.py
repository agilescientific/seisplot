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
import matplotlib.ticker as mtick
from PIL import Image

from obspy.segy.segy import readSEGY

# Import our stuff.
from notice import Notice
import utils
import plotter


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
    ninlines = section.traces[-1].header.trace_sequence_number_within_line
    last_tr = section.traces[-1].header.trace_sequence_number_within_segy_file
    nxlines = last_tr / ninlines

    nsamples = section.traces[0].header.number_of_samples_in_this_trace
    dt = section.traces[0].header.sample_interval_in_ms_for_this_trace
    ntraces = len(section.traces)
    tbase = 0.001 * np.arange(0, nsamples * dt, dt)
    tstart = 0
    tend = np.amax(tbase)

    # Make the data array.
    data = np.vstack([t.data for t in section.traces]).T

    threed = False
    if nxlines > 1:  # Then it's a 3D and `data` is an ensemble.
        threed = True
        cube = np.reshape(data.T, (ninlines, nxlines, nsamples))
        l = cfg['number']
        if cfg['direction'].lower()[0] == 'i':
            direction = 'inline'
            ntraces = nxlines
            l *= ninlines if (l < 1) else 1
            data = cube[l, :, :].T
        else:
            direction = 'xline'
            ntraces = ninlines
            l *= nxlines if (l < 1) else 1
            data = cube[:, l, :].T

    # Collect some other data. Use a for loop because there are several.
    elev, esp, ens, tsq = [], [], [], []
    for i, trace in enumerate(section.traces):
        elev.append(trace.header.receiver_group_elevation)
        esp.append(trace.header.energy_source_point_number)
        tsq.append(trace.header.trace_sequence_number_within_line)

        if threed:
            trs = []
            if direction == 'inline':
                cdp_label_text = 'Crossline number'
                trace_label_text = 'Trace number'
                ens.append(trace.header.for_3d_poststack_data_this_field_is_for_cross_line_number)
                trs.append(trace.header.for_3d_poststack_data_this_field_is_for_in_line_number)
            else:
                cdp_label_text = 'Inline number'
                trace_label_text = 'Trace number'
                ens.append(trace.header.for_3d_poststack_data_this_field_is_for_in_line_number)
                trs.append(trace.header.for_3d_poststack_data_this_field_is_for_cross_line_number)
            line_no = min(trs)
        else:
            cdp_label_text = 'CDP number'
            trace_label_text = 'Trace number'
            ens.append(trace.header.ensemble_number)
        min_tr, max_tr = 0, ntraces

    traces = (min_tr, max_tr)

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
    mt, mr, mb, ml = m,  2 * m, m, 2 * m
    mm = m  # padded margin between seismic and label

    # Width is determined by seismic width, plus sidelabel, plus margins.
    seismic_width = ntraces / cfg['tpi']
    w = ml + seismic_width + mm + wsl + mr  # inches

    # Height is given by ips, but with a minimum of mih inches
    seismic_height = cfg['ips'] * (tbase[-1] - tbase[0]) / 1000
    h_reqd = mb + seismic_height + mt  # inches
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
    par1.set_xlabel(cdp_label_text, fontsize=fs-2)
    par1.set_xticklabels(par1.get_xticks(), fontsize=fs-2)
    par1.xaxis.set_major_formatter(tickfmt)

    # Plot title
    title_ax = fig.add_axes([ssl, 1-mt/h, wsl/w, mt/(h)])
    title_ax = plotter.plot_title(title_ax, target, fs=1.5*fs, cfg=cfg)
    if threed:
        title_ax.text(0.0, 0.0, '{} {}'.format(direction.title(), line_no))

    # Plot text header.
    s = section.textual_file_header.decode()
    start = (h - 1.5*mt - fhh) / h
    head_ax = fig.add_axes([ssl, start, wsl/w, fhh/h])
    head_ax = plotter.plot_header(head_ax, s, fs=fs-1, cfg=cfg)

    # Plot histogram.
    # Params for histogram plot
    pady = 0.75 / h  # 0.75 inch
    padx = 0.75 / w   # 0.75 inch
    cstrip = 0.3/h   # color_strip height = 0.3 in
    charth = 1.5/h   # height of charts = 1.5 in
    chartw = wsl/w - mr/w - padx  # or ml/w for left-hand sidelabel; same thing
    chartx = (ssl + padx)
    histy = 1.5 * mb/h + charth + pady
    # Plot colourbar under histogram
    clrbar_ax = fig.add_axes([chartx, histy - cstrip, chartw, cstrip])
    clrbar_ax = plotter.plot_colourbar(clrbar_ax, cmap=cfg['cmap'])
    # Plot histogram itself
    hist_ax = fig.add_axes([chartx, histy, chartw, charth])
    hist_ax = plotter.plot_histogram(hist_ax,
                                     data,
                                     tickfmt,
                                     percentile=cfg['percentile'],
                                     fs=fs)

    # Plot spectrum.
    specy = 1.5 * mb/h
    spec_ax = fig.add_axes([chartx, specy, chartw, charth])

    try:
        spec_ax = plotter.plot_spectrum(spec_ax,
                                        data,
                                        dt,
                                        tickfmt,
                                        ntraces=20,
                                        fontsize=fs)
    except:
        pass

    # Plot seismic data.
    if cfg['display'].lower() in ['vd', 'varden', 'variable']:
        _ = ax.imshow(data,
                      cmap=cfg['cmap'],
                      clim=[-clip_val, clip_val],
                      extent=[0, ntraces, tbase[-1], tbase[0]],
                      aspect='auto'
                      )

    elif cfg['display'].lower() == 'wiggle':
        ax = plotter.wiggle_plot(ax,
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
        _ = ax.imshow(data,
                      cmap=cfg['cmap'],
                      clim=[-clip_val, clip_val],
                      extent=[0, ntraces, tbase[-1], tbase[0]],
                      aspect='auto'
                      )

        # wiggle plots go on top
        ax = plotter.wiggle_plot(ax,
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
    ax = plotter.decorate_seismic(ax, traces, trace_label_text, tickfmt, cfg)

    # Watermark.
    if cfg['watermark_text']:
        Notice.info("Adding watermark")
        ax = plotter.watermark_seismic(ax, cfg)

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
    defaults = {'line': 'inline',
                'number': 0.5,
                'sidelabel': 'right',
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
                'watermark_family': 'sans-serif',
                'watermark_style': 'normal',
                'watermark_weight': 'normal',
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
