#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seismic plotter.

:copyright: 2016 Agile Geoscience
:license: Apache 2.0
"""
import argparse
import os
import time
import glob
import re
import datetime

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from PIL import Image

from seismic import Seismic
from notice import Notice
import utils
import plotter


def main(target, cfg):
    """
    Puts everything together.
    """
    t0 = time.time()

    #####################################################################
    #
    # READ SEGY
    #
    #####################################################################
    s = Seismic.from_segy(target, params={'ndim': cfg['ndim']})

    # Set the line and/or xline number.
    try:
        n, xl = cfg['number']
    except:
        n, xl = cfg['number'], 0.5

    # Set the direction.
    if (s.ndim) == 2:
        direction = ['inline']
    elif cfg['direction'].lower()[0] == 'i':
        direction = ['inline']
    elif cfg['direction'].lower()[0] in ['x', 'c']:  # Allow 'crossline' too.
        direction = ['xline']
    elif cfg['direction'].lower()[0] == 't':
        direction = ['tslice']
    else:
        direction = ['xline', 'inline']

    # Get the data.
    ss = [Seismic.from_seismic(s, n=n, direction=d) for n, d in zip((n, xl), direction)]

    clip_val = np.percentile(s.data, cfg['percentile'])

    if clip_val < 10:
        fstr = '{:.3f}'
    elif clip_val < 100:
        fstr = '{:.2f}'
    elif clip_val < 1000:
        fstr = '{:.1f}'
    else:
        fstr = '{:.0f}'

    # Notify user of parameters.
    Notice.info("n_traces   {}".format(s.ntraces))
    Notice.info("n_samples  {}".format(s.nsamples))
    Notice.info("dt         {}".format(s.dt))
    Notice.info("t_start    {}".format(s.tstart))
    Notice.info("t_end      {}".format(s.tend))
    Notice.info("max_val    " + fstr.format(np.amax(s.data)))
    Notice.info("min_val    " + fstr.format(np.amin(s.data)))
    Notice.info("clip_val   " + fstr.format(clip_val))

    t1 = time.time()
    Notice.ok("Read data in {:.1f} s".format(t1-t0))

    #####################################################################
    #
    # MAKE PLOT
    #
    #####################################################################
    Notice.hr_header("Plotting")

    # Plot size parameters.
    fs = cfg['fontsize']
    wsl = 6  # Width of sidelabel, inches
    mih = 12  # Minimum plot height, inches
    fhh = 5  # File header box height, inches
    m = 0.5  # basic unit of margins, inches

    # Margins, CSS like: top, right, bottom, left.
    mt, mr, mb, ml = m, 2 * m, m, 2 * m
    mm = 2*m  # padded margin between seismic and label

    # Width is determined by seismic width, plus sidelabel, plus margins.
    # Height is given by ips, but with a minimum of mih inches.
    if 'tslice' in direction:
        print('doing tslice')
        seismic_width = max([s.ninlines for s in ss]) / cfg['tpi']
        seismic_height_raw = max([s.nxlines for s in ss]) / cfg['tpi']
        print(seismic_width, seismic_height_raw)
    else:
        seismic_width = [s.ntraces / cfg['tpi'] for s in ss]
        seismic_height_raw = cfg['ips'] * (s.tbasis[-1] - s.tbasis[0])

    w = ml + max(seismic_width) + mm + wsl + mr  # inches
    seismic_height = len(ss) * seismic_height_raw
    h_reqd = mb + seismic_height + 0.75*(len(ss)-1) + mt  # inches
    h = max(mih, h_reqd)

    # Calculate where to start sidelabel and seismic data.
    # Depends on whether sidelabel is on the left or right.
    if cfg['sidelabel'] == 'right':
        ssl = (ml + max(seismic_width) + mm) / w  # Start of side label (ratio)
        seismic_left = ml / w
    else:
        ssl = ml / w
        seismic_left = (ml + wsl + mm) / w

    adj = max(0, h - h_reqd) / 2
    seismic_bottom = (mb / h) + adj / h
    seismic_width_fraction = [sw / w for sw in seismic_width]
    seismic_height_fraction = seismic_height_raw / h

    # Publish some notices so user knows plot size.
    Notice.info("plot width   {:.2f} in".format(w))
    Notice.info("plot height  {:.2f} in".format(h))

    # Make the figure.
    fig = plt.figure(figsize=(w, h), facecolor='w')

    # Set the tickformat.
    tickfmt = mtick.FormatStrFormatter('%.0f')

    # Plot title.
    if cfg['title']:
        title = re.sub(r'_filename', target, cfg['title'])
        title_ax = fig.add_axes([ssl, 1-mt/h, wsl/w, mt/h])
        title_ax = plotter.plot_title(title_ax, title, fs=1.4*fs, cfg=cfg)

    # Plot title.
    if cfg['subtitle']:
        date = str(datetime.date.today())
        subtitle = re.sub(r'_date', date, cfg['subtitle'])
        subtitle_ax = fig.add_axes([ssl, 1-mt/h, wsl/w, mt/h])
        title_ax = plotter.plot_subtitle(subtitle_ax, subtitle, fs=0.75*fs, cfg=cfg)

    # Plot text header.
    start = (h - 1.5*mt - fhh) / h
    head_ax = fig.add_axes([ssl, start, wsl/w, fhh/h])
    head_ax = plotter.plot_header(head_ax, s.header, fs=fs-1, cfg=cfg)

    # Plot histogram.
    # Params for histogram plot.
    pady = 0.75 / h  # 0.75 inch
    padx = 0.75 / w   # 0.75 inch
    cstrip = 0.3/h   # color_strip height = 0.3 in
    charth = 1.5/h   # height of charts = 1.5 in
    chartw = wsl/w - mr/w - padx  # or ml/w for left-hand sidelabel; same thing
    chartx = (ssl + padx)
    histy = 1.5 * mb/h + charth + pady
    # Plot colourbar under histogram.
    clrbar_ax = fig.add_axes([chartx, histy - cstrip, chartw, cstrip])
    clrbar_ax = plotter.plot_colourbar(clrbar_ax, cmap=cfg['cmap'])
    # Plot histogram itself.
    hist_ax = fig.add_axes([chartx, histy, chartw, charth])
    hist_ax = plotter.plot_histogram(hist_ax,
                                     s.data,
                                     tickfmt,
                                     cfg,
                                     fs=fs)

    # Plot spectrum.
    specy = 1.5 * mb/h
    spec_ax = fig.add_axes([chartx, specy, chartw, charth])

    try:
        spec_ax = s.plot_spectrum(ax=spec_ax,
                                  tickfmt=tickfmt,
                                  ntraces=20,
                                  fontsize=fs,
                                  colour=utils.rgb_to_hex(cfg['highlight_colour']),
                                  )
    except:
        pass

    for i, line in enumerate(ss):
        # Add the seismic axis.
        ax = fig.add_axes([seismic_left,
                           seismic_bottom + i*seismic_height_fraction + i*pady,
                           seismic_width_fraction[i],
                           seismic_height_fraction
                           ])

        # Plot seismic data.
        if cfg['display'].lower() in ['vd', 'varden', 'variable', 'both']:
            _ = ax.imshow(line.data.T,
                          cmap=cfg['cmap'],
                          clim=[-clip_val, clip_val],
                          extent=[line.olines[0],
                                  line.olines[-1],
                                  1000*line.tbasis[-1],
                                  line.tbasis[0]],
                          aspect='auto'
                          )

            # This does not work: should cut off line at cfg['tmax']
            # ax.set_ylim(1000*cfg['tmax'] or 1000*line.tbasis[-1], line.tbasis[0])

        if cfg['display'].lower() in ['wiggle', 'both']:
            ax = line.wiggle_plot(cfg['number'], direction,
                                  ax=ax,
                                  skip=cfg['skip'],
                                  gain=cfg['gain'],
                                  rgb=cfg['colour'],
                                  alpha=cfg['opacity'],
                                  lw=cfg['lineweight'],
                                  tmax=cfg['tmax'],
                                  )

        if cfg['display'].lower() not in ['vd', 'varden', 'variable', 'wiggle', 'both']:
            Notice.fail("You must specify the type of display: wiggle, vd, both.")
            return

        # Seismic axis annotations.
        fs = cfg['fontsize'] - 2
        ax.set_ylabel(utils.LABELS[line.ylabel], fontsize=fs)
        ax.set_xlabel(utils.LABELS[line.xlabel], fontsize=fs, horizontalalignment='center')
        ax.set_xticklabels(ax.get_xticks(), fontsize=fs)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fs)
        ax.xaxis.set_major_formatter(tickfmt)
        ax.yaxis.set_major_formatter(tickfmt)

        # Watermark.
        if cfg['watermark_text']:
            ax = plotter.watermark_seismic(ax, cfg)

        # Make parasitic axes for labeling CDP number.

        # NEED TO RELABEL TO MATCH ACTUAL LINE

        ylim = ax.get_ylim()
        par1 = ax.twiny()
        par1.spines["top"].set_position(("axes", 1.0))
        par1.plot(line.slines, np.zeros_like(line.slines), alpha=0)
        par1.set_xlabel(utils.LABELS[line.slabel], fontsize=fs)
        par1.set_ylim(ylim)

        # Adjust ticks
        tx = par1.get_xticks()
        newtx = [line.slines[len(line.slines)*(i//len(tx))] for i, _ in enumerate(tx)]
        par1.set_xticklabels(newtx, fontsize=fs)

    t2 = time.time()
    Notice.ok("Built plot in {:.1f} s".format(t2-t1))

    #####################################################################
    #
    # SAVE FILE
    #
    #####################################################################
    Notice.hr_header("Saving")

    dname, fname, ext = utils.path_bits(target)
    outfile = cfg['outfile'] or ''
    if not os.path.splitext(outfile)[1]:
        outfile = os.path.join(cfg['outfile'] or dname, fname + '.png')

    fig.savefig(outfile)

    t3 = time.time()
    Notice.info("output file {}".format(outfile))
    Notice.ok("Saved output in {:.1f} s".format(t3-t2))

    if cfg['stain_paper'] or cfg['coffee_rings'] or cfg['distort'] or cfg['scribble']:
        fname = os.path.splitext(outfile)[0] + ".stupid.png"
        fig.savefig(fname)
    else:
        return

    #####################################################################
    #
    # SAVE STUPID FILE
    #
    #####################################################################
    Notice.hr_header("Applying the stupidity")

    stupid_image = Image.open(fname)
    if cfg['stain_paper']: utils.stain_paper(stupid_image)
    utils.add_rings(stupid_image, cfg['coffee_rings'])
    if cfg['scribble']: utils.add_scribble(stupid_image)
    stupid_image.save(fname)

    t4 = time.time()
    Notice.info("output file {}".format(fname))
    Notice.ok("Saved stupidity in {:.1f} s".format(t4-t3))

    return


if __name__ == "__main__":

    Notice.title()
    parser = argparse.ArgumentParser(description='Plot a SEGY file.')
    parser.add_argument("-c", "--config",
                        metavar="config file",
                        type=argparse.FileType('r'),
                        default="config.yaml",
                        nargs="?",
                        help="The name of a YAML config file. Default: config.yaml.")
    parser.add_argument('filename',
                        metavar='SEGY file',
                        type=str,
                        nargs='?',
                        help='The path to one or more SEGY files. Uses Unix-style pathname expansion.')
    parser.add_argument('-o', '--out',
                        metavar='output file',
                        type=str,
                        nargs='?',
                        default='',
                        help='The path to an output file. Default: same as input file, but with png file extension.')
    parser.add_argument('-R', '--recursive',
                        action='store_true',
                        help='Descend into subdirectories.')
    args = parser.parse_args()
    target = args.filename
    with args.config as f:
        cfg = yaml.load(f)
    Notice.hr_header("Initializing")
    Notice.info("config     {}".format(args.config.name))

    # Fill in 'missing' fields in cfg.
    cfg = {k: cfg.get(k, v) for k, v in utils.DEFAULTS.items()}
    cfg['outfile'] = args.out

    # Go do it!
    for t in glob.glob(target):
        Notice.hr_header("Processing file")
        Notice.info("filename   {}".format(t))
        main(t, cfg)
        Notice.hr_header("Done")
