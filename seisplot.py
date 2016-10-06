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
import glob

# Import 3rd party.
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from PIL import Image

# Import our stuff.
from seismic import Seismic
from notice import Notice
import utils
import plotter


def main(target, cfg):
    """
    Puts everything together.
    """
    t0 = time.time()

    # Read the file and get the data.
    s = Seismic.from_segy(target)
    direction = 'inline' if (cfg['direction'].lower()[0] == 'i') else 'xline'
    data = s.get_line(direction, cfg['number'])

    if s.ndim == 2:
        x_label_text = 'CDP number'
    else:
        x_label_text = 'Crossline number' if direction=='inline' else 'Inline number'
    tr_label_text = 'Trace number'

    clip_val = np.percentile(data, cfg['percentile'])

    # Notify user of parameters
    Notice.info("n_traces   {}".format(s.ntraces))
    Notice.info("n_samples  {}".format(s.nsamples))
    Notice.info("dt         {}".format(s.dt))
    Notice.info("t_start    {}".format(s.tstart))
    Notice.info("t_end      {}".format(s.tend))
    Notice.info("max_val    {:.3f}".format(np.amax(data)))
    Notice.info("min_val    {:.3f}".format(np.amin(data)))
    Notice.info("clip_val   {:.3f}".format(clip_val))

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
    seismic_width = s.ntraces / cfg['tpi']
    w = ml + seismic_width + mm + wsl + mr  # inches

    # Height is given by ips, but with a minimum of mih inches
    seismic_height = cfg['ips'] * (s.tbasis[-1] - s.tbasis[0])
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

    # Make parasitic axes for labeling CDP number.
    par1 = ax.twiny()
    par1.spines["top"].set_position(("axes", 1.0))
    tickfmt = mtick.FormatStrFormatter('%.0f')
    par1.plot(s.xlines, np.zeros_like(s.xlines))
    par1.set_xlabel(x_label_text, fontsize=fs-2)
    par1.set_xticklabels(par1.get_xticks(), fontsize=fs-2)
    par1.xaxis.set_major_formatter(tickfmt)

    # Plot title.
    if cfg['filename']:
        title_ax = fig.add_axes([ssl, 1-mt/h, wsl/w, mt/(h)])
        title_ax = plotter.plot_title(title_ax, target, fs=1.5*fs, cfg=cfg)
        if s.ndim == 3:
            title_ax.text(0.0, 0.0, '{} {}'.format(direction.title(), line_no))

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
                                     data,
                                     tickfmt,
                                     percentile=cfg['percentile'],
                                     fs=fs)

    # Plot spectrum.
    specy = 1.5 * mb/h
    spec_ax = fig.add_axes([chartx, specy, chartw, charth])

    try:
        spec_ax = s.plot_spectrum(ax=spec_ax,
                                  tickfmt=tickfmt,
                                  ntraces=20,
                                  fontsize=fs)
    except:
        pass

    # Plot seismic data.
    if cfg['display'].lower() in ['vd', 'varden', 'variable', 'both']:
        _ = ax.imshow(data.T,
                      cmap=cfg['cmap'],
                      clim=[-clip_val, clip_val],
                      extent=[0, s.ntraces, 1000*s.tbasis[-1], s.tbasis[0]],
                      aspect='auto'
                      )

    if cfg['display'].lower() in ['wiggle', 'both']:
        ax = s.wiggle_plot(cfg['number'], direction,
                           ax=ax,
                           skip=cfg['skip'],
                           gain=cfg['gain'],
                           rgb=cfg['colour'],
                           alpha=cfg['opacity'],
                           lw=cfg['lineweight']
                           )

    if cfg['display'].lower() == 'wiggle':
        ax.set_ylim(ax.get_ylim()[::-1])

    if cfg['display'].lower() not in ['vd', 'varden', 'variable', 'wiggle', 'both']:
        Notice.fail("You need to specify the type of display: wiggle, vd, or both.")
        return

    # Seismic axis annotations.
    ax = plotter.decorate_seismic(ax,
                                  s.trace_range(direction=direction),
                                  tr_label_text,
                                  tickfmt,
                                  cfg)

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
        fig.savefig(outfile, dpi=cfg['dpi'])
        t3 = time.time()
        Notice.ok(s.format(outfile, t3-t2))
    else:  # Do the default: save a PNG in the same dir as the target.
        stem, _ = os.path.splitext(target)
        fig.savefig(stem)
        t3 = time.time()
        Notice.ok(s.format(stem+'.png', t3-t2))

    if stupid:
        fig.savefig(stem + ".stupid.png", dpi=cfg['dpi'])
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

    s = "Saved stupid file {}.stupid.png in {:.1f} s"
    t4 = time.time()
    Notice.ok(s.format(stem, t4-t3))

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
    Notice.info("Config     {}".format(args.config.name))

    # Fill in 'missing' fields in cfg.
    cfg = {k: cfg.get(k, v) for k, v in utils.DEFAULTS.items()}
    cfg['outfile'] = args.out

    # Go do it!
    for t in glob.glob(target):
            Notice.hr_header("Processing file: {}".format(t))
            main(t, cfg)
            Notice.hr_header("Done")
