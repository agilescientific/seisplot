#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Utility functions for seisplot.

:copyright: 2016 Agile Geoscience
:license: Apache 2.0
"""
# Standard
import os
import re

# 3rd party
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


DEFAULTS = {'line': 'inline',
            'direction': 'inline',
            'number': 0.5,
            'sidelabel': 'right',
            'tpi': 10,
            'ips': 1,
            'skip': 2,
            'display': 'vd',
            'filename': True,
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
            'watermark_alpha': 0.5,
            'watermark_angle': 33,
            'watermark_cols': 6,
            'watermark_rows': 0,  # automatic
            'stain_paper': None,
            'coffee_rings': 0,
            'distort': False,
            'scribble': False,
            }

HEADERS = [
    'for_3d_poststack_data_this_field_is_for_cross_line_number',  # sawtooth for 3d
    'for_3d_poststack_data_this_field_is_for_in_line_number',     # steps for 3d
    'trace_number_within_the_ensemble',  # nothing for 2d, same as crossline number for 3d
    'trace_sequence_number_within_line',  # nothing for 2d, zero-based sawtooth for 3d
    'trace_sequence_number_within_segy_file',  # zero-based monotonic for 2d and 3d
    'ensemble_number',  # trace-number-based monotonic for 2d
    'trace_number_within_the_ensemble',
    'original_field_record_number',
    'energy_source_point_number',
    'trace_number_within_the_original_field_record',
#    'trace_identification_code',
]


def get_pattern_from_stream(stream, pattern_function):
    """
    Return the first non-zero-based monotonic header.
    If there isn't one, return the first monotonic header.
    """
    candidates = []
    for header in HEADERS:
        data = [t.header.__dict__[header] for t in stream.traces]
        if pattern_function(data):
            candidates.append(data)
    for candidate in candidates:
        if candidate[0] > 0:
            return candidate
    if candidates:
        return candidates[0]
    else:
        return None


def get_trace_indices(y, ntraces, random=False):
    """
    Get a subset of trace positions.
    Args:
        y (int): The total number of traces to choose from.
        ntraces (int): The number of traces to choose.
        random (bool): Whether to choose random traces, or
            to choose regularly-spaced.
    Yields:
        str: Full path to each file in turn.
    """
    if random:
        x = 0.05 + 0.9*np.random.random(ntraces)  # avoids edges
        ti = np.sort(x * y)
    else:
        n = ntraces + 1
        ti = np.arange(1./n, 1., 1./n) * y
    return np.round(ti).astype(int)


def max_opacity(image, maxi):
    """
    Adjust the maximum opacity of an image.
    """
    data = np.array(image)
    adj = maxi*255/np.amax(data)
    data[..., 3] = adj * data[..., 3]
    result = Image.fromarray(data)
    return result


def stain_paper(image):
    """
    Add a staining image to the paper.
    """
    fname = "resources/stained_and_folded_paper_.png"
    paper = Image.open(fname)

    # Adjust opacity.
    paper2 = max_opacity(paper, 0.25)

    # Crop paper to image size
    # If paper is bigger than image.
    left = int((paper2.size[0] - image.size[0]) / 2)
    right = left + image.size[0]
    upper = int((paper2.size[1] - image.size[1]) / 2)
    lower = upper + image.size[1]
    box = (left, upper, right, lower)
    paper2 = paper2.crop(box=box)

    # Do it.
    image.paste(paper2, (0, 0), paper2)
    return


def add_scribble(image):
    """
    Add a scribble.
    """
    files = list(listdir('resources', 'scribble_[0-9]+.png'))
    fname = np.random.choice(files)
    scribble = Image.open(fname)

    # Adjust opacity.
    scribble2 = max_opacity(scribble, 0.85)

    # Shrink a bit
    scribble2.thumbnail((256, 256), Image.ANTIALIAS)

    # Rotate the stain by a small random amount.
    angle = (np.random.random() - 0.5) * 20
    scribble2 = scribble2.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Find a random place near the upper corner.
    x = np.random.randint(image.size[0]-(1.5*scribble2.size[0]), image.size[0]-(0.8*scribble2.size[0]))
    y = np.random.randint(0, scribble2.size[1])

    # Do it.
    image.paste(scribble2, (x, y), scribble2)
    return


def add_a_ring(image):
    """
    Add a coffee cup ring.
    """
    fname = "resources/coffee_stain.png"
    stain = Image.open(fname)

    # Adjust opacity.
    stain2 = max_opacity(stain, 0.35)

    # Rotate the stain by a random amount.
    angle = np.random.random() * 360
    stain2 = stain2.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Find a random place for it.
    x = np.random.randint(-stain2.size[0]/2, image.size[0]-stain2.size[0]/2)
    y = np.random.randint(-stain2.size[1]/2, image.size[1]-stain2.size[1]/2)

    # Do it.
    image.paste(stain2, (x, y), stain2)
    return


def add_rings(image, n=0):
    """
    Add the required number of coffee rings.
    """
    if n:
        for i in range(n):
            add_a_ring(image)
    return


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
