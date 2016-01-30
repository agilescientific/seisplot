#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Simple seismic plotter.

:copyright: 2015 Agile Geoscience
:license: Apache 2.0
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def get_trace_indices(y, ntraces, spacing):
    if spacing == 'random':
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
    x = np.random.randint(-stain.size[0]/2, image.size[0]-stain2.size[0]/2)
    y = np.random.randint(-stain.size[1]/2, image.size[1]-stain2.size[1]/2)

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
