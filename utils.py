# -*- coding: utf-8 -*-
"""
Utility functions for seisplot.

:copyright: 2016 Agile Geoscience
:license: Apache 2.0
"""
import os
import errno
import sys

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import utils


LABELS = {'i': 'inline', 'x': 'xline', 't': 'time [ms]'}

DEFAULTS = {'ndim': 2,
            'line': 'inline',
            'direction': 'inline',
            'number': 0.5,
            'sidelabel': 'right',
            'tpi': 10,
            'plot_width': 17,
            'ips': 1,
            'plot_height': 11,
            'trange': [0, 0],
            'skip': 2,
            'display': 'vd',
            'title': '_filename',
            'subtitle': '_date',
            'credit': True,
            'gain': 1.0,
            'percentile': 99.0,
            'colour': [0, 0, 0],
            'opacity': 1.0,
            'lineweight': 0.2,
            'grid_time': False,
            'grid_traces': False,
            'grid_colour': [0, 0, 0],
            'grid_alpha': 0.15,
            'cmap': 'Greys',
            'interpolation': 'bicubic',
            'highlight_colour': [0, 0, 0],
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
    # 'trace_identification_code',
]


def rgb_to_hex(rgb):
    """
    Utility function to convert (r,g,b) triples to hex.
    http://ageo.co/1CFxXpO
    Args:
      rgb (tuple): A sequence of RGB values in the
        range 0-255 or 0-1.
    Returns:
      str: The hex code for the colour.
    """
    r, g, b = rgb[:3]
    if (r < 0) or (g < 0) or (b < 0):
            raise Exception("RGB values must all be 0-255 or 0-1")
    if (r > 255) or (g > 255) or (b > 255):
            raise Exception("RGB values must all be 0-255 or 0-1")
    if (0 < r < 1) or (0 < g < 1) or (0 < b < 1):
        if (r > 1) or (g > 1) or (b > 1):
            raise Exception("RGB values must all be 0-255 or 0-1")
    if (0 <= r <= 1) and (0 <= g <= 1) and (0 <= b <= 1):
        rgb = tuple([int(round(val * 255)) for val in [r, g, b]])
    else:
        rgb = (int(r), int(g), int(b))
    result = '#%02x%02x%02x' % rgb
    return result.lower()


def path_bits(path):
    """
    Returns the directory, the stem of the filename, and the extension.
    """
    d, f = os.path.split(path)
    s, e = os.path.splitext(f)
    return d, s, e


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
        y (int): The total number of traces to choose from. 1D or 2D.
        ntraces (int): The number of traces to choose.
        random (bool): Whether to choose random traces, or
            to choose regularly-spaced.
    Yields:
        ndarray. 1D array of ints or tuples, with the coords of the
            random traces.
    """
    total = np.product(y)
    if random:
        x = 0.05 + 0.9*np.random.random(ntraces)  # avoids edges
        ti = np.sort(x * total)
    else:
        n = ntraces + 1
        ti = np.arange(1./n, 1., 1./n) * total
    if len(y) > 1:
        # Re-form the 2D indices.
        ti = [(t%y[0], t//y[0]) for t in ti]
    return np.floor(ti).astype(int)


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
    files = list(utils.listdir('resources', 'scribble_[0-9]+.png'))
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


# Sadly, Python fails to provide the following magic number for us.
ERROR_INVALID_NAME = 123
'''
Windows-specific error code indicating an invalid pathname.

See Also
----------
https://msdn.microsoft.com/en-us/library/windows/desktop/ms681382%28v=vs.85%29.aspx
    Official listing of all such codes.
'''


def is_pathname_valid(pathname: str) -> bool:
    '''
    http://stackoverflow.com/questions/9532499/check-whether-a-path-is-valid-in-python-without-creating-a-file-at-the-paths-ta
    by Cecil Curry, mindshines.com

    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?


def is_path_creatable(pathname: str) -> bool:
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)


def is_path_exists_or_creatable(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS _and_
    either currently exists or is hypothetically creatable; `False` otherwise.

    This function is guaranteed to _never_ raise exceptions.
    '''
    try:
        # To prevent "os" module calls from raising undesirable exceptions on
        # invalid pathnames, is_pathname_valid() is explicitly called first.
        return is_pathname_valid(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname))
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught.
    except OSError:
        return False
