# -*- coding: utf-8 -*-
"""
Patterns for finding info in seismic headers heuristically.

:copyright: 2016-22 Agile Scientific
:license: Apache 2.0
"""
import numpy as np


def flat(arr):
    """
    Finds flat things (could be zeros)
        ___________________________

    """
    arr = np.array(arr)
    if arr.size == 0:
        return False
    mean = np.repeat(np.mean(arr), arr.size)
    nonzero_residuals = np.nonzero(arr - mean)[0]
    return nonzero_residuals.size < arr.size/100


def zero(arr):
    """
    Finds flat things (flat and almost all zeros)
        ___________________________

    """
    arr = np.array(arr)
    nonzero_residuals = np.nonzero(np.zeros_like(arr) - arr)[0]
    return nonzero_residuals.size < arr.size/100


def monotonic(arr):
    """     /
    Finds  /
          /
         /
        /
       /
    """
    arr = np.array(arr)
    if flat(arr):
        return False
    # Second derivative is zero
    return zero(np.diff(np.diff(arr)))


def count_spikes(arr):
    """
    Counts the spikes in

    ____/\____/\____/\____/\____/\____/\

    """
    arr = np.array(arr)
    if (arr.size == 0) or flat(arr) or monotonic(arr):
        return 0
    return len(np.where(np.abs(arr) > arr.mean())[0])


def normalize(arr):
    mi, ma = np.amin(arr), np.amax(arr)
    return (arr - mi) / (ma - mi)


def spikes(arr):
    """
    True if array is a sequence of regularly-spaced spikes.

    ____/\____/\____/\____/\____/\____/\

    """
    arr = np.array(arr)
    if (arr.size == 0) or flat(arr) or monotonic(arr):
        return False
    arr = normalize(arr)
    spikes = np.where(arr > arr.mean())[0]
    rest = np.ones_like(arr, dtype=bool)
    rest[spikes] = False
    return flat(arr[rest]) and flat(np.diff(arr[spikes]))


def sawtooth(arr):
    """
    Finds    /|  /|  /|  /|  /|
            / | / | / | / | / |
           /  |/  |/  |/  |/  |
    """
    arr = np.array(arr)
    if monotonic(arr):
        return False
    if count_spikes(np.diff(arr)) < arr.size/100:
        return False
    return (arr[0] != arr[1]) and spikes(np.diff(arr))


def stairstep(arr):
    """                    _____
    Finds             _____|
                 _____|
            _____|
       _____|
    """
    arr = np.array(arr)
    if monotonic(arr):
        return False
    if count_spikes(np.diff(arr)) < arr.size/100:
        return False
    return (arr[0] == arr[1]) and spikes(np.diff(arr))
