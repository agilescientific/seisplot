#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Seismic object for seisplot and beyond.

:copyright: 2016 Agile Geoscience
:license: Apache 2.0
"""
import matplotlib.pyplot as plt
import numpy as np
import obspy

import utils
import patterns


class Seismic(object):

    def __init__(self, data, dtype=float, params=None):
        
        if params is None:
            params = {}
        
        self.data = np.asarray(data, dtype=dtype)
        self.header = params.get('header', '')
        self.traces = params.get('traces', self.data.shape[0])
        self.inlines = params.get('inlines', None)
        self.xlines = params.get('xlines', None)
        self.ninlines = params.get('ninlines', 1)
        self.nxlines = params.get('nxlines', 0)
        self.tsamples = params.get('tsamples', 0)
        self.dt = params.get('dt', 0)

        if self.tsamples and self.tsamples != self.data.shape[-1]:
            t = self.tsamples
            self.tsamples = int(self.data.shape[-1])
            if t != self.tsamples:
                s = "Number of time samples changed to {} to match data."
                print(s.format(self.tsamples))

        # For when we have xline number but not inline number.
        # This happens when ObsPy reads a 3D.
        if self.ninlines == 1:
            if self.nxlines > 0:
                self.ninlines = int(self.data.shape[0] / self.nxlines)
        
        if self.ninlines > 1:
            x = self.nxlines
            self.nxlines = int(self.data.shape[0] / self.ninlines)
            if self.nxlines and x != self.nxlines:
                s = "nxlines changed to {} to match data."
                print(s.format(self.nxlines))
            self.data = self.data.reshape((self.ninlines, self.nxlines, self.data.shape[-1]))

        return
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim

    @classmethod
    def from_obspy(cls, stream, params=None):
        data = np.stack(t.data for t in stream.traces)
        if params is None:
            params = {}
        params['dt'] = params.get('dt', stream.binary_file_header.sample_interval_in_microseconds / 1000)
        
        # Since we have the headers, etc, we can get some info.
        if np.any(utils.get_pattern_from_stream(stream, patterns.sawtooth)):
            xlines = utils.get_pattern_from_stream(stream, patterns.sawtooth)
            nxlines = np.amax(xlines) - np.amin(xlines) + 1
            params['nxlines'] = params.get('nxlines') or nxlines
            params['xlines'] = params.get('xlines') or xlines

        x = np.array(list(stream.textual_file_header))  # Shouldn't need to .decode()
        params['header'] = '\n'.join(''.join(row) for row in x.reshape((40, 80)))

        return cls(data, params=params)

    @classmethod
    def from_segy(cls, segy_file, params=None):
        stream = obspy.io.segy.segy._read_segy(segy_file, unpack_headers=True)
        return cls.from_obspy(stream, params=params)

    def spectrum(self, signal, fs):
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
    
    def plot_spectrum(self, ax=None, tickfmt=None, ntraces=10, fontsize=10):
        """
        Plot a power spectrum.
        w is window length for smoothing filter
        """
        if ax is None:
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot(111)

        trace_indices = utils.get_trace_indices(self.data.shape[1],
                                                ntraces,
                                                random=True)
        fs = 1 / self.dt

        specs, peaks, mis, mas = [], [], [], []
        for ti in trace_indices:
            trace = self.data[:, ti]
            f, amp, fmi, fma = self.spectrum(trace, fs)

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

        ax.plot(f, db, lw=0)  # Plot invisible line to get the min
        y_min = ax.get_yticks()[0]
        ax.fill_between(f, y_min, db, lw=0, facecolor='k', alpha=0.5)
        ax.set_xlabel('frequency [Hz]', fontsize=fontsize - 4)
        ax.xaxis.set_label_coords(0.5, -0.12)
        ax.set_xlim([0, np.amax(f)])
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize - 4)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize - 4)
        ax.set_ylabel('power [dB]', fontsize=fontsize - 4)
        ax.text(.98, .95, 'AMPLITUDE SPECTRUM'+stats,
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=ax.transAxes, fontsize=fontsize - 3)
        ax.yaxis.set_major_formatter(tickfmt)
        ax.xaxis.set_major_formatter(tickfmt)
        ax.grid('on')
        return ax
    
    def plot(self, slc=None):
        if slc is None:
            slc = self.data.shape[0] // 2
        vm = np.percentile(self.data, 99)
        imparams = {'interpolation': 'none',
                  'cmap': "gray",
                  'vmin': -vm,
                  'vmax': vm,
                  'aspect': 'auto'
                 }
        if self.ndim == 1:
            plt.plot(self.data)
        elif self.ndim == 2:
            plt.imshow(self.data.T, **imparams)
            plt.colorbar()
        else:
            plt.imshow(self.data[slc].T, **imparams)
            plt.colorbar()
        plt.show()
        return