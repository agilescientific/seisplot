# -*- coding: utf-8 -*-
"""
Seismic object for seisplot and beyond.

:copyright: 2016 Agile Geoscience
:license: Apache 2.0
"""
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import obspy

import utils
import patterns


class SeismicError(Exception):
    """
    Generic error class.
    """
    pass


class Seismic(object):

    def __init__(self, data, dtype=float, params=None):

        if params is None:
            params = {}

        self.params = params
        self.data = np.asarray(data, dtype=dtype)
        self.header = params.get('header', '')
        self.ntraces = params.get('ntraces', self.data.shape[0])
        self.inlines = params.get('inlines', None)
        self.xlines = params.get('xlines', None)
        self.dimensions = params.get('dimensions', ['i', 'x', 't'])
        self.ninlines = params.get('ninlines', 1)
        self.nxlines = params.get('nxlines', 0)
        self.nsamples = params.get('nsamples', self.data.shape[-1])
        self.tstart = params.get('tstart', 0)
        self.dt = params.get('dt', 0)

        if self.nsamples and self.nsamples != self.data.shape[-1]:
            t = self.nsamples
            self.nsamples = int(self.data.shape[-1])
            if t != self.nsamples:
                s = "Number of time samples changed to {} to match data."
                print(s.format(self.nsamples))

        # For when we have xline number but not inline number.
        # This happens when ObsPy reads a 3D.
        if self.ninlines == 1:
            if self.nxlines > 0:
                self.ninlines = int(self.data.shape[0] / self.nxlines)

        if self.ninlines > 1:
            x = self.nxlines
            self.nxlines = int(self.data.shape[0] / self.ninlines)
            if self.nxlines and x != self.nxlines:
                s = "data shape {} changed to {} to match data."
                print(s.format(self.data.shape, (self.ninlines, self.nxlines, self.data.shape[-1])))
            self.data = self.data.reshape((self.ninlines, self.nxlines, self.data.shape[-1]))

        # Make sure there are no singleton dimensions.
        self.data = np.squeeze(self.data)

        if self.inlines is None:
            self.inlines = np.linspace(1, self.ninlines, self.ninlines)
        if self.xlines is None:
            self.xlines = np.linspace(1, self.nxlines, self.nxlines)

        # Guarantee we're getting a unique list.
        self.inlines = np.unique(self.inlines)
        self.xlines = np.unique(self.xlines)

        self.tbasis = np.arange(0, self.nsamples * self.dt, self.dt)
        return

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def tend(self):
        return np.amax(self.tbasis)

    def trace_range(self, direction):
        if direction.lower()[0] == 'x':
            return self.inlines[0], self.inlines[-1]
        return self.xlines[0], self.xlines[-1]

    @classmethod
    def from_obspy(cls, stream, params=None):
        data = np.stack(t.data for t in stream.traces)
        if params is None:
            params = {}
        dt = params.get('dt', stream.binary_file_header.sample_interval_in_microseconds)

        # ndim param can force 2d or 3d data
        ndim = params.get('ndim', 0)
        if ndim:
            params.pop('ndim')

        # Make certain it winds up in seconds. Most likely 0.0005 to 0.008.
        while dt > 0.02:
            dt *= 0.001

        params['dt'] = dt

        # Since we have the headers, we can try to guess the geometry.
        threed = False

        # Get the sawtooth header field. In a perfect workd, only works for 3D.
        xlines = utils.get_pattern_from_stream(stream, patterns.sawtooth)
        if np.any(xlines) and (ndim != 2):
            threed = True
            nxlines = np.amax(xlines) - np.amin(xlines) + 1
            params['nxlines'] = params.get('nxlines') or nxlines
            params['xlines'] = params.get('xlines') or xlines
            params['dimensions'] = ['i', 'x', 't']
        else:
            xlines = utils.get_pattern_from_stream(stream, patterns.monotonic)
            if np.any(xlines):
                nxlines = np.amax(xlines) - np.amin(xlines) + 1
                params['nxlines'] = params.get('nxlines') or nxlines
                params['xlines'] = params.get('xlines') or xlines
            params['dimensions'] = ['i', 't']

        params['ninlines'] = 1
        if threed:
            inlines = utils.get_pattern_from_stream(stream, patterns.stairstep)
            if np.any(inlines):
                ninlines = np.amax(inlines) - np.amin(inlines) + 1
                params['ninlines'] = params.get('ninlines') or ninlines
                params['inlines'] = params.get('inlines') or inlines

        x = np.array(list(stream.textual_file_header.decode()))
        params['header'] = '\n'.join(''.join(row) for row in x.reshape((40, 80)))

        headers = {
            'elevation': 'receiver_group_elevation',
            'fold': 'number_of_horizontally_stacked_traces_yielding_this_trace',
            'water_depth': 'water_depth_at_group',
        }

        for k, v in headers.items():
            params[k] = [t.header.__dict__[v] for t in stream.traces]

        return cls(data, params=params)

    @classmethod
    def from_segy(cls, segy_file, params=None):
        stream = obspy.io.segy.segy._read_segy(segy_file, unpack_headers=True, headonly=True)
        return cls.from_obspy(stream, params=params)

    @classmethod
    def from_seismic(cls, seismic, n=0, direction='inline'):
        params = seismic.params.copy()
        if seismic.ndim == 2:
            return seismic
        if direction.lower()[0] == 'i':
            if n < 1:
                n *= seismic.nxlines
                n = int(np.floor(n))
            data = seismic.data.copy()[n, ...]
            params['dimensions'] = ['i', 't']
            params['inlines'] = seismic.inlines[n]
            params['ninlines'] = 1
            params['xlines'] = seismic.xlines
            params['nxlines'] = len(seismic.xlines)
        elif direction.lower()[0] == 'x':
            if n < 1:
                n *= seismic.ninlines
                n = int(np.floor(n))
            data = seismic.data.copy()[:, n, :]
            params['dimensions'] = ['x', 't']
            params['inlines'] = seismic.inlines
            params['ninlines'] = len(seismic.inlines)
            params['xlines'] = seismic.xlines[n]
            params['nxlines'] = 1
        elif direction.lower()[0] == 't':
            if n < 1:
                n *= seismic.nsamples
                n = int(np.floor(n))
            data = seismic.data.copy()[..., n]
            params['dimensions'] = ['i', 'x']
        else:
            raise SeismicError("No corresponding data.")
        return cls(data, params=params)

    @property
    def inlineidx(self):
        """
        The inline names of every trace.
        """
        m = np.meshgrid(np.unique(self.inlines), np.unique(self.xlines))
        return m[0].T.flatten()

    @property
    def xlineidx(self):
        """
        The inline names of every trace.
        """
        m = np.meshgrid(np.unique(self.inlines), np.unique(self.xlines))
        return m[1].T.flatten()

    @property
    def olineidx(self):
        """
        The other-line numbers.
        """
        return self.xlineidx if self.dimensions[0] == 'i' else self.inlineidx

    @property
    def slineidx(self):
        """
        The self-line numbers.
        """
        return self.inlineidx if self.dimensions[0] == 'i' else self.xlineidx

    @property
    def slabel(self):
        """
        The self-label (what am I?).
        """
        return self.dimensions[0]

    @property
    def xlabel(self):
        """
        What you'd label the x-axis. If this is an inline, it'd be xline.
        """
        return 'x' if self.dimensions[0] == 'i' else 'i'

    @property
    def ylabel(self):
        return self.dimensions[-1]

    @staticmethod
    def spectrum(signal, fs, taper=True):
        if taper:
            windowed = signal * np.blackman(len(signal))
        else:
            windowed = signal
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

    def plot_spectrum(self,
                      ax=None,
                      tickfmt=None,
                      ntraces=20,
                      fontsize=10,
                      colour='k'):
        """
        Plot a power spectrum.
        w is window length for smoothing filter
        """
        if tickfmt is None:
                # Set the tickformat.
            tickfmt = mtick.FormatStrFormatter('%.0f')

        if ax is None:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)

        trace_indices = utils.get_trace_indices(self.data.shape[:-1],
                                                ntraces,
                                                random=True)
        fs = 1 / self.dt

        specs, peaks, mis, mas = [], [], [], []
        for ti in trace_indices:
            try:
                # 3D
                trace = self.data[ti[0], ti[1], :]
            except IndexError:
                # 2D
                trace = self.data[ti, :]

            if sum(trace) == 0:
                continue

            f, amp, fmi, fma = self.spectrum(trace, fs)

            peak = f[np.argmax(amp)]

            specs.append(amp)
            peaks.append(peak)
            mis.append(fmi)
            mas.append(fma)

        spec = np.nanmean(np.dstack(specs), axis=-1)
        spec = np.squeeze(spec)
        db = 20 * np.log10(spec)
        db = db - np.amax(db)
        f_peak = np.mean(peaks)
        f_min = np.amin(mis)
        f_max = np.amax(mas)
        dt = 1000 // fs
        f_nyquist = fs // 2

        statstring = "\nMin: {:.1f} Hz\nPeak: {:.1f} Hz\nMax: {:.1f} Hz\nNyquist ({} ms): {} Hz"
        stats = statstring.format(f_min, f_peak, f_max, dt, f_nyquist)

        ax.plot(f, db, lw=0)  # Plot invisible line to get the min
        y_min = ax.get_yticks()[0]
        ax.fill_between(f, y_min, db, lw=0, facecolor=colour, alpha=0.6)
        ax.set_xlabel('frequency [Hz]', fontsize=fontsize - 4)
        ax.xaxis.set_label_coords(0.5, -0.12)
        ax.set_xlim([0, np.amax(f)])
        ax.set_xticklabels(ax.get_xticks(), fontsize=fontsize - 4)
        ax.set_yticklabels(ax.get_yticks(), fontsize=fontsize - 4)
        ax.set_ylabel('power [dB]', fontsize=fontsize - 4)
        ax.text(.98, .95, 'AMPLITUDE SPECTRUM',
                horizontalalignment='right',
                verticalalignment='top',
                fontweight='bold',
                color=colour,
                transform=ax.transAxes, fontsize=fontsize - 3)
        ax.text(.98, .95, stats,
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes, fontsize=fontsize - 3)
        ax.yaxis.set_major_formatter(tickfmt)
        ax.xaxis.set_major_formatter(tickfmt)

        ax.grid('on')
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle('-')
            line.set_alpha(0.2)

        return ax

    def get_data(self, l=1, direction=None):
        if self.ndim < 3:
            return self.data
        if (direction is None) or (direction.lower()[0] == 'i'):
            if l < 1:
                l *= self.ninlines
            return self.data[int(l), :, :]
        else:
            if l < 1:
                l *= self.nxlines
            return self.data[:, int(l), :]

    inline = partial(get_data, direction='i')
    xline = partial(get_data, direction='x')

    def wiggle_plot(self, l=1, direction='i',
                    ax=None,
                    skip=1,
                    perc=99.0,
                    gain=1.0,
                    rgb=(0, 0, 0),
                    alpha=0.5,
                    lw=0.2,
                    ):
        """
        Plots wiggle traces of seismic data. Skip=1, every trace, skip=2, every
        second trace, etc.
        """
        if ax is None:
            fig = plt.figure(figsize=(16, 8))
            ax = fig.add_subplot(111)

        data = self.get_data(l, direction)
        rgba = list(rgb) + [alpha]
        sc = np.percentile(data, perc)  # Normalization factor
        wigdata = data[::skip, :]
        xpos = np.arange(self.ntraces)[::skip]

        for x, trace in zip(xpos, wigdata):
            # Compute high resolution trace.
            amp = gain * trace / sc + x
            t = 1000 * self.tbasis
            hypertime = np.linspace(t[0], t[-1], (10 * t.size - 1) + 1)
            hyperamp = np.interp(hypertime, t, amp)

            # Plot the line, then the fill.
            ax.plot(hyperamp, hypertime, 'k', lw=lw)
            ax.fill_betweenx(hypertime, hyperamp, x,
                             where=hyperamp > x,
                             facecolor=rgba,
                             lw=0,
                             )

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


class Seismic2D(Seismic):
    def __init__(self, data, dtype=float, params=None):

        # First generate the parent object.
        super().__init__(data, dtype, params)

        self.ndim = 2


class Seismic3D(Seismic):
    def __init__(self, data, dtype=float, params=None):

        # First generate the parent object.
        super().__init__(data, dtype, params)

        self.ndim = 3
