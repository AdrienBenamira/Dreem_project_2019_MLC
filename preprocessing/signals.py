from typing import List, Union
import numpy as np
from scipy.signal import iirfilter, sosfilt

__all__ = ["BandPass", "ExtractBands", "ExtractSpectrum"]


class ExtractBands:
    def __init__(self, bands: Union[str, List[str]] = None):
        """
        Args:
            bands: bands to get. If None or '*', all available bands are returned.
        """
        self.extractors = {
            'delta': BandPass(freq_min=0, freq_max=4),
            'theta': BandPass(freq_min=4, freq_max=8),
            'alpha': BandPass(freq_min=8, freq_max=13),
            'beta': BandPass(freq_min=13, freq_max=22)
        }
        self.bands = self.extractors.keys() if (bands is None or bands == '*') else bands
        self.frequencies = [(self.extractors[band].freq_min, self.extractors[band].freq_max) for band in self.bands]

    def __call__(self, batch_signal):
        """
        Extract the different wave bands from the signal
        Args:
            signal: eeg signal. Shape of number_of_signals x signal_length

        Returns:
            List of all bands from the signal
        """
        signals = []
        for signal in batch_signal:
            s = []
            for band in self.bands:
                s.append(self.extractors[band](signal))
            signals.append(s)
        return np.array(signals)


class ExtractSpectrum:
    def __init__(self, window, sampling_freq=50):
        self.sampling_freq = sampling_freq
        self.window = window

    def __call__(self, signal):
        """
        Args:
            signal: signal. Either one batch of signals of dimension 1500 (sampled at 50hz) or all bands for one batch
             signals of dimension (batch_size x number_bands x 1500)
        """
        if len(signal.shape) == 2:
            return self.get_spectrum(signal)
        elif len(signal.shape) == 3:  # decomposed in bands
            features = []
            for band in range(signal.shape[1]):
                features.append(self.get_spectrum(signal[:, band]))
            return np.array(features)

    def get_spectrum(self, signal):
        range_signal = range(0, len(signal), self.window)
        d = np.zeros((signal.shape[0], len(list(range_signal)), self.window))
        for b in range(signal.shape[0]):  # for each signal in batch
            for i, k in enumerate(range_signal):
                sub_signal = signal[b, k:k + self.window]
                spectrum = abs(np.fft.fft(sub_signal) / self.sampling_freq)
                d[b, i, :] = spectrum
        return d


class BandPass:
    """
    Band pass filter
    """

    def __init__(self, freq_min: float, freq_max: float, sampl_freq: float = 50, order: int = 4):
        """
        Args:
            freq_min: min frequency
            freq_max: max frequency
            order: order of the filter
        """
        self.freq_min = freq_min
        self.freq_max = freq_max
        btype = 'band' if freq_min != 0 else 'low'
        nyquist_freq = 0.5 * sampl_freq
        freqs = [freq_min / nyquist_freq, freq_max / nyquist_freq] if btype == 'band' else freq_max / nyquist_freq
        self.sos = iirfilter(order, freqs, btype=btype, ftype='butter', output='sos')

    def __call__(self, signal):
        return sosfilt(self.sos, signal)
