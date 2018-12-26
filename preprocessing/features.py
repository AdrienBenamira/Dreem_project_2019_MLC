import numpy as np
from typing import Union, List
from preprocessing import ExtractBands

__all__ = ["ExtractFeatures"]


class ExtractFeatures:
    def __init__(self, features=None, bands: Union[str, List[str]] = None, window=100, sampling_freq=50):
        """

        Args:
            bands: bands to use if precised. Otherwise, does not separate in bands
            window: size of the sliding window
            sampling_freq:
            features: list of features to use. If None, all of them. Available features:
                - min
                - max
                - frequency
                - energy
                - mmd: maximum-minimum distance
                - esis: EnergySis
                - band-usage: How much each band is used. At each time t, the band b is used if it is the band with
                    highest amplitude.
        """
        self.name = "extract_" + "_".join(features) + "_features" + (
            "with_" + "_".join(bands) + "_bands" if bands is not None else "")
        self.window = window
        self.sampling_freq = sampling_freq
        available_features = {
            'min': lambda s, _, __, k: min(s),
            'max': lambda s, _, __, k: max(s),
            'frequency': self.get_frequency,
            'energy': self.get_energy,
            'mmd': self.get_mmd,
            'esis': self.get_esis,
            'band-usage': self.get_band_usage
        }
        features = features if features is not None else available_features.keys()
        self.features = [available_features[feature] for feature in features]
        self.bands = bands
        self.extract_bands = ExtractBands(self.bands) if self.bands is not None else None
        self.band_usage = None

    def __call__(self, signal, target):
        """
        Args:
            signal: signal. Either one batch of signals of dimension 1500 (sampled at 50hz) or all bands for one batch of
             signals of dimension (batch_size x number_bands x 1500)
        """
        if self.extract_bands is not None:
            signal = self.extract_bands(signal, target)
        features = None
        if len(signal.shape) == 2:
            features = self.get_features(signal, None, signal)
        elif len(signal.shape) == 3:  # decomposed in bands
            features = []
            for band in range(signal.shape[1]):
                features.append(self.get_features(signal[:, band], band, signal))
            features = np.array(features)
            # features = features.reshape((features.shape[0] * features.shape[1]))
        return features

    def get_features(self, signal, band, full_signal):
        """
        Args:
            signal: shape (batch_size x signal size)
            band: band key
            full_signal: shape (batch_size x bands x signal size) if bands otherwise same as signal
        """
        features = [[feature(signal[k], band, full_signal, k) for feature in self.features] for k in
                    range(signal.shape[0])]
        return np.array(features)

    def get_spectrum(self, signal):
        spectrum = np.fft.fft(signal) / self.sampling_freq
        freqs = np.fft.fftfreq(signal.shape[0])
        return spectrum, freqs * self.sampling_freq

    def get_frequency(self, signal, band, signals, k):
        spectrum, freqs = self.get_spectrum(signal)
        threshold = 0.99 * max(abs(spectrum))
        mask = abs(spectrum) >= threshold
        # spectrum = spectrum[mask]
        freqs = freqs[mask]
        mask_pos = freqs >= 0
        return 0 if len(abs(freqs[mask_pos])) == 0 else abs(freqs[mask_pos])[0]

    def get_energy(self, signal, band, signals, k):
        return sum(np.power(signal, 2)) / len(signal)

    def get_mmd(self, signal, band, signals, k):
        """
        Feature from "Sleep Stage Classification Using EEG Signal Analysis:
            A Comprehensive Survey and New Investigation", Aboalayon et al.
        """
        range_signal = range(0, len(signal), self.window)
        d = np.zeros(len(list(range_signal)))
        non_zeros_values = []
        to_add = 0
        for i, k in enumerate(range_signal):
            sub_signal = signal[k:k + self.window]
            arg_min = np.argmin(sub_signal)
            arg_max = np.argmax(sub_signal)
            if max(sub_signal) > 1000 or min(sub_signal) < -1000:  # Remove if spike
                d[i] = 0
                to_add += 1
            else:
                d[i] = np.sqrt(
                    (max(sub_signal) - min(sub_signal)) ** 2 + ((arg_max - arg_min) / self.sampling_freq) ** 2)
                non_zeros_values.append(d[i])
        if to_add and len(non_zeros_values) > 0:
            to_add *= np.mean(non_zeros_values)
        return d.sum() + to_add

    def get_esis(self, signal, band, signals, k):
        """
        Feature from "Sleep Stage Classification Using EEG Signal Analysis:
            A Comprehensive Survey and New Investigation", Aboalayon et al.
        """
        assert band is not None, "Need to decompose in bands for esis feature"
        freq_min, freq_max = self.extract_bands.frequencies[band]
        freq = (freq_min + freq_max) / 2
        return self.get_energy(signal, band, signals, k) * freq * self.window

    def get_inflexion(self, signal, band, signals, k):
        h = 1 / self.sampling_freq
        num_inflexion = 0
        for k in range(1, len(signal) - 1):
            d = (signal[k - 1] + signal[k + 1] - 2 * signal[k]) / (2 * h)
            if abs(d) < 0.01:
                num_inflexion += 1
        return num_inflexion

    def get_band_usage(self, signal, band, signals, k):
        assert len(signals.shape) == 3, "Band usage needs to be extracted in bands"
        range_signal = range(0, len(signal), self.window)
        if self.band_usage is None:  # We compute everything first
            self.band_usage = np.zeros((signals.shape[0], signals.shape[1], len(range_signal)))
            for i in range(signals.shape[0]):  # For all batches
                for k, t in enumerate(range_signal):  # we go through the time series
                    subsignals = signals[:, :, t:t+self.window]
                    values_bands = np.array([subsignals[i, b].max() for b in range(signals.shape[1])])
                    band_key = np.argmax(values_bands)
                    self.band_usage[i, band_key, k] = 1
        return self.band_usage[k, band].sum() / self.band_usage[k].sum()
