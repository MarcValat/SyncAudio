from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import stft

DEFAULT_N_FFT = 1024
DEFAULT_HOP = 256


def _stft_magnitude(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    _, _, zxx = stft(signal, window="hann", nperseg=n_fft, noverlap=n_fft - hop)
    return np.abs(zxx)


def _percussive_component(mag: np.ndarray, harm_win: int = 17, perc_win: int = 17) -> np.ndarray:
    """Isolate the percussive (transient) part of a spectrogram via median filtering.

    Music/SFX transients (hits, impacts) are sparse across time but spread
    across frequency, while harmonic content (sustained tones, vowels in
    speech) is smooth across time but narrow in frequency — median-filtering
    each way and soft-masking separates them (Fitzgerald, 2010).
    """
    harmonic = median_filter(mag, size=(1, harm_win))
    percussive = median_filter(mag, size=(perc_win, 1))
    eps = 1e-10
    mask = percussive / (percussive + harmonic + eps)
    return mag * mask


def _onset_envelope(percussive_mag: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux: a 1D onset-strength signal per frame."""
    log_mag = np.log1p(percussive_mag)
    flux = np.diff(log_mag, axis=1)
    flux = np.maximum(flux, 0.0)
    env = flux.sum(axis=0)
    return np.concatenate(([0.0], env))


def extract_envelope(
    pcm: np.ndarray,
    sample_rate: int,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
) -> tuple[np.ndarray, float]:
    """Compute a language-robust onset envelope for a mono PCM signal.

    Returns ``(envelope, frame_rate)`` where ``frame_rate`` is the number of
    envelope frames per second (so downstream lags convert cleanly to
    seconds).
    """
    mag = _stft_magnitude(pcm, n_fft=n_fft, hop=hop)
    percussive = _percussive_component(mag)
    env = _onset_envelope(percussive)
    frame_rate = sample_rate / hop
    return env, frame_rate
