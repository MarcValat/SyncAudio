from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import stft

DEFAULT_N_FFT = 1024
DEFAULT_HOP = 256

# scipy.ndimage releases the GIL, so plain threads parallelize the median
# filters (~90% of analysis time) across cores.
_WORKERS = os.cpu_count() or 1
_pool = ThreadPoolExecutor(max_workers=_WORKERS, thread_name_prefix="syncaudio-median")


def _parallel_median_filter(mag: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``median_filter(mag, size)``, split into time chunks run in parallel.

    Each chunk is padded with the neighbouring frames the filter window
    reaches into, then trimmed back, so the result is bit-identical to a
    single call (the file's true edges keep scipy's own border handling).
    """
    frames = mag.shape[1]
    halo = size[1] // 2
    chunks = min(_WORKERS, max(1, frames // 256))
    if chunks == 1:
        return median_filter(mag, size=size)
    bounds = np.linspace(0, frames, chunks + 1, dtype=int)

    def run(i: int) -> np.ndarray:
        a, b = bounds[i], bounds[i + 1]
        lo, hi = max(0, a - halo), min(frames, b + halo)
        return median_filter(mag[:, lo:hi], size=size)[:, a - lo : b - lo]

    return np.concatenate(list(_pool.map(run, range(chunks))), axis=1)


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
    harmonic = _parallel_median_filter(mag, (1, harm_win))
    percussive = _parallel_median_filter(mag, (perc_win, 1))
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
