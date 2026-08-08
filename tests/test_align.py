from __future__ import annotations

import numpy as np

from syncaudio.align import estimate_offset
from syncaudio.features import extract_envelope

SAMPLE_RATE = 16000


def _make_bed(duration_s: float, sr: int, seed: int, hits_per_second: float = 2.0) -> np.ndarray:
    """A common "music/SFX" bed: short broadband noise bursts (percussive hits)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    bed = np.zeros(n, dtype=np.float64)
    burst_len = int(0.05 * sr)
    envelope = np.hanning(burst_len)
    hit_times = rng.uniform(0, duration_s - 0.2, int(duration_s * hits_per_second))
    for t in hit_times:
        start = int(t * sr)
        if start + burst_len > n:
            continue
        bed[start : start + burst_len] += rng.standard_normal(burst_len) * envelope
    return bed


def _make_dialogue(duration_s: float, sr: int, seed: int, words_per_second: float = 1.5) -> np.ndarray:
    """Independent, voice-like content: sustained multi-tone bursts (harmonic-ish).

    Duration/amplitude/pitch vary per burst (like real speech) rather than
    repeating one fixed shape -- a uniform "comb" of identical pulses would
    cross-correlate with itself at many lags regardless of actual content,
    which is a fixture-realism artifact, not something a real dub exhibits.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    signal = np.zeros(n, dtype=np.float64)
    word_times = rng.uniform(0, duration_s - 0.5, int(duration_s * words_per_second))
    for t in word_times:
        start = int(t * sr)
        length = int(rng.uniform(0.08, 0.35) * sr)
        if start + length > n:
            continue
        local_t = np.arange(length) / sr
        freqs = rng.uniform(120, 700, rng.integers(2, 5))
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        tone = sum(np.sin(2 * np.pi * f * local_t + p) for f, p in zip(freqs, phases))
        envelope = np.hanning(length)
        gain = rng.uniform(0.15, 0.45)
        signal[start : start + length] += gain * tone * envelope
    return signal


def _shift(signal: np.ndarray, offset_samples: int) -> np.ndarray:
    shifted = np.zeros_like(signal)
    if offset_samples >= 0:
        shifted[offset_samples:] = signal[: len(signal) - offset_samples]
    else:
        shifted[: len(signal) + offset_samples] = signal[-offset_samples:]
    return shifted


def test_recovers_known_offset_despite_different_dialogue() -> None:
    duration_s = 20.0
    offset_s = 1.234

    bed = _make_bed(duration_s, SAMPLE_RATE, seed=1)
    shifted_bed = _shift(bed, int(round(offset_s * SAMPLE_RATE)))

    reference = bed + _make_dialogue(duration_s, SAMPLE_RATE, seed=2)
    candidate = shifted_bed + _make_dialogue(duration_s, SAMPLE_RATE, seed=3)

    ref_env, frame_rate = extract_envelope(reference, SAMPLE_RATE)
    cand_env, _ = extract_envelope(candidate, SAMPLE_RATE)

    estimate = estimate_offset(ref_env, cand_env, frame_rate)

    assert abs(estimate.offset_seconds - offset_s) < 0.05
    assert estimate.confidence > 0.2
    assert not estimate.ambiguous


def test_recovers_negative_offset() -> None:
    duration_s = 15.0
    offset_s = -0.6

    bed = _make_bed(duration_s, SAMPLE_RATE, seed=5)
    shifted_bed = _shift(bed, int(round(offset_s * SAMPLE_RATE)))

    reference = bed + _make_dialogue(duration_s, SAMPLE_RATE, seed=6)
    candidate = shifted_bed + _make_dialogue(duration_s, SAMPLE_RATE, seed=7)

    ref_env, frame_rate = extract_envelope(reference, SAMPLE_RATE)
    cand_env, _ = extract_envelope(candidate, SAMPLE_RATE)

    estimate = estimate_offset(ref_env, cand_env, frame_rate)

    assert abs(estimate.offset_seconds - offset_s) < 0.05


def test_unrelated_tracks_get_low_confidence() -> None:
    # Long enough that two independent, sparse burst trains are very unlikely
    # to line up by pure chance (short clips are inherently more prone to
    # this kind of coincidental false positive -- see README caveats).
    duration_s = 120.0
    track_a = _make_dialogue(duration_s, SAMPLE_RATE, seed=100)
    track_b = _make_dialogue(duration_s, SAMPLE_RATE, seed=200)

    env_a, frame_rate = extract_envelope(track_a, SAMPLE_RATE)
    env_b, _ = extract_envelope(track_b, SAMPLE_RATE)

    estimate = estimate_offset(env_a, env_b, frame_rate)

    assert estimate.confidence < 0.2
