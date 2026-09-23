from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

# Two peaks within this many seconds of each other are treated as "the same"
# peak rather than a competing, ambiguous match.
_SECONDARY_PEAK_MIN_DISTANCE_S = 1.0
_SECONDARY_PEAK_RATIO = 0.85

# Lags are only considered when the two envelopes overlap by at least this
# fraction of the shorter one. Two things motivate keeping this high (rather
# than allowing any partial overlap): (1) the correlation-vs-overlap-count
# normalization is an approximation that gets noisier as the window shrinks,
# and (2) mixing lags with very different overlap sizes into one background
# population for the confidence z-score below violates its implicit
# equal-variance assumption, inflating apparent confidence at partial-overlap
# lags. Keeping the search band close to full overlap keeps the compared
# lags statistically comparable. This does mean only offsets up to roughly
# (1 - fraction) * shorter track's duration are found -- ample for realistic
# dub sync (seconds to minutes) relative to a track that is minutes to hours
# long.
_MIN_OVERLAP_FRACTION = 0.9

# Peak-prominence excess (in standard deviations, after subtracting the
# chance-maximum expected from testing many lags -- see _peak_confidence)
# that maps to confidence 0 and 1 respectively.
_EXCESS_FOR_ZERO_CONFIDENCE = 3.0
_EXCESS_FOR_FULL_CONFIDENCE = 7.0


@dataclass(frozen=True)
class OffsetEstimate:
    offset_seconds: float
    confidence: float
    ambiguous: bool


def _standardize(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def estimate_offset(
    reference_envelope: np.ndarray,
    candidate_envelope: np.ndarray,
    frame_rate: float,
) -> OffsetEstimate:
    """Estimate how much the candidate envelope is shifted vs. the reference.

    Positive ``offset_seconds`` means the candidate lags behind the reference
    (its matching content appears later); negative means it leads.
    """
    ref = _standardize(reference_envelope)
    cand = _standardize(candidate_envelope)
    n_ref, n_cand = len(ref), len(cand)

    raw_corr = fftconvolve(cand, ref[::-1], mode="full")
    overlap = np.convolve(np.ones(n_cand), np.ones(n_ref), mode="full")
    norm_corr = raw_corr / np.maximum(overlap, 1.0)

    min_overlap = _MIN_OVERLAP_FRACTION * min(n_ref, n_cand)
    search_curve = np.where(overlap >= min_overlap, norm_corr, -np.inf)

    peak_idx = int(np.argmax(search_curve))
    delta = _parabolic_refine(search_curve, peak_idx)
    offset_samples = (peak_idx - (n_ref - 1)) + delta

    confidence = _peak_confidence(search_curve, peak_idx)
    ambiguous = _has_competing_peak(search_curve, peak_idx, frame_rate)

    return OffsetEstimate(
        offset_seconds=offset_samples / frame_rate,
        confidence=confidence,
        ambiguous=ambiguous,
    )


def _parabolic_refine(curve: np.ndarray, peak_idx: int) -> float:
    """Sub-frame peak refinement via parabolic interpolation of neighbours."""
    if peak_idx <= 0 or peak_idx >= len(curve) - 1:
        return 0.0
    y0, y1, y2 = curve[peak_idx - 1 : peak_idx + 2]
    if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2)):
        return 0.0
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return 0.0
    delta = 0.5 * (y0 - y2) / denom
    return float(np.clip(delta, -1.0, 1.0))


def _peak_confidence(curve: np.ndarray, peak_idx: int) -> float:
    """How much the best lag stands out from the rest of the search curve.

    Many lags are tested (one per frame of overlap), so even under the null
    hypothesis of two unrelated tracks, the *maximum* of that many noisy
    correlation values is expected to sit several standard deviations above
    the mean just by chance -- extreme value theory puts that expected chance
    maximum at roughly ``sqrt(2 * ln(N))`` for ``N`` samples. A raw z-score
    ignores this and reads as spuriously confident for unrelated, bursty
    audio (many candidate lags -> a large expected chance maximum). Confidence
    is instead based on how far the peak's z-score exceeds that chance
    maximum, squashed to [0, 1].
    """
    finite = curve[np.isfinite(curve)]
    n = finite.size
    background_std = finite.std()
    if background_std < 1e-12 or n < 2:
        return 0.0
    z_score = (curve[peak_idx] - finite.mean()) / background_std
    expected_chance_max = np.sqrt(2.0 * np.log(n))
    excess = z_score - expected_chance_max
    span = _EXCESS_FOR_FULL_CONFIDENCE - _EXCESS_FOR_ZERO_CONFIDENCE
    return float(np.clip((excess - _EXCESS_FOR_ZERO_CONFIDENCE) / span, 0.0, 1.0))


def _has_competing_peak(curve: np.ndarray, peak_idx: int, frame_rate: float) -> bool:
    min_distance = max(1, int(_SECONDARY_PEAK_MIN_DISTANCE_S * frame_rate))
    peak_value = curve[peak_idx]
    if peak_value <= 0:
        return True
    threshold = _SECONDARY_PEAK_RATIO * peak_value
    lo = max(0, peak_idx - min_distance)
    hi = min(len(curve), peak_idx + min_distance + 1)
    far_region = np.concatenate([curve[:lo], curve[hi:]])
    return bool(far_region.size and far_region.max() >= threshold)
