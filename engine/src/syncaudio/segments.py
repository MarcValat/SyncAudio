from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from syncaudio.align import estimate_offset

DEFAULT_WINDOW_S = 30.0
DEFAULT_HOP_S = 10.0
DEFAULT_MARGIN_S = 8.0

# Boundary refinement pass (see refine_segments): a small window/hop, applied
# only in a zoomed-in neighbourhood of each coarse boundary, localizes a jump
# far more precisely than the coarse pass alone -- a coarse window straddles
# up to a whole DEFAULT_WINDOW_S of mixed before/after content near the true
# jump, which is what limits the coarse pass's boundary precision.
_REFINE_ZOOM_S = 45.0
_REFINE_WINDOW_S = 8.0
_REFINE_HOP_S = 2.0

# A jump between consecutive windows bigger than this (and sustained, not a
# one-window blip) starts a new segment.
_JUMP_THRESHOLD_S = 0.75
# If a single line fits every window within this tolerance, the whole track
# is one segment (constant offset or pure drift) rather than several. Looser
# than a single global estimate's precision: each window sees far less audio
# than a full-track analysis, so its own estimate is noisier.
_RESIDUAL_TOL_S = 0.4
# Below this spread between a segment's start/end offset, treat it as a
# constant shift rather than drift (looser than render.py's threshold: these
# per-window estimates come from shorter, noisier slices).
_DRIFT_EPS_S = 0.2


@dataclass(frozen=True)
class WindowOffset:
    """Offset estimate local to one time window (see ``windowed_offsets``)."""

    time_s: float
    offset_seconds: float
    confidence: float
    ambiguous: bool


def windowed_offsets(
    ref_env: np.ndarray,
    cand_env: np.ndarray,
    frame_rate: float,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    margin_s: float = DEFAULT_MARGIN_S,
    search_start_s: float = 0.0,
    search_end_s: float | None = None,
) -> list[WindowOffset]:
    """Slide a window across the reference envelope, estimating a local offset in each.

    Unlike a single global ``estimate_offset`` call (which assumes one
    constant offset for the whole track), this recovers how the offset
    *changes over time* -- the building block for detecting drift (a
    steadily changing offset) and discontinuous jumps (a suddenly changing
    one), rather than just a single constant shift.

    Each reference window of ``window_s`` seconds is searched against a
    *wider* candidate window (padded by ``margin_s`` on each side), so a
    local offset up to ``margin_s`` away from zero can still be found even
    though the two envelopes are sliced independently per window.

    ``search_start_s``/``search_end_s`` restrict where windows are placed
    (default: the whole track) -- used to re-run this locally, at a finer
    resolution, around a boundary already found by a coarser pass (see
    ``refine_segments``) instead of paying that resolution everywhere.
    """
    window_frames = int(round(window_s * frame_rate))
    hop_frames = int(round(hop_s * frame_rate))
    margin_frames = int(round(margin_s * frame_rate))
    n_ref = len(ref_env)
    end_frame = n_ref if search_end_s is None else min(n_ref, int(round(search_end_s * frame_rate)))

    results: list[WindowOffset] = []
    i = int(round(search_start_s * frame_rate))
    while i + window_frames <= end_frame:
        ref_slice = ref_env[i : i + window_frames]
        cand_start = max(0, i - margin_frames)
        cand_end = min(len(cand_env), i + window_frames + margin_frames)
        cand_slice = cand_env[cand_start:cand_end]

        if cand_slice.size >= window_frames:
            estimate = estimate_offset(ref_slice, cand_slice, frame_rate)
            # estimate_offset assumes both slices start at the same instant;
            # cand_slice actually starts (cand_start - i) frames away from
            # ref_slice, so that difference has to be added back in.
            true_offset = estimate.offset_seconds + (cand_start - i) / frame_rate
            results.append(
                WindowOffset(
                    time_s=i / frame_rate,
                    offset_seconds=true_offset,
                    confidence=estimate.confidence,
                    ambiguous=estimate.ambiguous,
                )
            )
        i += hop_frames
    return results


@dataclass(frozen=True)
class Segment:
    """One piece of the timeline with its own correction.

    ``offset_start``/``offset_end`` are the candidate's offset (same sign
    convention as ``estimate_offset``: positive = lags) at the start/end of
    this segment. Equal (within ``_DRIFT_EPS_S``) means a constant shift
    (pad/trim, as in ``render.correction_filter``); different means linear
    drift across the segment (needs a time-stretch, not yet implemented).
    """

    start_s: float
    end_s: float
    offset_start: float
    offset_end: float

    @property
    def is_drift(self) -> bool:
        return abs(self.offset_end - self.offset_start) > _DRIFT_EPS_S

    @property
    def mean_offset(self) -> float:
        return (self.offset_start + self.offset_end) / 2.0


def classify_segments(
    windows: Sequence[WindowOffset],
    total_duration_s: float,
    *,
    jump_threshold_s: float = _JUMP_THRESHOLD_S,
    residual_tol_s: float = _RESIDUAL_TOL_S,
) -> list[Segment]:
    """Turn a windowed offset series into a small number of correction segments.

    ``ambiguous`` windows (competing correlation peak -- see ``align.py``) are
    excluded from deciding both the linear fit and the jump boundaries: at
    the window scale, real content residuals mean *confidence* alone stays
    low even for correct estimates (see README caveats), so ``ambiguous`` is
    used as the reliability signal instead of a confidence threshold.

    First tries a single line through every usable window; if that already
    explains the data within ``residual_tol_s``, the whole track is one
    segment (flat = constant offset, sloped = pure drift). Otherwise, splits
    into segments wherever the offset jumps by more than ``jump_threshold_s``
    and stays there, fitting a line within each.
    """
    usable = [w for w in windows if not w.ambiguous]
    if len(usable) < 2:
        usable = list(windows)
    if not usable:
        return [Segment(0.0, total_duration_s, 0.0, 0.0)]

    times = np.array([w.time_s for w in usable])
    offsets = np.array([w.offset_seconds for w in usable])

    if len(usable) >= 2:
        slope, intercept = np.polyfit(times, offsets, 1)
        residuals = offsets - (intercept + slope * times)
        if np.max(np.abs(residuals)) <= residual_tol_s:
            return [
                Segment(
                    start_s=0.0,
                    end_s=total_duration_s,
                    offset_start=float(intercept),
                    offset_end=float(intercept + slope * total_duration_s),
                )
            ]

    groups: list[list[WindowOffset]] = [[usable[0]]]
    for w in usable[1:]:
        current = groups[-1]
        median = float(np.median([x.offset_seconds for x in current]))
        if abs(w.offset_seconds - median) > jump_threshold_s:
            groups.append([w])
        else:
            current.append(w)

    segments: list[Segment] = []
    for idx, group in enumerate(groups):
        seg_start = 0.0 if idx == 0 else (groups[idx - 1][-1].time_s + group[0].time_s) / 2.0
        seg_end = (
            total_duration_s
            if idx == len(groups) - 1
            else (group[-1].time_s + groups[idx + 1][0].time_s) / 2.0
        )
        if len(group) >= 2:
            g_times = np.array([g.time_s for g in group])
            g_offsets = np.array([g.offset_seconds for g in group])
            g_slope, g_intercept = np.polyfit(g_times, g_offsets, 1)
            offset_start = float(g_intercept + g_slope * seg_start)
            offset_end = float(g_intercept + g_slope * seg_end)
        else:
            offset_start = offset_end = group[0].offset_seconds
        segments.append(Segment(seg_start, seg_end, offset_start, offset_end))
    return segments


def refine_boundary(
    ref_env: np.ndarray,
    cand_env: np.ndarray,
    frame_rate: float,
    approx_time_s: float,
    offset_before: float,
    offset_after: float,
    *,
    zoom_s: float = _REFINE_ZOOM_S,
    window_s: float = _REFINE_WINDOW_S,
    hop_s: float = _REFINE_HOP_S,
) -> float:
    """Pinpoint a jump more precisely than the coarse windowed pass did.

    Re-runs the windowed search with a much smaller window/hop, but only in
    a zone around ``approx_time_s``: cheap, because it only touches a small
    neighbourhood, and more precise there, because a small window straddles
    far less of the actual transition than the coarse ``DEFAULT_WINDOW_S``
    one did (that straddling -- mixed before/after content in one window --
    is what limits the coarse pass's boundary precision to roughly its
    window size).

    Individual fine windows can still be noisy right at the transition
    (mixed content briefly confuses the correlation), so rather than trust
    the first window that crosses to the other side -- fragile, a single
    stray reading can fake a crossing -- this picks the split point that
    minimizes the *total* variance on each side across every fine window in
    the zone, a fit anchored on the whole picture rather than one sample.
    """
    margin_s = abs(offset_after - offset_before) / 2.0 + 3.0
    start = max(0.0, approx_time_s - zoom_s)
    end = approx_time_s + zoom_s
    fine = windowed_offsets(
        ref_env,
        cand_env,
        frame_rate,
        window_s=window_s,
        hop_s=hop_s,
        margin_s=margin_s,
        search_start_s=start,
        search_end_s=end,
    )
    usable = [w for w in fine if not w.ambiguous]
    if len(usable) < 4:
        return approx_time_s

    times = np.array([w.time_s for w in usable])
    offsets = np.array([w.offset_seconds for w in usable])

    best_cost = None
    best_m = None
    for m in range(2, len(usable) - 1):
        left, right = offsets[:m], offsets[m:]
        cost = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
        if best_cost is None or cost < best_cost:
            best_cost, best_m = cost, m
    if best_m is None:
        return approx_time_s

    boundary = (times[best_m - 1] + times[best_m]) / 2.0
    return float(np.clip(boundary, start, end))


def refine_segments(
    ref_env: np.ndarray,
    cand_env: np.ndarray,
    frame_rate: float,
    segments: Sequence[Segment],
) -> list[Segment]:
    """Refine every internal boundary of ``segments`` with ``refine_boundary``."""
    if len(segments) < 2:
        return list(segments)

    refined = [segments[0]]
    for cur in segments[1:]:
        prev = refined[-1]
        original_boundary = prev.end_s
        boundary = refine_boundary(
            ref_env, cand_env, frame_rate,
            approx_time_s=original_boundary,
            offset_before=prev.offset_end,
            offset_after=cur.offset_start,
        )
        # Refinement only searches a local zoom window, but a noisy/spurious
        # coarse segment (e.g. a one-window outlier) can still send the fine
        # crossing search off to a nonsensical point -- never let a boundary
        # cross into a neighbouring segment's own span, and fall back to the
        # coarse estimate rather than emit an invalid (non-monotonic) range.
        if not (prev.start_s < boundary < cur.end_s):
            boundary = original_boundary
        refined[-1] = dataclasses.replace(prev, end_s=boundary)
        refined.append(dataclasses.replace(cur, start_s=boundary))
    return refined
