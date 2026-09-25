"""In-memory cache for the expensive part of track analysis.

Measured on a real 6-minute track: ffmpeg extraction (decode to PCM) takes
~0.3-0.4s, negligible; the envelope computation (STFT + harmonic/percussive
median-filter separation, in ``features.py``) took ~7s single-threaded,
~1.5s now that the median filters run across all cores (8 here) -- still
most of the total, and pure CPU-bound numpy/scipy work with no I/O to
speed up (so e.g. ``mkvextract`` wouldn't help: it only demuxes, and
decoding was never the bottleneck). ``align``/``segments``/``render`` each
re-extract and re-analyze their reference (and every candidate) from
scratch, even across separate calls on the exact same track -- this cache
lets a second request for the same (file, track, analysis window) skip
straight to the correlation math instead of redoing that work.

Concurrent requests for the *same* key (e.g. a background prefetch and a
user-triggered detection racing each other) are coalesced: only the first
caller actually extracts/analyzes, every other caller for that key blocks
on it and reuses its result instead of redundantly repeating the same ~7s
of work in parallel.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np

from syncaudio.features import extract_envelope
from syncaudio.ffmpeg_backend import extract_pcm
from syncaudio.models import AudioTrackSpec

# The one sample rate every analysis path (align, segments, render) uses --
# defined here, once, so they can't silently drift apart into separate
# literals that happen to match (they used to be).
ANALYSIS_SAMPLE_RATE = 16000

_MAX_ENTRIES = 64
_CacheKey = tuple[str, int, int, float, float | None]

_cache: OrderedDict[_CacheKey, tuple[np.ndarray, float]] = OrderedDict()
_inflight: dict[_CacheKey, threading.Event] = {}
_lock = threading.Lock()


def _key(spec: AudioTrackSpec, sample_rate: int, start: float, duration: float | None) -> _CacheKey:
    idx = spec.stream_index if spec.stream_index is not None else 0
    return (str(Path(spec.path).resolve()), idx, sample_rate, start, duration)


def get_envelope(
    spec: AudioTrackSpec,
    sample_rate: int = ANALYSIS_SAMPLE_RATE,
    start: float = 0.0,
    duration: float | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> tuple[np.ndarray, float]:
    """Return ``(envelope, frame_rate)`` for ``spec``, computing it only on a cache miss."""
    key = _key(spec, sample_rate, start, duration)

    while True:
        with _lock:
            cached = _cache.get(key)
            if cached is not None:
                _cache.move_to_end(key)
                log(f"[cache] {spec.raw} déjà analysée, réutilisation")
                return cached

            event = _inflight.get(key)
            if event is None:
                # We're first for this key: claim it, everyone else waits on us.
                event = threading.Event()
                _inflight[key] = event
                is_owner = True
            else:
                is_owner = False

        if is_owner:
            break

        log(f"[cache] {spec.raw} : analyse déjà en cours ailleurs, attente...")
        event.wait()
        # Loop back around: the owner either populated the cache (common
        # case, we'll hit it above) or failed (rare), in which case we
        # retry the whole thing and become the new owner ourselves.

    try:
        log(f"[extraction] {spec.raw} ...")
        pcm = extract_pcm(spec, sample_rate=sample_rate, start=start or None, duration=duration)
        log(f"[analyse] {spec.raw} : calcul du spectrogramme et de l'enveloppe...")
        result = extract_envelope(pcm, sample_rate)
        with _lock:
            _cache[key] = result
            _cache.move_to_end(key)
            while len(_cache) > _MAX_ENTRIES:
                _cache.popitem(last=False)
        return result
    finally:
        with _lock:
            _inflight.pop(key, None)
        event.set()


def clear() -> None:
    """Drop every cached entry (mainly for tests, to avoid cross-test leakage)."""
    with _lock:
        _cache.clear()
        _inflight.clear()
