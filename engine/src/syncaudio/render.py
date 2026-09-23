from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from syncaudio.align import estimate_offset
from syncaudio.features import extract_envelope
from syncaudio.ffmpeg_backend import (
    FFmpegError,
    extract_pcm,
    probe_audio_streams,
    probe_duration,
    resolve_ffmpeg,
)
from syncaudio.models import AudioTrackSpec
from syncaudio.segments import DEFAULT_HOP_S, DEFAULT_MARGIN_S, DEFAULT_WINDOW_S, Segment, detect_segments

# Below this, a detected offset is treated as measurement noise and left
# uncorrected (skips an unneeded re-encode): real-content correlation
# residuals of a few tens of ms are expected even for a genuinely constant
# offset (see engine/tests/fixtures/MANIFEST.json ground-truth comparisons).
_NO_CORRECTION_THRESHOLD_S = 0.05
_ANALYSIS_SAMPLE_RATE = 16000


def _stream_index(spec: AudioTrackSpec) -> int:
    return spec.stream_index if spec.stream_index is not None else 0


@dataclass(frozen=True)
class TrackCorrection:
    track: AudioTrackSpec
    language: str | None
    offset_seconds: float
    confidence: float
    ambiguous: bool

    @property
    def needs_correction(self) -> bool:
        return abs(self.offset_seconds) > _NO_CORRECTION_THRESHOLD_S


@dataclass(frozen=True)
class SegmentedTrackCorrection:
    """Like ``TrackCorrection``, but for a track whose offset isn't constant.

    Produced by ``plan_segmented_correction`` (drift/jump-aware, via
    ``segments.detect_segments``) instead of ``plan_corrections``, and
    applied with ``segment_correction_filter`` instead of
    ``correction_filter``.
    """

    track: AudioTrackSpec
    language: str | None
    segments: list[Segment]


def correction_filter(offset_seconds: float) -> str | None:
    """ffmpeg audio filter that cancels a candidate's ``offset_seconds`` lag.

    Positive ``offset_seconds`` means the candidate lags the reference, so
    its content is trimmed earlier by that much; negative means it leads, so
    silence is prepended to delay it. Either way the filter ends in ``apad``
    (pad indefinitely), and the caller caps the render with a global ``-t``
    set to the reference's duration -- together that covers both "pad the
    now-too-short audio" and "cut the now-too-long audio" without a separate
    code path for each. Returns ``None`` when the offset is within
    measurement noise and no correction is needed.
    """
    if abs(offset_seconds) <= _NO_CORRECTION_THRESHOLD_S:
        return None
    if offset_seconds > 0:
        return f"atrim=start={offset_seconds:.6f},asetpts=PTS-STARTPTS,apad"
    delay_ms = -offset_seconds * 1000.0
    return f"adelay={delay_ms:.3f}:all=1,apad"


def _atempo_chain(factor: float) -> str:
    """Chain of one or more ``atempo`` filters reaching ``factor`` overall.

    A single ``atempo`` instance only accepts [0.5, 2.0] (ffmpeg errors
    outside that); realistic drift factors are always well inside it, but
    this stays correct at any factor by chaining multiple instances.
    """
    if factor <= 0:
        raise ValueError(f"Facteur atempo invalide : {factor}")
    steps = []
    remaining = factor
    while remaining > 2.0:
        steps.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        steps.append(0.5)
        remaining /= 0.5
    steps.append(remaining)
    return ",".join(f"atempo={s:.6f}" for s in steps)


def segment_correction_filter(segments: Sequence[Segment], input_label: str, output_label: str) -> str:
    """Build a ffmpeg filter_complex fragment realigning one track, segment by segment.

    For each segment, extracts the candidate-time span that corresponds to
    that segment's reference-time span -- ``[start_s + offset_start, end_s +
    offset_end]`` -- and time-stretches it (``atempo``) to fit exactly into
    ``end_s - start_s``. When a segment is a constant shift (``offset_start
    == offset_end``), the extracted span is already the right length and the
    stretch factor is 1 (no-op) -- this is the drift case's formula
    degenerating into the same result as a plain shift, so one code path
    covers both instead of two. The segments are then concatenated in order,
    reproducing the reference's timeline exactly.

    ``apad=whole_dur`` guarantees each segment's pre-stretch extraction is
    exactly the expected length even where the real candidate audio runs out
    (segment too close to a track's start/end) -- silence fills the gap
    instead of desyncing everything concatenated after it.
    """
    if not segments:
        raise ValueError("segment_correction_filter needs at least one segment")

    chains = []
    seg_labels = []
    for i, seg in enumerate(segments):
        duration = seg.end_s - seg.start_s
        cand_start = seg.start_s + seg.offset_start
        cand_end = seg.end_s + seg.offset_end
        span = cand_end - cand_start
        factor = span / duration if duration > 1e-6 else 1.0

        parts = [f"atrim=start={max(0.0, cand_start):.6f}:end={max(0.0, cand_end):.6f}", "asetpts=PTS-STARTPTS"]
        if cand_start < 0:
            parts.append(f"adelay={-cand_start * 1000:.3f}:all=1")
        parts.append(f"apad=whole_dur={max(span, 0.0):.6f}")
        if abs(factor - 1.0) > 1e-4:
            parts.append(_atempo_chain(factor))

        seg_label = f"{output_label}_{i}"
        chains.append(f"[{input_label}]{','.join(parts)}[{seg_label}]")
        seg_labels.append(f"[{seg_label}]")

    concat = f"{''.join(seg_labels)}concat=n={len(segments)}:v=0:a=1[{output_label}]"
    return ";".join([*chains, concat])


def _analyze(spec: AudioTrackSpec, start: float, duration: float | None) -> tuple:
    pcm = extract_pcm(spec, sample_rate=_ANALYSIS_SAMPLE_RATE, start=start or None, duration=duration)
    return extract_envelope(pcm, _ANALYSIS_SAMPLE_RATE)


def plan_corrections(
    reference: AudioTrackSpec,
    candidates: Sequence[AudioTrackSpec],
    *,
    start: float = 0.0,
    duration: float | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> list[TrackCorrection]:
    """Detect the constant offset of each of ``candidates`` vs. ``reference``.

    ``reference`` and each of ``candidates`` may point at different files --
    this is what lets ``render`` mix tracks from a second, "donor" file in
    with a primary one.
    """
    lang_cache: dict[str, dict[int, str | None]] = {}

    def language_of(spec: AudioTrackSpec) -> str | None:
        if spec.path not in lang_cache:
            lang_cache[spec.path] = {s.index: s.language for s in probe_audio_streams(spec.path)}
        return lang_cache[spec.path].get(_stream_index(spec))

    log(f"[analyse] reference {reference.raw} ...")
    ref_env, frame_rate = _analyze(reference, start, duration)

    corrections = []
    for spec in candidates:
        log(f"[analyse] piste {spec.raw} ...")
        env, _ = _analyze(spec, start, duration)
        estimate = estimate_offset(ref_env, env, frame_rate)
        corrections.append(
            TrackCorrection(
                track=spec,
                language=language_of(spec),
                offset_seconds=estimate.offset_seconds,
                confidence=estimate.confidence,
                ambiguous=estimate.ambiguous,
            )
        )
    return corrections


def plan_segmented_correction(
    reference: AudioTrackSpec,
    candidate: AudioTrackSpec,
    *,
    start: float = 0.0,
    duration: float | None = None,
    window_s: float = DEFAULT_WINDOW_S,
    hop_s: float = DEFAULT_HOP_S,
    margin_s: float = DEFAULT_MARGIN_S,
    log: Callable[[str], None] = lambda msg: None,
) -> SegmentedTrackCorrection:
    """Drift/jump-aware counterpart of ``plan_corrections``, for one candidate track."""
    segs = detect_segments(
        reference, candidate, start=start, duration=duration, window_s=window_s, hop_s=hop_s, margin_s=margin_s, log=log
    )
    language = None
    streams = {s.index: s for s in probe_audio_streams(candidate.path)}
    if _stream_index(candidate) in streams:
        language = streams[_stream_index(candidate)].language
    return SegmentedTrackCorrection(track=candidate, language=language, segments=segs)


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(f"Échec ffmpeg :\n{' '.join(cmd)}\n{proc.stderr.decode(errors='replace')}")


def render(
    input_path: str,
    reference_index: int,
    corrections: Sequence[TrackCorrection],
    output_path: str,
    *,
    audio_only: bool = False,
    imported_subs: Sequence[tuple[AudioTrackSpec, float]] = (),
    segmented_corrections: Sequence[SegmentedTrackCorrection] = (),
) -> list[str]:
    """Apply ``corrections`` and write the result.

    ``input_path`` supplies the video (and, unless overridden by
    ``corrections``, everything else): it is always input 0 and always
    provides the reference audio, subtitles and attachments untouched. Each
    entry of ``corrections`` may point at ``input_path`` itself (the original
    single-file use case) or at a different "donor" file, in which case that
    file is added as an extra ffmpeg input and its corrected track is mixed
    in. ``imported_subs`` is a list of (subtitle track, offset_seconds) pairs
    from donor files, shifted by ``-offset_seconds`` (a pure timestamp shift,
    unlike audio's trim/pad -- there's no audio content to cut or pad in a
    subtitle stream) and added as extra subtitle tracks. ``segmented_corrections``
    are tracks with a non-constant offset (drift and/or jumps, from
    ``plan_segmented_correction``), applied with ``segment_correction_filter``
    instead of the plain constant-offset ``correction_filter``.

    With ``audio_only``, writes one corrected ``.flac`` per entry of
    ``corrections``/``segmented_corrections`` next to ``output_path`` (named
    after its stem and source file) instead of a remuxed container, and
    returns their paths. Otherwise remuxes into a single MKV -- corrected
    tracks re-encoded to flac, everything else stream-copied -- capped to the
    reference's duration, and returns ``[output_path]``.
    """
    ffmpeg = resolve_ffmpeg()
    ref_duration = probe_duration(input_path)

    if audio_only:
        out_base = Path(output_path)
        written = []
        for corr in corrections:
            idx = _stream_index(corr.track)
            filt = correction_filter(corr.offset_seconds)
            donor = Path(corr.track.path).stem
            track_out = out_base.with_name(f"{out_base.stem}.{donor}.track{idx}.flac")
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", corr.track.path, "-map", f"0:a:{idx}"]
            if filt:
                cmd += ["-af", filt]
            cmd += ["-t", str(ref_duration), str(track_out)]
            _run(cmd)
            written.append(str(track_out))
        for seg_corr in segmented_corrections:
            idx = _stream_index(seg_corr.track)
            donor = Path(seg_corr.track.path).stem
            track_out = out_base.with_name(f"{out_base.stem}.{donor}.track{idx}.flac")
            filt = segment_correction_filter(seg_corr.segments, f"0:a:{idx}", "out")
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", seg_corr.track.path,
                "-filter_complex", filt,
                "-map", "[out]",
                "-t", str(ref_duration), str(track_out),
            ]
            _run(cmd)
            written.append(str(track_out))
        return written

    streams_ref = {s.index: s for s in probe_audio_streams(input_path)}

    inputs: list[list[str]] = [["-i", input_path]]
    input_index_for_path: dict[str, int] = {input_path: 0}

    def input_index_for(path: str) -> int:
        if path not in input_index_for_path:
            input_index_for_path[path] = len(inputs)
            inputs.append(["-i", path])
        return input_index_for_path[path]

    filter_complex_parts: list[str] = []
    audio_map_args = ["-map", f"0:a:{reference_index}"]
    audio_codec_args = ["-c:a:0", "copy"]
    metadata_args: list[str] = []
    if reference_index in streams_ref and streams_ref[reference_index].language:
        metadata_args += ["-metadata:s:a:0", f"language={streams_ref[reference_index].language}"]

    pos = 1
    for corr in corrections:
        spec = corr.track
        idx = _stream_index(spec)
        in_idx = input_index_for(spec.path)
        filt = correction_filter(corr.offset_seconds)
        if filt:
            label = f"a{pos}"
            filter_complex_parts.append(f"[{in_idx}:a:{idx}]{filt}[{label}]")
            audio_map_args += ["-map", f"[{label}]"]
            audio_codec_args += [f"-c:a:{pos}", "flac"]
        else:
            audio_map_args += ["-map", f"{in_idx}:a:{idx}"]
            audio_codec_args += [f"-c:a:{pos}", "copy"]
        if corr.language:
            metadata_args += [f"-metadata:s:a:{pos}", f"language={corr.language}"]
        pos += 1

    for seg_corr in segmented_corrections:
        spec = seg_corr.track
        idx = _stream_index(spec)
        in_idx = input_index_for(spec.path)
        label = f"a{pos}"
        filter_complex_parts.append(segment_correction_filter(seg_corr.segments, f"{in_idx}:a:{idx}", label))
        audio_map_args += ["-map", f"[{label}]"]
        audio_codec_args += [f"-c:a:{pos}", "flac"]
        if seg_corr.language:
            metadata_args += [f"-metadata:s:a:{pos}", f"language={seg_corr.language}"]
        pos += 1

    sub_map_args: list[str] = []
    for spec, offset in imported_subs:
        idx = _stream_index(spec)
        new_idx = len(inputs)
        inputs.append(["-itsoffset", f"{-offset:.6f}", "-i", spec.path])
        sub_map_args += ["-map", f"{new_idx}:s:{idx}"]

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for inp in inputs:
        cmd += inp
    if filter_complex_parts:
        cmd += ["-filter_complex", ";".join(filter_complex_parts)]
    cmd += ["-map", "0:v:0", *audio_map_args, "-map", "0:s?", "-map", "0:t?", *sub_map_args]
    cmd += ["-c:v", "copy", *audio_codec_args, "-c:s", "copy", *metadata_args]
    cmd += ["-t", str(ref_duration), output_path]
    _run(cmd)
    return [output_path]
