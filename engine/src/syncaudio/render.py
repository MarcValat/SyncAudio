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

# Below this, a detected offset is treated as measurement noise and left
# uncorrected (skips an unneeded re-encode): real-content correlation
# residuals of a few tens of ms are expected even for a genuinely constant
# offset (see engine/tests/fixtures/MANIFEST.json ground-truth comparisons).
_NO_CORRECTION_THRESHOLD_S = 0.05
_ANALYSIS_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class TrackCorrection:
    index: int
    language: str | None
    offset_seconds: float
    confidence: float
    ambiguous: bool

    @property
    def needs_correction(self) -> bool:
        return abs(self.offset_seconds) > _NO_CORRECTION_THRESHOLD_S


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


def _analyze(input_path: str, index: int, start: float, duration: float | None) -> tuple:
    spec = AudioTrackSpec(raw=f"{input_path}@{index}", path=input_path, stream_index=index)
    pcm = extract_pcm(spec, sample_rate=_ANALYSIS_SAMPLE_RATE, start=start or None, duration=duration)
    return extract_envelope(pcm, _ANALYSIS_SAMPLE_RATE)


def plan_corrections(
    input_path: str,
    reference_index: int,
    track_indices: Sequence[int],
    *,
    start: float = 0.0,
    duration: float | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> list[TrackCorrection]:
    """Detect the constant offset of each of ``track_indices`` vs. the reference."""
    streams = {s.index: s for s in probe_audio_streams(input_path)}

    log(f"[analyse] reference @{reference_index} ...")
    ref_env, frame_rate = _analyze(input_path, reference_index, start, duration)

    corrections = []
    for idx in track_indices:
        log(f"[analyse] piste @{idx} ...")
        env, _ = _analyze(input_path, idx, start, duration)
        estimate = estimate_offset(ref_env, env, frame_rate)
        corrections.append(
            TrackCorrection(
                index=idx,
                language=streams[idx].language if idx in streams else None,
                offset_seconds=estimate.offset_seconds,
                confidence=estimate.confidence,
                ambiguous=estimate.ambiguous,
            )
        )
    return corrections


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
) -> list[str]:
    """Apply ``corrections`` and write the result.

    With ``audio_only``, writes one corrected ``.flac`` per entry of
    ``corrections`` next to ``output_path`` (named after its stem) instead of
    a remuxed container, and returns their paths. Otherwise remuxes into a
    single MKV -- video, subtitles and attachments copied untouched, the
    reference audio copied untouched, corrected tracks re-encoded to flac --
    capped to the reference's duration, and returns ``[output_path]``.
    """
    ffmpeg = resolve_ffmpeg()
    streams = {s.index: s for s in probe_audio_streams(input_path)}
    ref_duration = probe_duration(input_path)

    if audio_only:
        out_base = Path(output_path)
        written = []
        for corr in corrections:
            filt = correction_filter(corr.offset_seconds)
            track_out = out_base.with_name(f"{out_base.stem}.track{corr.index}.flac")
            cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", input_path]
            cmd += ["-map", f"0:a:{corr.index}"]
            if filt:
                cmd += ["-af", filt]
            cmd += ["-t", str(ref_duration), str(track_out)]
            _run(cmd)
            written.append(str(track_out))
        return written

    corrected_by_index = {c.index: c for c in corrections}
    audio_track_order = [reference_index] + sorted(corrected_by_index)

    filter_complex_parts = []
    map_args: list[str] = []
    codec_args: list[str] = []
    metadata_args: list[str] = []
    for pos, idx in enumerate(audio_track_order):
        corr = corrected_by_index.get(idx)
        filt = correction_filter(corr.offset_seconds) if corr else None
        if filt:
            label = f"a{idx}"
            filter_complex_parts.append(f"[0:a:{idx}]{filt}[{label}]")
            map_args += ["-map", f"[{label}]"]
            codec_args += [f"-c:a:{pos}", "flac"]
        else:
            map_args += ["-map", f"0:a:{idx}"]
            codec_args += [f"-c:a:{pos}", "copy"]
        lang = streams[idx].language if idx in streams else None
        if lang:
            metadata_args += [f"-metadata:s:a:{pos}", f"language={lang}"]

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", input_path]
    if filter_complex_parts:
        cmd += ["-filter_complex", ";".join(filter_complex_parts)]
    cmd += ["-map", "0:v:0", *map_args, "-map", "0:s?", "-map", "0:t?"]
    cmd += ["-c:v", "copy", *codec_args, "-c:s", "copy", *metadata_args]
    cmd += ["-t", str(ref_duration), output_path]
    _run(cmd)
    return [output_path]
