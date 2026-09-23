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
    subtitle stream) and added as extra subtitle tracks.

    With ``audio_only``, writes one corrected ``.flac`` per entry of
    ``corrections`` next to ``output_path`` (named after its stem and source
    file) instead of a remuxed container, and returns their paths. Otherwise
    remuxes into a single MKV -- corrected tracks re-encoded to flac,
    everything else stream-copied -- capped to the reference's duration, and
    returns ``[output_path]``.
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

    for pos, corr in enumerate(corrections, start=1):
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
