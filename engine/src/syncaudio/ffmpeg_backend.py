from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from syncaudio.models import AudioStreamInfo, AudioTrackSpec

_STREAM_RE = re.compile(
    r"^\s*Stream #\d+:(?P<index>\d+)(?:\((?P<lang>[^)]+)\))?:\s*Audio:\s*"
    r"(?P<codec>[^,]+),\s*(?P<rate>\d+)\s*Hz,\s*(?P<channels>[^,]+)"
)
_DURATION_RE = re.compile(r"Duration:\s*(?P<h>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)")
_DURATION_START_RE = re.compile(r"Duration:\s*\d+:\d+:\d+(?:\.\d+)?,\s*start:\s*(?P<start>-?\d+(?:\.\d+)?)")
_SUBTITLE_STREAM_RE = re.compile(r"^\s*Stream #\d+:(?P<index>\d+)(?:\([^)]+\))?:\s*Subtitle:\s*(?P<codec>\S+)")


class FFmpegError(RuntimeError):
    """Raised when the ffmpeg binary is missing or a media operation fails."""


@lru_cache(maxsize=1)
def resolve_ffmpeg() -> str:
    """Locate an ffmpeg executable: prefer one on PATH, else the bundled one."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - defensive
        raise FFmpegError(
            "ffmpeg introuvable : ni sur le PATH, ni via le paquet imageio-ffmpeg."
        ) from exc


def parse_track_spec(raw: str) -> AudioTrackSpec:
    """Parse a CLI track argument: ``path`` or ``path@INDEX``.

    ``@`` is used as separator (rather than ``:``) so Windows drive letters
    like ``C:\\...`` are never ambiguous.
    """
    if "@" in raw:
        path, _, index_str = raw.rpartition("@")
        try:
            index = int(index_str)
        except ValueError as exc:
            raise ValueError(
                f"Index de piste invalide dans {raw!r} : {index_str!r} n'est pas un entier."
            ) from exc
        return AudioTrackSpec(raw=raw, path=path, stream_index=index)
    return AudioTrackSpec(raw=raw, path=raw, stream_index=None)


def probe_audio_streams(path: str) -> list[AudioStreamInfo]:
    """List the audio streams of a media file.

    The returned ``index`` is 0-based among *audio* streams only (matching
    ffmpeg's ``-map 0:a:N`` selector), not the container's global stream index.
    """
    ffmpeg = resolve_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
    )
    stderr = proc.stderr
    if "Invalid data found" in stderr or "No such file or directory" in stderr:
        raise FFmpegError(f"Impossible de lire {path!r} :\n{stderr}")

    streams: list[AudioStreamInfo] = []
    for line in stderr.splitlines():
        match = _STREAM_RE.match(line)
        if not match:
            continue
        channels_raw = match.group("channels").strip()
        channels = _CHANNEL_LAYOUTS.get(channels_raw)
        streams.append(
            AudioStreamInfo(
                index=len(streams),
                codec=match.group("codec").strip(),
                language=match.group("lang"),
                channels=channels,
                sample_rate=int(match.group("rate")),
            )
        )
    if not streams:
        raise FFmpegError(f"Aucune piste audio trouvée dans {path!r}.")
    return streams


def probe_duration(path: str) -> float:
    """Return the container's total duration in seconds, as reported by ffmpeg."""
    ffmpeg = resolve_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
    )
    match = _DURATION_RE.search(proc.stderr)
    if not match:
        raise FFmpegError(f"Impossible de déterminer la durée de {path!r}.")
    return int(match["h"]) * 3600 + int(match["m"]) * 60 + float(match["s"])


@lru_cache(maxsize=256)
def probe_stream_start_time(path: str, stream_index: int) -> float:
    """Container-level presentation delay of one audio stream (0.0 if none).

    A track remuxed with a per-track delay (e.g. mkvtoolnix's ``--sync``, or
    any tool that shifts a track's block timestamps instead of re-encoding
    it) only starts *presenting* at this offset. ffmpeg handles it
    inconsistently when decoding to raw audio (verified empirically): with
    no ``-ss``, or ``-ss`` below the delay, output starts at the track's
    first sample (delay dropped); with ``-ss T`` at or past the delay, it
    lands on own-time ``T - delay`` (delay honored). See ``_seek_args``,
    which uses this value to keep every windowed extraction in the track's
    own timeline.

    Method: ffmpeg's default output muxing normalizes away a stream's start
    time (``-avoid_negative_ts make_zero``); isolating the stream into its
    own container with ``-copyts`` (which disables that) and re-probing it
    reveals ffmpeg's own internal understanding of the delay. Best-effort:
    returns 0.0 on any failure rather than raising, since this must never
    break the actual detection/render pipeline it's decoupled from.
    """
    ffmpeg = resolve_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="syncaudio-starttime-") as tmp_dir:
        tmp_path = str(Path(tmp_dir) / "probe.mka")
        extract = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", path, "-map", f"0:a:{stream_index}", "-c", "copy", "-copyts", tmp_path,
            ],
            capture_output=True,
        )
        if extract.returncode != 0:
            return 0.0
        probe = subprocess.run([ffmpeg, "-hide_banner", "-i", tmp_path], capture_output=True, text=True)
        match = _DURATION_START_RE.search(probe.stderr)
        return float(match["start"]) if match else 0.0


def _seek_args(spec: AudioTrackSpec, start: float) -> list[str]:
    """``-ss`` args landing on ``start`` in the track's own timeline.

    Whole-track extraction (detection, waveforms) always sees the track from
    its first sample, container delay dropped. A plain ``-ss start`` only
    agrees with that when ``start`` is below the delay; past it, ffmpeg
    honors the delay and every clip ends up shifted by it. Seeking to
    ``start + delay`` is always in the honored regime and lands exactly on
    own-time ``start``, for any ``start``.
    """
    stream_index = spec.stream_index if spec.stream_index is not None else 0
    # -ss is relative to the file's own start, i.e. the earliest stream.
    delay = max(0.0, probe_stream_start_time(spec.path, stream_index) - _probe_format_start_time(spec.path))
    return ["-ss", f"{start + delay:.6f}"]


@lru_cache(maxsize=256)
def _probe_format_start_time(path: str) -> float:
    probe = subprocess.run([resolve_ffmpeg(), "-hide_banner", "-i", path], capture_output=True, text=True)
    match = _DURATION_START_RE.search(probe.stderr)
    return float(match["start"]) if match else 0.0


def _list_subtitle_streams(path: str) -> list[re.Match[str]]:
    ffmpeg = resolve_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
    )
    return [m for line in proc.stderr.splitlines() if (m := _SUBTITLE_STREAM_RE.match(line))]


def probe_subtitle_codec(path: str, index: int) -> str:
    """Return the codec name (e.g. ``subrip``, ``ass``) of subtitle stream ``index`` in ``path``."""
    subtitle_streams = _list_subtitle_streams(path)
    if index >= len(subtitle_streams):
        raise FFmpegError(f"Piste de sous-titres @{index} absente de {path!r} ({len(subtitle_streams)} trouvée(s)).")
    return subtitle_streams[index]["codec"]


def probe_subtitle_count(path: str) -> int:
    """Return how many subtitle streams ``path`` has."""
    return len(_list_subtitle_streams(path))


_CHANNEL_LAYOUTS = {
    "mono": 1,
    "stereo": 2,
    "2.1": 3,
    "5.1": 6,
    "5.1(side)": 6,
    "7.1": 8,
}


def extract_pcm(
    spec: AudioTrackSpec,
    sample_rate: int = 16000,
    start: float | None = None,
    duration: float | None = None,
) -> np.ndarray:
    """Decode a track to mono float32 PCM in [-1, 1] at ``sample_rate`` Hz.

    ``start``/``duration`` (seconds) restrict decoding to a window of the
    track: ``-ss`` is placed *before* ``-i`` so ffmpeg seeks directly to that
    point instead of decoding everything up to it, which is what makes
    windowed extraction actually fast on long files.
    """
    ffmpeg = resolve_ffmpeg()
    stream_index = spec.stream_index if spec.stream_index is not None else 0
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += _seek_args(spec, start)
    cmd += ["-i", spec.path]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += [
        "-map",
        f"0:a:{stream_index}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(
            f"Échec de l'extraction audio pour {spec.raw!r} :\n"
            f"{proc.stderr.decode(errors='replace')}"
        )
    pcm = np.frombuffer(proc.stdout, dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


_PEAKS_SAMPLE_RATE = 22050


def extract_peaks(
    spec: AudioTrackSpec, buckets: int, start: float = 0.0, duration: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-bucket (min, max) amplitude envelope of a track window, for drawing a waveform.

    Returns ``(mins, maxes, actual_duration)`` -- ``actual_duration`` is the
    real decoded length, which can be shorter than requested ``duration``
    near a track's end. Downsampling to ``buckets`` happens here, not in the
    browser: decoding is cheap (this is plain ffmpeg PCM extraction, not the
    STFT/HPSS analysis path -- confirmed even a whole 6-minute track decodes
    in well under a second), but shipping raw samples over HTTP for a
    multi-minute track would not be. A whole-track call (``duration=None``)
    is how the GUI learns a track's total duration in the first place.
    """
    pcm = extract_pcm(spec, sample_rate=_PEAKS_SAMPLE_RATE, start=start or None, duration=duration)
    actual_duration = len(pcm) / _PEAKS_SAMPLE_RATE
    if len(pcm) == 0 or buckets <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), actual_duration

    bucket_size = max(1, len(pcm) // buckets)
    usable = pcm[: bucket_size * buckets] if len(pcm) >= bucket_size * buckets else pcm
    n = len(usable) // bucket_size
    if n == 0:
        return np.array([pcm.min()], dtype=np.float32), np.array([pcm.max()], dtype=np.float32), actual_duration
    chunks = usable[: n * bucket_size].reshape(n, bucket_size)
    return chunks.min(axis=1), chunks.max(axis=1), actual_duration


def extract_wav_clip(spec: AudioTrackSpec, start: float, duration: float, sample_rate: int = 44100) -> bytes:
    """Encode a short window of a track as playable WAV bytes.

    Unlike ``extract_pcm`` (mono, 16kHz, raw samples only ever consumed by
    numpy for analysis), this keeps the track's original channel layout at a
    normal playback rate and wraps it in a proper WAV header -- for the
    GUI's listen-before-render preview, not analysis. Only ever called with
    a short ``duration`` (a preview clip, not a whole track), so this stays
    fast even without the analysis cache.
    """
    ffmpeg = resolve_ffmpeg()
    stream_index = spec.stream_index if spec.stream_index is not None else 0
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        *_seek_args(spec, start), "-i", spec.path, "-t", str(duration),
        "-map", f"0:a:{stream_index}",
        "-ar", str(sample_rate),
        "-f", "wav", "-acodec", "pcm_s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise FFmpegError(
            f"Échec de l'extraction du clip pour {spec.raw!r} :\n"
            f"{proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout
