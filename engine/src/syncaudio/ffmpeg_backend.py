from __future__ import annotations

import re
import shutil
import subprocess
from functools import lru_cache

import numpy as np

from syncaudio.models import AudioStreamInfo, AudioTrackSpec

_STREAM_RE = re.compile(
    r"^\s*Stream #\d+:(?P<index>\d+)(?:\((?P<lang>[^)]+)\))?:\s*Audio:\s*"
    r"(?P<codec>[^,]+),\s*(?P<rate>\d+)\s*Hz,\s*(?P<channels>[^,]+)"
)


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
        cmd += ["-ss", str(start)]
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
