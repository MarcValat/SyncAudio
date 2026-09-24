from __future__ import annotations

import struct
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from syncaudio.ffmpeg_backend import (
    extract_pcm,
    extract_peaks,
    extract_wav_clip,
    parse_track_spec,
    probe_audio_streams,
    resolve_ffmpeg,
)
from syncaudio.models import AudioTrackSpec


def test_parse_track_spec_plain_path() -> None:
    spec = parse_track_spec(r"C:\videos\movie.mkv")
    assert spec == AudioTrackSpec(raw=r"C:\videos\movie.mkv", path=r"C:\videos\movie.mkv", stream_index=None)


def test_parse_track_spec_with_stream_index_and_drive_letter() -> None:
    spec = parse_track_spec(r"C:\videos\movie.mkv@2")
    assert spec.path == r"C:\videos\movie.mkv"
    assert spec.stream_index == 2


def test_parse_track_spec_invalid_index() -> None:
    with pytest.raises(ValueError):
        parse_track_spec("movie.mkv@abc")


def _write_wav(path: Path, sr: int, freq: float, duration_s: float) -> None:
    n = int(sr * duration_s)
    t = np.arange(n) / sr
    samples = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(struct.pack(f"<{n}h", *samples))


@pytest.fixture(scope="module")
def wav_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("media") / "tone.wav"
    _write_wav(path, sr=44100, freq=440.0, duration_s=1.0)
    return path


@pytest.fixture(scope="module")
def multi_track_mkv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_dir = tmp_path_factory.mktemp("media")
    wav_a = tmp_dir / "a.wav"
    wav_b = tmp_dir / "b.wav"
    _write_wav(wav_a, sr=44100, freq=440.0, duration_s=1.0)
    _write_wav(wav_b, sr=44100, freq=880.0, duration_s=1.0)

    mkv = tmp_dir / "multi.mkv"
    ffmpeg = resolve_ffmpeg()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:d=1",
        "-i",
        str(wav_a),
        "-i",
        str(wav_b),
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:a",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:1",
        "language=fre",
        "-shortest",
        str(mkv),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return mkv


def test_resolve_ffmpeg_finds_a_binary() -> None:
    path = resolve_ffmpeg()
    assert path
    assert Path(path).exists() or path == "ffmpeg"


def test_extract_pcm_from_plain_wav(wav_file: Path) -> None:
    spec = parse_track_spec(str(wav_file))
    pcm = extract_pcm(spec, sample_rate=16000)
    assert pcm.dtype == np.float32
    assert abs(len(pcm) - 16000) < 100
    assert np.abs(pcm).max() <= 1.0


def test_extract_peaks_bucket_count_and_range(wav_file: Path) -> None:
    spec = parse_track_spec(str(wav_file))
    mins, maxes, duration = extract_peaks(spec, buckets=50)
    assert len(mins) == 50
    assert len(maxes) == 50
    assert abs(duration - 1.0) < 0.05  # wav_file is a 1s tone
    assert (mins <= 0).all() and (maxes >= 0).all()  # a sine wave crosses zero in every bucket
    assert (mins >= -1.0).all() and (maxes <= 1.0).all()


def test_extract_peaks_whole_track_when_duration_omitted(wav_file: Path) -> None:
    spec = parse_track_spec(str(wav_file))
    _, _, duration = extract_peaks(spec, buckets=10)
    assert abs(duration - 1.0) < 0.05


def test_extract_wav_clip_is_a_playable_wav(wav_file: Path) -> None:
    spec = parse_track_spec(str(wav_file))
    clip = extract_wav_clip(spec, start=0.1, duration=0.3, sample_rate=44100)
    assert clip[:4] == b"RIFF"
    assert clip[8:12] == b"WAVE"
    assert len(clip) > 0.3 * 44100 * 2 * 0.5  # roughly duration * rate * bytes/sample, some slack


def test_probe_and_extract_multi_track_mkv(multi_track_mkv: Path) -> None:
    streams = probe_audio_streams(str(multi_track_mkv))
    assert len(streams) == 2
    assert {s.language for s in streams} == {"eng", "fre"}
    assert [s.index for s in streams] == [0, 1]

    spec_first = parse_track_spec(f"{multi_track_mkv}@0")
    spec_second = parse_track_spec(f"{multi_track_mkv}@1")
    pcm_first = extract_pcm(spec_first, sample_rate=16000)
    pcm_second = extract_pcm(spec_second, sample_rate=16000)

    assert len(pcm_first) > 0
    assert len(pcm_second) > 0
    shortest = min(len(pcm_first), len(pcm_second))
    assert not np.allclose(pcm_first[:shortest], pcm_second[:shortest])
