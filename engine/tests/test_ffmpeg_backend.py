from __future__ import annotations

import io
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
    probe_stream_start_time,
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


def test_probe_stream_start_time_reports_container_level_delay(tmp_path: Path) -> None:
    """Simulates a track remuxed with a per-track delay (e.g. mkvtoolnix's
    --sync), via ffmpeg's -itsoffset -- both shift a track's presentation
    timestamps without touching its audio essence, so this is a faithful
    stand-in even without mkvtoolnix installed."""
    ffmpeg = resolve_ffmpeg()
    sr = 44100
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    _write_wav(wav_a, sr=sr, freq=440.0, duration_s=2.0)
    _write_wav(wav_b, sr=sr, freq=880.0, duration_s=2.0)

    mkv = tmp_path / "delayed.mkv"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=3",
            "-i", str(wav_a),
            "-itsoffset", "1.0", "-i", str(wav_b),
            "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-shortest", str(mkv),
        ],
        check=True, capture_output=True,
    )

    assert abs(probe_stream_start_time(str(mkv), 0) - 0.0) < 0.05
    assert abs(probe_stream_start_time(str(mkv), 1) - 1.0) < 0.05


def test_windowed_extraction_timeline_is_absolute_despite_container_delay(tmp_path: Path) -> None:
    """Without compensation, ffmpeg drops a container delay when seeking
    below it but honors it past it -- so a clip's content depended on where
    it started. Every start must land on the same own-time position."""
    ffmpeg = resolve_ffmpeg()
    sr = 16000
    n = sr * 6
    samples = np.zeros(n)
    samples[sr * 4 : sr * 4 + sr // 5] = np.random.default_rng(0).uniform(-0.8, 0.8, sr // 5)
    wav = tmp_path / "burst.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((samples * 32767).astype("<i2").tobytes())
    def burst_at(pcm: np.ndarray) -> float:
        energy = np.convolve(np.abs(pcm), np.ones(sr // 20) / (sr // 20), mode="same")
        return int(np.argmax(energy)) / sr

    # With an undelayed video stream (the real-world case: the file starts
    # at 0, only this track at 1.0) and alone (the file itself starts at 1.0).
    layouts = {
        "with_video": ["-f", "lavfi", "-i", "color=c=black:s=64x64:d=7", "-itsoffset", "1.0", "-i", str(wav), "-map", "0:v", "-map", "1:a"],
        "audio_only": ["-itsoffset", "1.0", "-i", str(wav)],
    }
    for name, args in layouts.items():
        mkv = tmp_path / f"{name}.mkv"
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args, str(mkv)], check=True, capture_output=True)
        spec = AudioTrackSpec(raw=str(mkv), path=str(mkv), stream_index=0)
        for start in (0.5, 1.0, 2.0, 3.0):
            pcm = extract_pcm(spec, sample_rate=sr, start=start, duration=4.0)
            assert abs(start + burst_at(pcm) - 4.1) < 0.05, (name, start)
            with wave.open(io.BytesIO(extract_wav_clip(spec, start, 4.0, sample_rate=sr))) as w:
                clip = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
            assert abs(start + burst_at(clip) - 4.1) < 0.05, (name, start)


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
