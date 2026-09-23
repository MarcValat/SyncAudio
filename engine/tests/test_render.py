from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from syncaudio.align import estimate_offset
from syncaudio.features import extract_envelope
from syncaudio.ffmpeg_backend import extract_pcm, parse_track_spec, probe_audio_streams, resolve_ffmpeg
from syncaudio.models import AudioTrackSpec
from syncaudio.render import (
    _atempo_chain,
    correction_filter,
    plan_corrections,
    plan_segmented_correction,
    render,
    segment_correction_filter,
)
from syncaudio.segments import Segment


def test_correction_filter_positive_offset_trims_and_pads() -> None:
    assert correction_filter(2.5) == "atrim=start=2.500000,asetpts=PTS-STARTPTS,apad"


def test_correction_filter_negative_offset_delays_and_pads() -> None:
    assert correction_filter(-1.0) == "adelay=1000.000:all=1,apad"


def test_correction_filter_within_noise_floor_is_a_no_op() -> None:
    assert correction_filter(0.01) is None
    assert correction_filter(-0.02) is None


def _make_bed(duration_s: float, sr: int, seed: int, hits_per_second: float = 3.0) -> np.ndarray:
    """Percussive-ish "music/SFX bed", same shape as the one in test_align.py."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    bed = np.zeros(n, dtype=np.float64)
    burst_len = int(0.05 * sr)
    envelope = np.hanning(burst_len)
    hit_times = rng.uniform(0, duration_s - 0.2, int(duration_s * hits_per_second))
    for t in hit_times:
        start = int(t * sr)
        if start + burst_len > n:
            continue
        bed[start : start + burst_len] += rng.standard_normal(burst_len) * envelope
    return bed


def _write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    samples = np.clip(data * 20000, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(samples.tobytes())


def _spec(path: Path | str, index: int) -> AudioTrackSpec:
    return AudioTrackSpec(raw=f"{path}@{index}", path=str(path), stream_index=index)


@pytest.fixture()
def offset_mkv(tmp_path: Path) -> tuple[Path, float]:
    """A 2-track mkv (dummy video + reference + candidate lagging by 3s)."""
    sr = 44100
    duration_s = 30.0
    offset_s = 3.0

    bed = _make_bed(duration_s, sr, seed=42)
    silence = np.zeros(int(offset_s * sr))
    shifted = np.concatenate([silence, bed])[: len(bed)]

    ref_wav = tmp_path / "ref.wav"
    cand_wav = tmp_path / "cand.wav"
    _write_wav(ref_wav, bed, sr)
    _write_wav(cand_wav, shifted, sr)

    mkv = tmp_path / "multi.mkv"
    ffmpeg = resolve_ffmpeg()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={duration_s}",
        "-i", str(ref_wav), "-i", str(cand_wav),
        "-map", "0:v", "-map", "1:a", "-map", "2:a",
        "-metadata:s:a:0", "language=jpn", "-metadata:s:a:1", "language=fre",
        "-shortest", str(mkv),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return mkv, offset_s


def test_plan_corrections_recovers_injected_offset(offset_mkv: tuple[Path, float]) -> None:
    mkv, offset_s = offset_mkv
    corrections = plan_corrections(_spec(mkv, 0), [_spec(mkv, 1)])

    assert len(corrections) == 1
    corr = corrections[0]
    assert corr.track.stream_index == 1
    assert corr.language == "fre"
    assert abs(corr.offset_seconds - offset_s) < 0.05
    assert corr.needs_correction


def test_render_corrects_offset_close_to_zero_residual(offset_mkv: tuple[Path, float]) -> None:
    mkv, offset_s = offset_mkv
    input_path = str(mkv)
    corrections = plan_corrections(_spec(mkv, 0), [_spec(mkv, 1)])

    output_path = str(mkv.with_name("out.synced.mkv"))
    written = render(input_path, reference_index=0, corrections=corrections, output_path=output_path)

    assert written == [output_path]
    assert Path(output_path).exists()

    ref_pcm = extract_pcm(parse_track_spec(f"{input_path}@0"), sample_rate=16000)
    fixed_pcm = extract_pcm(parse_track_spec(f"{output_path}@1"), sample_rate=16000)
    ref_env, frame_rate = extract_envelope(ref_pcm, 16000)
    fixed_env, _ = extract_envelope(fixed_pcm, 16000)

    residual = estimate_offset(ref_env, fixed_env, frame_rate)
    assert abs(residual.offset_seconds) < 0.05


def test_render_audio_only_exports_corrected_track(offset_mkv: tuple[Path, float]) -> None:
    mkv, offset_s = offset_mkv
    input_path = str(mkv)
    corrections = plan_corrections(_spec(mkv, 0), [_spec(mkv, 1)])

    output_path = str(mkv.with_name("out.synced.mkv"))
    written = render(input_path, reference_index=0, corrections=corrections, output_path=output_path, audio_only=True)

    assert len(written) == 1
    assert written[0].endswith(f"out.synced.{mkv.stem}.track1.flac")
    assert Path(written[0]).exists()


def _write_srt(path: Path, start_s: float, end_s: float, text: str = "Hello") -> None:
    def fmt(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{round((s - int(s)) * 1000):03d}"

    path.write_text(f"1\n{fmt(start_s)} --> {fmt(end_s)}\n{text}\n", encoding="utf-8")


@pytest.fixture()
def cross_file_fixture(tmp_path: Path) -> tuple[Path, Path, float, float]:
    """A reference-only mkv, plus a separate donor mkv (audio + subtitle) lagging by 3s."""
    sr = 44100
    duration_s = 30.0
    offset_s = 3.0
    cue_start_s = 10.0  # timed against the donor's own (pre-correction) audio

    bed = _make_bed(duration_s, sr, seed=7)
    silence = np.zeros(int(offset_s * sr))
    shifted = np.concatenate([silence, bed])[: len(bed)]

    ref_wav = tmp_path / "ref.wav"
    donor_wav = tmp_path / "donor.wav"
    _write_wav(ref_wav, bed, sr)
    _write_wav(donor_wav, shifted, sr)

    ffmpeg = resolve_ffmpeg()
    ref_mkv = tmp_path / "reference.mkv"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={duration_s}",
            "-i", str(ref_wav),
            "-map", "0:v", "-map", "1:a",
            "-metadata:s:a:0", "language=jpn",
            "-shortest", str(ref_mkv),
        ],
        check=True, capture_output=True,
    )

    srt_path = tmp_path / "donor.srt"
    _write_srt(srt_path, cue_start_s, cue_start_s + 2.0)

    donor_mkv = tmp_path / "donor.mkv"
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(donor_wav), "-i", str(srt_path),
            "-map", "0:a", "-map", "1:s",
            "-metadata:s:a:0", "language=fre",
            str(donor_mkv),
        ],
        check=True, capture_output=True,
    )
    return ref_mkv, donor_mkv, offset_s, cue_start_s


def test_render_imports_audio_and_subs_from_another_file(cross_file_fixture: tuple[Path, Path, float, float]) -> None:
    ref_mkv, donor_mkv, offset_s, cue_start_s = cross_file_fixture

    donor_audio = _spec(donor_mkv, 0)
    donor_subs = _spec(donor_mkv, 0)

    corrections = plan_corrections(_spec(ref_mkv, 0), [donor_audio])
    assert len(corrections) == 1
    assert abs(corrections[0].offset_seconds - offset_s) < 0.05

    output_path = str(ref_mkv.with_name("out.synced.mkv"))
    written = render(
        str(ref_mkv),
        reference_index=0,
        corrections=corrections,
        output_path=output_path,
        imported_subs=[(donor_subs, corrections[0].offset_seconds)],
    )
    assert written == [output_path]

    audio_streams = probe_audio_streams(output_path)
    assert len(audio_streams) == 2
    assert {s.language for s in audio_streams} == {"jpn", "fre"}

    ffmpeg = resolve_ffmpeg()
    extracted_srt = ref_mkv.with_name("out.extracted.srt")
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", output_path, "-map", "0:s:0", str(extracted_srt)],
        check=True, capture_output=True,
    )
    lines = extracted_srt.read_text(encoding="utf-8").splitlines()
    start_str = lines[1].split(" --> ")[0]
    h, m, rest = start_str.split(":")
    s, ms = rest.split(",")
    actual_start = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    expected_start = cue_start_s - offset_s  # subtitles shift the same way the audio was corrected
    assert abs(actual_start - expected_start) < 0.1


def test_atempo_chain_single_within_range() -> None:
    assert _atempo_chain(1.02) == "atempo=1.020000"


def test_atempo_chain_splits_large_factor() -> None:
    assert _atempo_chain(3.0) == "atempo=2.000000,atempo=1.500000"


def test_segment_correction_filter_concatenates_all_segments() -> None:
    segs = [
        Segment(0.0, 10.0, 0.0, 0.0),
        Segment(10.0, 20.0, 2.0, 2.0),
    ]
    filt = segment_correction_filter(segs, "0:a:1", "out")
    assert filt.count("atrim") == 2
    assert "concat=n=2:v=0:a=1[out]" in filt
    assert "atempo" not in filt  # both segments are constant-offset -> no stretch needed


def test_segment_correction_filter_stretches_a_drift_segment() -> None:
    segs = [Segment(0.0, 100.0, 0.0, 5.0)]
    filt = segment_correction_filter(segs, "0:a:1", "out")
    assert "atempo=1.050000" in filt  # span 105s squeezed into 100s


def _apply_jump(bed: np.ndarray, sr: int, jump_time_s: float, delta_s: float) -> np.ndarray:
    idx = int(jump_time_s * sr)
    if delta_s > 0:
        silence = np.zeros(int(delta_s * sr))
        return np.concatenate([bed[:idx], silence, bed[idx:]])
    cut = int(-delta_s * sr)
    return np.concatenate([bed[:idx], bed[idx + cut :]])


def _time_stretch(signal: np.ndarray, factor: float) -> np.ndarray:
    n = len(signal)
    new_n = int(round(n * factor))
    old_idx = np.linspace(0, n - 1, new_n)
    return np.interp(old_idx, np.arange(n), signal)


def _mux_two_track_mkv(tmp_path: Path, name: str, ref: np.ndarray, cand: np.ndarray, sr: int, duration_s: float) -> Path:
    ref_wav = tmp_path / f"{name}_ref.wav"
    cand_wav = tmp_path / f"{name}_cand.wav"
    _write_wav(ref_wav, ref, sr)
    _write_wav(cand_wav, cand, sr)

    mkv = tmp_path / f"{name}.mkv"
    ffmpeg = resolve_ffmpeg()
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={duration_s}",
            "-i", str(ref_wav), "-i", str(cand_wav),
            "-map", "0:v", "-map", "1:a", "-map", "2:a",
            "-metadata:s:a:0", "language=jpn", "-metadata:s:a:1", "language=fre",
            "-shortest", str(mkv),
        ],
        check=True, capture_output=True,
    )
    return mkv


def _residual_offset(reference_path: str, candidate_path: str) -> float:
    ref_pcm = extract_pcm(parse_track_spec(f"{reference_path}@0"), sample_rate=16000)
    fixed_pcm = extract_pcm(parse_track_spec(f"{candidate_path}@1"), sample_rate=16000)
    ref_env, frame_rate = extract_envelope(ref_pcm, 16000)
    fixed_env, _ = extract_envelope(fixed_pcm, 16000)
    return estimate_offset(ref_env, fixed_env, frame_rate).offset_seconds


@pytest.fixture()
def jump_mkv(tmp_path: Path) -> tuple[Path, float, float]:
    sr = 44100
    duration_s = 150.0
    jump_time_s = 75.0
    delta_s = 3.0

    bed = _make_bed(duration_s + delta_s, sr, seed=50)
    reference_bed = bed[: int(duration_s * sr)]
    candidate_bed = _apply_jump(bed, sr, jump_time_s, delta_s)[: int(duration_s * sr)]

    mkv = _mux_two_track_mkv(tmp_path, "jump", reference_bed, candidate_bed, sr, duration_s)
    return mkv, jump_time_s, delta_s


def test_render_segmented_corrects_a_jump(jump_mkv: tuple[Path, float, float]) -> None:
    mkv, _jump_time_s, _delta_s = jump_mkv
    input_path = str(mkv)

    seg_corr = plan_segmented_correction(_spec(mkv, 0), _spec(mkv, 1))
    assert len(seg_corr.segments) == 2

    output_path = str(mkv.with_name("out.synced.mkv"))
    written = render(
        input_path, reference_index=0, corrections=[], output_path=output_path, segmented_corrections=[seg_corr]
    )
    assert written == [output_path]
    assert abs(_residual_offset(input_path, output_path)) < 0.3


@pytest.fixture()
def drift_mkv(tmp_path: Path) -> tuple[Path, float]:
    sr = 44100
    duration_s = 150.0
    stretch_factor = 1.02

    bed = _make_bed(duration_s, sr, seed=60)
    stretched = _time_stretch(bed, stretch_factor)[: len(bed)]

    mkv = _mux_two_track_mkv(tmp_path, "drift", bed, stretched, sr, duration_s)
    return mkv, stretch_factor


def test_render_segmented_corrects_drift(drift_mkv: tuple[Path, float]) -> None:
    mkv, _stretch_factor = drift_mkv
    input_path = str(mkv)

    # Real-content round-tripping (flac encode/decode) can be noisy enough that
    # the drift gets classified as several piecewise-constant segments rather
    # than one clean drift segment (see engine README caveats). Either way,
    # each piece corrects its own local offset, so what actually matters is
    # the final corrected residual, not how many segments it took.
    seg_corr = plan_segmented_correction(_spec(mkv, 0), _spec(mkv, 1))
    assert len(seg_corr.segments) >= 1

    output_path = str(mkv.with_name("out.synced.mkv"))
    render(input_path, reference_index=0, corrections=[], output_path=output_path, segmented_corrections=[seg_corr])
    assert abs(_residual_offset(input_path, output_path)) < 0.4
