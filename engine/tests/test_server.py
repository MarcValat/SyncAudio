from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from syncaudio.ffmpeg_backend import resolve_ffmpeg
from syncaudio.server import app

client = TestClient(app)


def _make_bed(duration_s: float, sr: int, seed: int, hits_per_second: float = 3.0) -> np.ndarray:
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


@pytest.fixture()
def offset_mkv(tmp_path: Path) -> tuple[Path, float]:
    """A 2-track mkv (dummy video + reference + candidate lagging by 3s)."""
    sr = 44100
    duration_s = 30.0
    offset_s = 3.0

    bed = _make_bed(duration_s, sr, seed=42)
    shifted = np.concatenate([np.zeros(int(offset_s * sr)), bed])[: len(bed)]

    ref_wav = tmp_path / "ref.wav"
    cand_wav = tmp_path / "cand.wav"
    _write_wav(ref_wav, bed, sr)
    _write_wav(cand_wav, shifted, sr)

    mkv = tmp_path / "multi.mkv"
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
    return mkv, offset_s


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_probe_lists_tracks(offset_mkv: tuple[Path, float]) -> None:
    mkv, _ = offset_mkv
    resp = client.get("/probe", params={"path": str(mkv)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"] == str(mkv)
    assert [t["index"] for t in body["tracks"]] == [0, 1]
    assert [t["language"] for t in body["tracks"]] == ["jpn", "fre"]


def test_probe_missing_file_returns_400() -> None:
    resp = client.get("/probe", params={"path": "does-not-exist.mkv"})
    assert resp.status_code == 400


def test_align_endpoint_recovers_offset(offset_mkv: tuple[Path, float]) -> None:
    mkv, offset_s = offset_mkv
    resp = client.post(
        "/align",
        json={"reference": {"path": str(mkv), "index": 0}, "candidates": [{"path": str(mkv), "index": 1}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert abs(body["results"][0]["offset_seconds"] - offset_s) < 0.1


def test_segments_endpoint_returns_a_segment(offset_mkv: tuple[Path, float]) -> None:
    mkv, offset_s = offset_mkv
    resp = client.post(
        "/segments",
        json={
            "reference": {"path": str(mkv), "index": 0},
            "track": {"path": str(mkv), "index": 1},
            "window_s": 10.0,
            "hop_s": 5.0,
            "margin_s": 5.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["segments"]) >= 1
    assert abs(body["segments"][0]["offset_start"] - offset_s) < 0.5


def test_render_endpoint_writes_a_corrected_file(offset_mkv: tuple[Path, float]) -> None:
    mkv, offset_s = offset_mkv
    output_path = str(mkv.with_name("out.synced.mkv"))
    resp = client.post(
        "/render",
        json={
            "input_path": str(mkv),
            "reference_index": 0,
            "track_indices": [1],
            "output_path": output_path,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["written"] == [output_path]
    assert Path(output_path).exists()
    assert len(body["corrections"]) == 1
    assert abs(body["corrections"][0]["offset_seconds"] - offset_s) < 0.1


def test_render_endpoint_segmented(offset_mkv: tuple[Path, float]) -> None:
    mkv, offset_s = offset_mkv
    output_path = str(mkv.with_name("out.segmented.mkv"))
    resp = client.post(
        "/render",
        json={
            "input_path": str(mkv),
            "reference_index": 0,
            "track_indices": [1],
            "output_path": output_path,
            "segmented": True,
            "window_s": 10.0,
            "hop_s": 5.0,
            "margin_s": 5.0,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["written"] == [output_path]
    assert body["corrections"][0]["segments"] is not None
    assert body["corrections"][0]["offset_seconds"] is None


def test_render_endpoint_unknown_track_returns_400(offset_mkv: tuple[Path, float]) -> None:
    mkv, _ = offset_mkv
    resp = client.post(
        "/render",
        json={"input_path": str(mkv), "reference_index": 0, "track_indices": [7]},
    )
    assert resp.status_code == 400


def _drain_job_ws(job_id: str) -> list[dict]:
    events = []
    with client.websocket_connect(f"/jobs/{job_id}/ws") as ws:
        while True:
            event = ws.receive_json()
            events.append(event)
            if event["type"] in ("done", "error"):
                break
    return events


def test_job_align_streams_progress_then_result(offset_mkv: tuple[Path, float]) -> None:
    mkv, offset_s = offset_mkv
    resp = client.post(
        "/jobs/align",
        json={"reference": {"path": str(mkv), "index": 0}, "candidates": [{"path": str(mkv), "index": 1}]},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    events = _drain_job_ws(job_id)
    assert any(e["type"] == "log" for e in events)  # got at least one progress message
    assert events[-1]["type"] == "done"
    result = events[-1]["result"]
    assert abs(result["results"][0]["offset_seconds"] - offset_s) < 0.1

    # Also available via polling, after the fact.
    status = client.get(f"/jobs/{job_id}").json()
    assert status["status"] == "done"
    assert status["messages"]  # the same log lines are kept for late/polling clients


def test_job_render_writes_file_and_streams_progress(offset_mkv: tuple[Path, float]) -> None:
    mkv, offset_s = offset_mkv
    output_path = str(mkv.with_name("out.job.mkv"))
    resp = client.post(
        "/jobs/render",
        json={
            "input_path": str(mkv),
            "reference_index": 0,
            "track_indices": [1],
            "output_path": output_path,
        },
    )
    job_id = resp.json()["job_id"]

    events = _drain_job_ws(job_id)
    assert any(e["type"] == "log" for e in events)
    assert events[-1]["type"] == "done"
    result = events[-1]["result"]
    assert result["written"] == [output_path]
    assert Path(output_path).exists()
    assert abs(result["corrections"][0]["offset_seconds"] - offset_s) < 0.1


def test_job_ws_unknown_job_id_reports_error() -> None:
    with client.websocket_connect("/jobs/does-not-exist/ws") as ws:
        event = ws.receive_json()
    assert event["type"] == "error"


def test_job_error_is_reported_not_left_hanging() -> None:
    resp = client.post(
        "/jobs/render",
        json={"input_path": "does-not-exist.mkv", "reference_index": 0},
    )
    job_id = resp.json()["job_id"]
    events = _drain_job_ws(job_id)
    assert events[-1]["type"] == "error"
