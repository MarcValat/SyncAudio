"""FastAPI sidecar exposing the engine over HTTP.

Thin layer over the same functions the CLI uses (``render.py``,
``segments.py``, ``ffmpeg_backend.py``) -- the CLI remains a perfectly valid
client on its own; this is an additional one for the future GUI.

Two ways to call align/segments/render:
- Directly (``GET /probe``, ``POST /align``, ``POST /segments``, ``POST
  /render``): synchronous, blocks until done. Simple, fine for scripting or
  quick checks (Starlette runs ``def`` routes in a threadpool, so one
  in-flight request doesn't block others).
- As a job (``POST /jobs/align`` etc.): returns a ``job_id`` immediately,
  runs in a background thread, and ``WS /jobs/{job_id}/ws`` streams the same
  progress messages the CLI prints ("[analyse] ...") as they happen, ending
  with the result -- or poll ``GET /jobs/{job_id}`` instead of using the
  WebSocket. This is what a GUI should use for anything on a real file
  (tens of seconds), so it can show live progress instead of a frozen
  spinner.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from syncaudio.ffmpeg_backend import FFmpegError, probe_audio_streams
from syncaudio.jobs import Job, get_job, start_job
from syncaudio.models import AudioTrackSpec
from syncaudio.render import (
    SegmentedTrackCorrection,
    TrackCorrection,
    plan_corrections,
    plan_segmented_correction,
    render as render_tracks,
)
from syncaudio.segments import DEFAULT_HOP_S, DEFAULT_MARGIN_S, DEFAULT_WINDOW_S, Segment, detect_segments

app = FastAPI(title="SyncAudio", version="0.1.0")

_NO_LOG: Callable[[str], None] = lambda _msg: None  # noqa: E731


class TrackRef(BaseModel):
    """A single track, addressed like the CLI's ``chemin`` / ``chemin@INDEX``."""

    path: str
    index: int | None = None

    def to_spec(self) -> AudioTrackSpec:
        raw = f"{self.path}@{self.index}" if self.index is not None else self.path
        return AudioTrackSpec(raw=raw, path=self.path, stream_index=self.index)


def _http_error(exc: FFmpegError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


class JobStarted(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    messages: list[str]
    result: dict | None = None
    error: str | None = None


def _job_status_response(job: Job) -> JobStatusResponse:
    messages, _cursor, status, result, error = job.snapshot()
    return JobStatusResponse(job_id=job.id, status=status, messages=messages, result=result, error=error)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job inconnu : {job_id}")
    return _job_status_response(job)


@app.websocket("/jobs/{job_id}/ws")
async def job_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream a job's progress messages as they happen, ending with its result or error."""
    await websocket.accept()
    job = get_job(job_id)
    if job is None:
        await websocket.send_json({"type": "error", "message": f"Job inconnu : {job_id}"})
        await websocket.close()
        return

    since = 0
    try:
        while True:
            new_messages, since, status, result, error = job.snapshot(since)
            for message in new_messages:
                await websocket.send_json({"type": "log", "message": message})
            if status != "running":
                if status == "done":
                    await websocket.send_json({"type": "done", "result": result})
                else:
                    await websocket.send_json({"type": "error", "message": error})
                break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
    await websocket.close()


class TrackInfo(BaseModel):
    index: int
    codec: str | None
    language: str | None
    channels: int | None
    sample_rate: int | None


class ProbeResponse(BaseModel):
    path: str
    tracks: list[TrackInfo]


@app.get("/probe", response_model=ProbeResponse)
def probe(path: str) -> ProbeResponse:
    try:
        streams = probe_audio_streams(path)
    except FFmpegError as exc:
        raise _http_error(exc) from exc
    return ProbeResponse(
        path=path,
        tracks=[
            TrackInfo(index=s.index, codec=s.codec, language=s.language, channels=s.channels, sample_rate=s.sample_rate)
            for s in streams
        ],
    )


class AlignRequest(BaseModel):
    reference: TrackRef
    candidates: list[TrackRef]
    start: float = 0.0
    duration: float | None = None


class AlignResult(BaseModel):
    track: str
    language: str | None
    offset_seconds: float
    confidence: float
    ambiguous: bool


class AlignResponse(BaseModel):
    reference: str
    results: list[AlignResult]


def _do_align(req: AlignRequest, log: Callable[[str], None] = _NO_LOG) -> AlignResponse:
    try:
        corrections = plan_corrections(
            req.reference.to_spec(),
            [c.to_spec() for c in req.candidates],
            start=req.start,
            duration=req.duration,
            log=log,
        )
    except FFmpegError as exc:
        raise _http_error(exc) from exc
    return AlignResponse(
        reference=req.reference.to_spec().raw,
        results=[
            AlignResult(
                track=c.track.raw,
                language=c.language,
                offset_seconds=c.offset_seconds,
                confidence=c.confidence,
                ambiguous=c.ambiguous,
            )
            for c in corrections
        ],
    )


@app.post("/align", response_model=AlignResponse)
def align(req: AlignRequest) -> AlignResponse:
    return _do_align(req)


@app.post("/jobs/align", response_model=JobStarted)
def start_align_job(req: AlignRequest) -> JobStarted:
    job = start_job(lambda log: _do_align(req, log).model_dump())
    return JobStarted(job_id=job.id)


class SegmentsRequest(BaseModel):
    reference: TrackRef
    track: TrackRef
    start: float = 0.0
    duration: float | None = None
    window_s: float = DEFAULT_WINDOW_S
    hop_s: float = DEFAULT_HOP_S
    margin_s: float = DEFAULT_MARGIN_S


class SegmentOut(BaseModel):
    start_s: float
    end_s: float
    offset_start: float
    offset_end: float
    is_drift: bool

    @staticmethod
    def from_segment(seg: Segment) -> SegmentOut:
        return SegmentOut(
            start_s=seg.start_s, end_s=seg.end_s, offset_start=seg.offset_start, offset_end=seg.offset_end,
            is_drift=seg.is_drift,
        )

    def to_segment(self) -> Segment:
        return Segment(start_s=self.start_s, end_s=self.end_s, offset_start=self.offset_start, offset_end=self.offset_end)


class SegmentsResponse(BaseModel):
    reference: str
    track: str
    segments: list[SegmentOut]


def _do_segments(req: SegmentsRequest, log: Callable[[str], None] = _NO_LOG) -> SegmentsResponse:
    try:
        segs = detect_segments(
            req.reference.to_spec(),
            req.track.to_spec(),
            start=req.start,
            duration=req.duration,
            window_s=req.window_s,
            hop_s=req.hop_s,
            margin_s=req.margin_s,
            log=log,
        )
    except FFmpegError as exc:
        raise _http_error(exc) from exc
    return SegmentsResponse(
        reference=req.reference.to_spec().raw,
        track=req.track.to_spec().raw,
        segments=[SegmentOut.from_segment(s) for s in segs],
    )


@app.post("/segments", response_model=SegmentsResponse)
def segments_endpoint(req: SegmentsRequest) -> SegmentsResponse:
    return _do_segments(req)


@app.post("/jobs/segments", response_model=JobStarted)
def start_segments_job(req: SegmentsRequest) -> JobStarted:
    job = start_job(lambda log: _do_segments(req, log).model_dump())
    return JobStarted(job_id=job.id)


class SubsPair(BaseModel):
    subs: TrackRef
    audio: TrackRef


def _track_key(spec: AudioTrackSpec) -> tuple[str, int]:
    idx = spec.stream_index if spec.stream_index is not None else 0
    return (str(Path(spec.path).resolve()), idx)


class RenderRequest(BaseModel):
    input_path: str
    reference_index: int
    track_indices: list[int] | None = None
    only_imports: bool = False
    import_audio: list[TrackRef] = []
    subs: list[SubsPair] = []
    output_path: str | None = None
    audio_only: bool = False
    segmented: bool = False
    window_s: float = DEFAULT_WINDOW_S
    hop_s: float = DEFAULT_HOP_S
    margin_s: float = DEFAULT_MARGIN_S
    start: float = 0.0
    duration: float | None = None


class RenderedTrack(BaseModel):
    track: str
    language: str | None
    offset_seconds: float | None = None  # None for a segmented (non-constant) correction
    segments: list[SegmentOut] | None = None


class RenderResponse(BaseModel):
    written: list[str]
    corrections: list[RenderedTrack]


def _resolve_targets(input_path: str, reference_index: int, track_indices: list[int] | None, only_imports: bool) -> list[int]:
    try:
        streams = probe_audio_streams(input_path)
    except FFmpegError as exc:
        raise _http_error(exc) from exc
    all_indices = [s.index for s in streams]
    if reference_index not in all_indices:
        raise HTTPException(400, f"Index de référence {reference_index} absent de {input_path!r} (pistes : {all_indices}).")

    if only_imports and track_indices:
        raise HTTPException(400, "only_imports et track_indices sont incompatibles.")
    if only_imports:
        targets: list[int] = []
    elif track_indices:
        targets = sorted(track_indices)
    else:
        targets = [i for i in all_indices if i != reference_index]
    if reference_index in targets:
        raise HTTPException(400, "La piste de référence ne peut pas aussi être une piste à corriger.")
    unknown = [i for i in targets if i not in all_indices]
    if unknown:
        raise HTTPException(400, f"Index(es) inconnu(s) : {unknown} (pistes disponibles : {all_indices}).")
    return targets


def _do_render(req: RenderRequest, log: Callable[[str], None] = _NO_LOG) -> RenderResponse:
    targets = _resolve_targets(req.input_path, req.reference_index, req.track_indices, req.only_imports)

    for ref in req.import_audio:
        try:
            ext_streams = {s.index for s in probe_audio_streams(ref.path)}
        except FFmpegError as exc:
            raise _http_error(exc) from exc
        idx = ref.index if ref.index is not None else 0
        if idx not in ext_streams:
            raise HTTPException(400, f"Index audio {idx} absent de {ref.path!r} (pistes : {sorted(ext_streams)}).")

    reference_spec = AudioTrackSpec(raw=f"{req.input_path}@{req.reference_index}", path=req.input_path, stream_index=req.reference_index)
    same_file_specs = [AudioTrackSpec(raw=f"{req.input_path}@{i}", path=req.input_path, stream_index=i) for i in targets]
    import_audio_specs = [ref.to_spec() for ref in req.import_audio]
    candidates = same_file_specs + import_audio_specs
    candidate_keys = [_track_key(c) for c in candidates]

    subs_positions: list[int] = []
    for pair in req.subs:
        key = _track_key(pair.audio.to_spec())
        if key not in candidate_keys:
            raise HTTPException(
                400,
                f"subs {pair.subs.path}@{pair.subs.index} : la piste audio indiquée ne correspond à aucune piste "
                "corrigée (track_indices ou import_audio, même fichier@index).",
            )
        subs_positions.append(candidate_keys.index(key))

    if req.output_path is None:
        stem = Path(req.input_path)
        while stem.suffix:
            stem = stem.with_suffix("")
        output_path = str(stem) + ".synced.mkv"
    else:
        output_path = req.output_path

    try:
        if req.segmented:
            seg_corrections: list[SegmentedTrackCorrection] = [
                plan_segmented_correction(
                    reference_spec, spec, start=req.start, duration=req.duration,
                    window_s=req.window_s, hop_s=req.hop_s, margin_s=req.margin_s, log=log,
                )
                for spec in candidates
            ]
            segmented_imported_subs = [
                (pair.subs.to_spec(), seg_corrections[pos].segments) for pair, pos in zip(req.subs, subs_positions)
            ]
            log(f"[rendu] écriture de {output_path} ...")
            written = render_tracks(
                req.input_path, req.reference_index, corrections=[], output_path=output_path,
                audio_only=req.audio_only, segmented_corrections=seg_corrections,
                segmented_imported_subs=segmented_imported_subs,
            )
            corrections_out = [
                RenderedTrack(track=sc.track.raw, language=sc.language, segments=[SegmentOut.from_segment(s) for s in sc.segments])
                for sc in seg_corrections
            ]
        else:
            corrections: list[TrackCorrection] = plan_corrections(
                reference_spec, candidates, start=req.start, duration=req.duration, log=log
            )
            imported_subs = [(pair.subs.to_spec(), corrections[pos].offset_seconds) for pair, pos in zip(req.subs, subs_positions)]
            log(f"[rendu] écriture de {output_path} ...")
            written = render_tracks(
                req.input_path, req.reference_index, corrections, output_path,
                audio_only=req.audio_only, imported_subs=imported_subs,
            )
            corrections_out = [
                RenderedTrack(track=c.track.raw, language=c.language, offset_seconds=c.offset_seconds) for c in corrections
            ]
    except FFmpegError as exc:
        raise _http_error(exc) from exc

    return RenderResponse(written=written, corrections=corrections_out)


@app.post("/render", response_model=RenderResponse)
def render_endpoint(req: RenderRequest) -> RenderResponse:
    return _do_render(req)


@app.post("/jobs/render", response_model=JobStarted)
def start_render_job(req: RenderRequest) -> JobStarted:
    job = start_job(lambda log: _do_render(req, log).model_dump())
    return JobStarted(job_id=job.id)
