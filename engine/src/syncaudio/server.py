"""FastAPI sidecar exposing the engine over HTTP.

Thin layer over the same functions the CLI uses (``render.py``,
``segments.py``, ``ffmpeg_backend.py``) -- the CLI remains a perfectly valid
client on its own; this is an additional one for the future GUI. Endpoints
are synchronous (Starlette runs ``def`` routes in a threadpool, so one
in-flight request doesn't block others), which keeps this first version
simple; progress streaming for long-running renders is a planned follow-up,
not yet implemented.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from syncaudio.ffmpeg_backend import FFmpegError, probe_audio_streams
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


class TrackRef(BaseModel):
    """A single track, addressed like the CLI's ``chemin`` / ``chemin@INDEX``."""

    path: str
    index: int | None = None

    def to_spec(self) -> AudioTrackSpec:
        raw = f"{self.path}@{self.index}" if self.index is not None else self.path
        return AudioTrackSpec(raw=raw, path=self.path, stream_index=self.index)


def _http_error(exc: FFmpegError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


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


@app.post("/align", response_model=AlignResponse)
def align(req: AlignRequest) -> AlignResponse:
    try:
        corrections = plan_corrections(
            req.reference.to_spec(),
            [c.to_spec() for c in req.candidates],
            start=req.start,
            duration=req.duration,
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


@app.post("/segments", response_model=SegmentsResponse)
def segments_endpoint(req: SegmentsRequest) -> SegmentsResponse:
    try:
        segs = detect_segments(
            req.reference.to_spec(),
            req.track.to_spec(),
            start=req.start,
            duration=req.duration,
            window_s=req.window_s,
            hop_s=req.hop_s,
            margin_s=req.margin_s,
        )
    except FFmpegError as exc:
        raise _http_error(exc) from exc
    return SegmentsResponse(
        reference=req.reference.to_spec().raw,
        track=req.track.to_spec().raw,
        segments=[SegmentOut.from_segment(s) for s in segs],
    )


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


@app.post("/render", response_model=RenderResponse)
def render_endpoint(req: RenderRequest) -> RenderResponse:
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
                    window_s=req.window_s, hop_s=req.hop_s, margin_s=req.margin_s,
                )
                for spec in candidates
            ]
            segmented_imported_subs = [
                (pair.subs.to_spec(), seg_corrections[pos].segments) for pair, pos in zip(req.subs, subs_positions)
            ]
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
                reference_spec, candidates, start=req.start, duration=req.duration
            )
            imported_subs = [(pair.subs.to_spec(), corrections[pos].offset_seconds) for pair, pos in zip(req.subs, subs_positions)]
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
