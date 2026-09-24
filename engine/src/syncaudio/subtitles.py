from __future__ import annotations

import re
from collections.abc import Sequence

from syncaudio.segments import Segment

# Matches SRT ("00:01:23,450") and ASS/SSA ("0:01:23.45") timestamps
# wherever they appear in the file -- both formats only use this pattern for
# cue timing, so a blind substitution (rather than a full parser) is enough
# to shift every cue without disturbing anything else (styling, text, ...).
_SRT_TS = re.compile(r"\d{2}:\d{2}:\d{2},\d{3}")
_ASS_TS = re.compile(r"\d+:\d{2}:\d{2}\.\d{2}")

_CODEC_TO_FORMAT = {
    "subrip": "srt",
    "ass": "ass",
    "ssa": "ass",
}


def format_for_codec(codec: str) -> str:
    """Map a probed subtitle codec name to the format ``shift_subtitle_text`` understands."""
    fmt = _CODEC_TO_FORMAT.get(codec.strip().lower())
    if fmt is None:
        raise ValueError(
            f"Format de sous-titres non supporté pour --segmented : {codec!r} (seuls srt/ass/ssa le sont)."
        )
    return fmt


def _parse_srt_time(s: str) -> float:
    hms, ms = s.split(",")
    h, m, sec = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def _format_srt_time(t: float) -> str:
    total_ms = round(max(t, 0.0) * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_ass_time(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _format_ass_time(t: float) -> str:
    total_cs = round(max(t, 0.0) * 100)
    h, rem = divmod(total_cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def remap_time(t: float, segments: Sequence[Segment]) -> float:
    """Map a time in the candidate's ORIGINAL timeline to the corrected output timeline.

    Inverts the same affine per-segment relationship ``render.segment_correction_filter``
    uses to extract audio: within a segment, ``candidate_time = ref_time +
    offset_start + slope*(ref_time - start_s)``. A subtitle cue at
    ``candidate_time`` therefore belongs at ``ref_time`` in the corrected
    (reference-timeline) output. Cues outside every segment's candidate-time
    span (rare -- right at a file's very start/end) are extrapolated with a
    plain shift from the nearest segment's edge offset.
    """
    if not segments:
        return t
    for seg in segments:
        cand_start = seg.start_s + seg.offset_start
        cand_end = seg.end_s + seg.offset_end
        if cand_start <= t <= cand_end:
            duration = seg.end_s - seg.start_s
            factor = (cand_end - cand_start) / duration if duration > 1e-9 else 1.0
            if factor <= 0:
                factor = 1.0
            return seg.start_s + (t - cand_start) / factor

    first, last = segments[0], segments[-1]
    if t < first.start_s + first.offset_start:
        return t - first.offset_start
    return t - last.offset_end


def shift_subtitle_text(text: str, segments: Sequence[Segment], fmt: str) -> str:
    """Rewrite every cue timestamp in ``text`` per ``remap_time``."""
    if fmt == "srt":
        parse, render_time, pattern = _parse_srt_time, _format_srt_time, _SRT_TS
    elif fmt == "ass":
        parse, render_time, pattern = _parse_ass_time, _format_ass_time, _ASS_TS
    else:
        raise ValueError(f"Format de sous-titres inconnu : {fmt!r}")

    def repl(match: re.Match[str]) -> str:
        return render_time(remap_time(parse(match.group(0)), segments))

    return pattern.sub(repl, text)
