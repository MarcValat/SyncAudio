from __future__ import annotations

import pytest

from syncaudio.segments import Segment
from syncaudio.subtitles import format_for_codec, remap_time, shift_subtitle_text


def test_format_for_codec_known() -> None:
    assert format_for_codec("subrip") == "srt"
    assert format_for_codec("ass") == "ass"
    assert format_for_codec("SSA") == "ass"


def test_format_for_codec_unknown_raises() -> None:
    with pytest.raises(ValueError):
        format_for_codec("hdmv_pgs_subtitle")


def test_remap_time_constant_segment_is_a_plain_shift() -> None:
    # Candidate lags the reference by 5s throughout -> a cue at candidate-time
    # 65s belongs at reference-time 60s in the corrected output.
    segs = [Segment(0.0, 100.0, 5.0, 5.0)]
    assert remap_time(65.0, segs) == pytest.approx(60.0)


def test_remap_time_drift_segment_interpolates() -> None:
    # Candidate starts in sync and drifts to +10s lag by t=100 (reference
    # time): the candidate-time span [0, 110] maps onto reference [0, 100].
    segs = [Segment(0.0, 100.0, 0.0, 10.0)]
    assert remap_time(0.0, segs) == pytest.approx(0.0)
    assert remap_time(110.0, segs) == pytest.approx(100.0)
    assert remap_time(55.0, segs) == pytest.approx(50.0)


def test_remap_time_picks_the_right_segment_across_a_jump() -> None:
    segs = [
        Segment(0.0, 50.0, 0.0, 0.0),
        Segment(50.0, 100.0, 4.0, 4.0),
    ]
    assert remap_time(25.0, segs) == pytest.approx(25.0)
    assert remap_time(80.0, segs) == pytest.approx(76.0)


def test_shift_subtitle_text_srt() -> None:
    srt = (
        "1\n00:00:10,000 --> 00:00:12,000\nHello\n\n"
        "2\n00:00:20,000 --> 00:00:22,000\nWorld\n"
    )
    segs = [Segment(0.0, 100.0, 5.0, 5.0)]
    shifted = shift_subtitle_text(srt, segs, "srt")
    assert "00:00:05,000 --> 00:00:07,000" in shifted
    assert "00:00:15,000 --> 00:00:17,000" in shifted
    assert "Hello" in shifted and "World" in shifted


def test_shift_subtitle_text_ass() -> None:
    ass = "Dialogue: 0,0:00:10.00,0:00:12.00,Default,,0,0,0,,Hello there\n"
    segs = [Segment(0.0, 100.0, 5.0, 5.0)]
    shifted = shift_subtitle_text(ass, segs, "ass")
    assert "0:00:05.00,0:00:07.00" in shifted
    assert "Hello there" in shifted


def test_shift_subtitle_text_unknown_format_raises() -> None:
    with pytest.raises(ValueError):
        shift_subtitle_text("whatever", [Segment(0.0, 1.0, 0.0, 0.0)], "vtt")
