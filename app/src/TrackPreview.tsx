import { useEffect, useRef, useState } from "react";
import { fetchClip, fetchWaveform, type SegmentOut } from "./api";
import { formatTime } from "./SegmentChart";
import { Waveform, type HighlightRegion } from "./Waveform";

const PREVIEW_DURATION_S = 12;
const WAVEFORM_BUCKETS = 800;
const MIN_VIEW_DURATION_S = 1;

/**
 * The reference-time offset applying at `t`, per the current segments
 * (linearly interpolated across a drift segment, clamped to the first/last
 * segment's edge offset outside the analyzed range). Used to pick where in
 * the candidate track a given reference moment actually is.
 */
function offsetAt(segments: SegmentOut[], t: number): number {
  if (segments.length === 0) return 0;
  if (t <= segments[0].start_s) return segments[0].offset_start;
  const last = segments[segments.length - 1];
  if (t >= last.end_s) return last.offset_end;
  for (const seg of segments) {
    if (t >= seg.start_s && t <= seg.end_s) {
      if (seg.end_s <= seg.start_s) return seg.offset_start;
      const frac = (t - seg.start_s) / (seg.end_s - seg.start_s);
      return seg.offset_start + frac * (seg.offset_end - seg.offset_start);
    }
  }
  return 0;
}

function segmentAt(segments: SegmentOut[], t: number): SegmentOut | null {
  return segments.find((s) => t >= s.start_s && t <= s.end_s) ?? segments[segments.length - 1] ?? null;
}

interface DiffRegions {
  /** Candidate-native time: the part of the original candidate that a
   * constant-offset correction cuts away. */
  removed: [number, number] | null;
  /** Reference time: the part of the corrected result that's added silence. */
  added: [number, number] | null;
}

/**
 * Git-diff-style removed/added regions for a constant-offset correction,
 * mirroring exactly what render.py's correction_filter (offset>0: trim the
 * candidate's head, pad the tail / offset<0: pad the head, the candidate's
 * tail runs past the reference's duration and is dropped) does to the whole
 * file. Only meaningful near the file's actual start/end -- for an interior
 * segment far from either edge, both regions naturally fall outside any
 * reasonable view and nothing is drawn. Deliberately NOT modelling gaps/
 * overlaps between adjacent segments at a jump boundary: that needs
 * knowledge of neighbouring segments, not just the one under the cursor --
 * scoped out for now, same as the rest of this preview.
 */
function computeDiffRegions(offset: number, refDuration: number, candDuration: number): DiffRegions {
  if (!isFinite(offset) || Math.abs(offset) < 1e-6 || refDuration <= 0 || candDuration <= 0) {
    return { removed: null, added: null };
  }
  if (offset > 0) {
    const removed: [number, number] = [0, Math.min(offset, candDuration)];
    const keptCandDuration = candDuration - offset;
    const added: [number, number] | null = keptCandDuration < refDuration ? [Math.max(0, keptCandDuration), refDuration] : null;
    return { removed, added };
  }
  const abs = -offset;
  const added: [number, number] = [0, Math.min(abs, refDuration)];
  const cutPoint = refDuration - abs;
  const removed: [number, number] | null = candDuration > cutPoint ? [Math.max(0, cutPoint), candDuration] : null;
  return { removed, added };
}

interface PeaksData {
  min: number[];
  max: number[];
  dataStart: number;
  dataEnd: number;
}

interface TrackPreviewProps {
  filePath: string;
  referenceIndex: number;
  trackIndex: number;
  segments: SegmentOut[];
}

/** Compare the reference and a candidate track together -- always-visible,
 * zoomable waveforms (reference / candidate as-is / corrected result, the
 * last two git-diff-highlighted) plus actual audio playback -- to check a
 * correction by eye and by ear before spending a full render on it. */
export function TrackPreview({ filePath, referenceIndex, trackIndex, segments }: TrackPreviewProps) {
  const [previewStart, setPreviewStart] = useState(() => (segments.length ? segments[0].start_s : 0));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refUrl, setRefUrl] = useState<string | null>(null);
  const [candUrl, setCandUrl] = useState<string | null>(null);
  // Where the currently-loaded <audio> clips actually start, in reference
  // time -- diverges from `previewStart` once an in-clip seek moves the
  // marker without reloading (see handleWaveformSeek), so currentTime math
  // must be anchored on this, not on `previewStart`.
  const [loadedClipStart, setLoadedClipStart] = useState<number | null>(null);
  const [refMuted, setRefMuted] = useState(false);
  const [candMuted, setCandMuted] = useState(false);
  // Only set while audio is actually playing (see startCursorLoop); when
  // null, the displayed marker falls back to `previewStart` below, so the
  // red line is always shown, not just during playback.
  const [liveCursor, setLiveCursor] = useState<number | null>(null);

  const [refDuration, setRefDuration] = useState<number | null>(null);
  const [candDuration, setCandDuration] = useState<number | null>(null);
  const [viewStart, setViewStart] = useState(0);
  const [viewDuration, setViewDuration] = useState<number | null>(null);
  const [refPeaks, setRefPeaks] = useState<PeaksData | null>(null);
  const [candPeaks, setCandPeaks] = useState<PeaksData | null>(null);
  const [finalPeaks, setFinalPeaks] = useState<PeaksData | null>(null);
  const [diff, setDiff] = useState<DiffRegions>({ removed: null, added: null });
  const [waveformLoading, setWaveformLoading] = useState(false);
  const [waveformError, setWaveformError] = useState<string | null>(null);

  const refAudioRef = useRef<HTMLAudioElement>(null);
  const candAudioRef = useRef<HTMLAudioElement>(null);
  const rafRef = useRef<number | null>(null);

  const appliedOffset = offsetAt(segments, previewStart);
  const displayCursor = liveCursor ?? previewStart;

  useEffect(() => {
    return () => {
      if (refUrl) URL.revokeObjectURL(refUrl);
      if (candUrl) URL.revokeObjectURL(candUrl);
    };
  }, [refUrl, candUrl]);

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Learn each track's total duration once, and default the view to the
  // reference's full length -- this is what makes the waveform "always
  // visible, whole track" instead of gated behind a play click.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [refWave, candWave] = await Promise.all([
          fetchWaveform(filePath, referenceIndex, 0, null, 8),
          fetchWaveform(filePath, trackIndex, 0, null, 8),
        ]);
        if (cancelled) return;
        setRefDuration(refWave.duration);
        setCandDuration(candWave.duration);
        setViewStart(0);
        setViewDuration(refWave.duration);
      } catch (err) {
        if (!cancelled) setWaveformError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filePath, referenceIndex, trackIndex]);

  // Re-fetch all three waveforms whenever the shared view (pan/zoom) changes.
  useEffect(() => {
    if (viewDuration === null || refDuration === null || candDuration === null) return;
    let cancelled = false;
    (async () => {
      setWaveformLoading(true);
      setWaveformError(null);
      try {
        const viewCenter = viewStart + viewDuration / 2;
        const offsetForView = offsetAt(segments, viewCenter);
        const seg = segmentAt(segments, viewCenter);

        const refStart = Math.max(0, viewStart);
        const refEnd = Math.min(refDuration, viewStart + viewDuration);
        const refPromise =
          refEnd > refStart ? fetchWaveform(filePath, referenceIndex, refStart, refEnd - refStart, WAVEFORM_BUCKETS) : null;

        // Track 2 (candidate, as-is): the SAME numeric window as the
        // reference view -- not offset-shifted -- so the correction is
        // directly visible as a spatial shift between the two waveforms,
        // the same way two git diff panes line up by position.
        const candStart = Math.max(0, viewStart);
        const candEnd = Math.min(candDuration, viewStart + viewDuration);
        const candPromise =
          candEnd > candStart ? fetchWaveform(filePath, trackIndex, candStart, candEnd - candStart, WAVEFORM_BUCKETS) : null;

        // Track 3 (corrected result): candidate content shifted by the
        // offset, i.e. what will actually occupy this reference-time span
        // after correction.
        const shiftedStart = Math.max(0, viewStart + offsetForView);
        const shiftedEnd = Math.min(candDuration, viewStart + offsetForView + viewDuration);
        const finalPromise =
          shiftedEnd > shiftedStart
            ? fetchWaveform(filePath, trackIndex, shiftedStart, shiftedEnd - shiftedStart, WAVEFORM_BUCKETS)
            : null;

        const [refWave, candWave, finalWave] = await Promise.all([refPromise, candPromise, finalPromise]);
        if (cancelled) return;

        setRefPeaks(refWave ? { min: refWave.peaks_min, max: refWave.peaks_max, dataStart: refStart, dataEnd: refStart + refWave.duration } : null);
        setCandPeaks(
          candWave ? { min: candWave.peaks_min, max: candWave.peaks_max, dataStart: candStart, dataEnd: candStart + candWave.duration } : null,
        );
        setFinalPeaks(
          finalWave
            ? {
                min: finalWave.peaks_min,
                max: finalWave.peaks_max,
                dataStart: shiftedStart - offsetForView,
                dataEnd: shiftedStart + finalWave.duration - offsetForView,
              }
            : null,
        );
        setDiff(seg && !seg.is_drift ? computeDiffRegions(offsetForView, refDuration, candDuration) : { removed: null, added: null });
      } catch (err) {
        if (!cancelled) setWaveformError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setWaveformLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filePath, referenceIndex, trackIndex, viewStart, viewDuration, refDuration, candDuration, segments]);

  function zoom(factor: number) {
    if (viewDuration === null || refDuration === null) return;
    const newDuration = Math.max(MIN_VIEW_DURATION_S, Math.min(refDuration, viewDuration * factor));
    let newStart = previewStart - newDuration / 2;
    newStart = Math.max(0, Math.min(Math.max(0, refDuration - newDuration), newStart));
    setViewStart(newStart);
    setViewDuration(newDuration);
  }

  function resetZoom() {
    if (refDuration === null) return;
    setViewStart(0);
    setViewDuration(refDuration);
  }

  function stopCursorLoop() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setLiveCursor(null);
  }

  function startCursorLoop(startAt: number) {
    function tick() {
      const refEl = refAudioRef.current;
      if (refEl && !refEl.paused && !refEl.ended) {
        setLiveCursor(startAt + refEl.currentTime);
        rafRef.current = requestAnimationFrame(tick);
      } else {
        stopCursorLoop();
      }
    }
    rafRef.current = requestAnimationFrame(tick);
  }

  /** Loads a fresh 12s clip pair starting at `startAt` (defaults to the
   * current position field) and plays them. Takes an explicit argument
   * rather than always reading `previewStart` so a click-to-seek during
   * playback (see handleWaveformSeek) can jump straight to the clicked
   * time without waiting for the state update to land first. */
  async function loadAndPlay(startAt: number = previewStart) {
    setLoading(true);
    setError(null);
    try {
      const candidateStart = Math.max(0, startAt + offsetAt(segments, startAt));
      const [refBlob, candBlob] = await Promise.all([
        fetchClip(filePath, referenceIndex, startAt, PREVIEW_DURATION_S),
        fetchClip(filePath, trackIndex, candidateStart, PREVIEW_DURATION_S),
      ]);
      setRefUrl(URL.createObjectURL(refBlob));
      setCandUrl(URL.createObjectURL(candBlob));
      setLoadedClipStart(startAt);
      setLoading(false);
      requestAnimationFrame(() => {
        const refEl = refAudioRef.current;
        const candEl = candAudioRef.current;
        if (refEl) {
          refEl.currentTime = 0;
          refEl.play().catch((e) => setError(`Lecture (référence) refusée par le navigateur : ${e}`));
        }
        if (candEl) {
          candEl.currentTime = 0;
          candEl.play().catch((e) => setError(`Lecture (piste corrigée) refusée par le navigateur : ${e}`));
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setLoading(false);
    }
  }

  /** Click-to-seek on any of the three waveforms. If the clicked time is
   * still within the clip currently loaded for playback, seek both <audio>
   * elements instantly (no network round trip); otherwise, if something
   * was playing, reload a fresh clip starting there so listening continues
   * uninterrupted instead of silently going stale. Either way, the position
   * field (and thus the always-visible marker) follows the click. */
  function handleWaveformSeek(t: number) {
    const rounded = Math.round(t * 10) / 10;
    const refEl = refAudioRef.current;
    const candEl = candAudioRef.current;
    const wasPlaying = !!refEl && !refEl.paused && !refEl.ended;
    const withinLoadedClip =
      loadedClipStart !== null && t >= loadedClipStart && t <= loadedClipStart + PREVIEW_DURATION_S;

    setPreviewStart(rounded);

    if (withinLoadedClip && refEl && candEl) {
      refEl.currentTime = t - loadedClipStart!;
      const loadedCandidateStart = loadedClipStart! + offsetAt(segments, loadedClipStart!);
      candEl.currentTime = Math.max(0, t + offsetAt(segments, t) - loadedCandidateStart);
      setLiveCursor(t);
      return;
    }

    if (wasPlaying) {
      loadAndPlay(rounded);
    }
  }

  /** Jump the position (and zoom the view) to segment `seg` -- avoids the
   * user having to eyeball the chart or hunt through a whole-track view to
   * land inside the right segment. */
  function goToSegment(seg: SegmentOut) {
    setPreviewStart(Math.round(((seg.start_s + seg.end_s) / 2) * 10) / 10);
    setViewStart(seg.start_s);
    setViewDuration(Math.max(MIN_VIEW_DURATION_S, seg.end_s - seg.start_s));
  }

  function stop() {
    if (refAudioRef.current) {
      refAudioRef.current.pause();
      refAudioRef.current.currentTime = 0;
    }
    if (candAudioRef.current) {
      candAudioRef.current.pause();
      candAudioRef.current.currentTime = 0;
    }
    stopCursorLoop();
  }

  const removedHighlight: HighlightRegion[] = diff.removed ? [{ start: diff.removed[0], end: diff.removed[1], kind: "removed" }] : [];
  const addedHighlight: HighlightRegion[] = diff.added ? [{ start: diff.added[0], end: diff.added[1], kind: "added" }] : [];

  return (
    <div className="preview">
      {segments.length > 1 && (
        <div className="preview-segment-picks">
          Aller à :
          {segments.map((seg, i) => (
            <button key={i} className="small-button" onClick={() => goToSegment(seg)}>
              {formatTime(seg.start_s)}–{formatTime(seg.end_s)} (
              {((seg.offset_start + seg.offset_end) / 2).toFixed(2)}s)
            </button>
          ))}
        </div>
      )}

      <div className="waveform-zoom-controls">
        <span>Zoom :</span>
        <button className="small-button" onClick={() => zoom(0.5)}>
          + (zoomer)
        </button>
        <button className="small-button" onClick={() => zoom(2)}>
          − (dézoomer)
        </button>
        <button className="small-button" onClick={resetZoom}>
          Piste entière
        </button>
        {waveformLoading && <span className="preview-offset">chargement...</span>}
      </div>

      {(diff.removed || diff.added) && (
        <div className="waveform-legend">
          <span>
            <span className="waveform-legend-swatch removed" /> sera supprimé
          </span>
          <span>
            <span className="waveform-legend-swatch added" /> sera ajouté (silence)
          </span>
        </div>
      )}

      {waveformError && <p className="error">{waveformError}</p>}

      {viewDuration !== null && (
        <div className="waveforms">
          <Waveform
            viewStart={viewStart}
            viewDuration={viewDuration}
            peaksMin={refPeaks?.min ?? null}
            peaksMax={refPeaks?.max ?? null}
            dataStart={refPeaks?.dataStart ?? viewStart}
            dataEnd={refPeaks?.dataEnd ?? viewStart}
            cursor={displayCursor}
            onSeek={handleWaveformSeek}
            label="Référence"
            className="waveform-reference"
          />
          <Waveform
            viewStart={viewStart}
            viewDuration={viewDuration}
            peaksMin={candPeaks?.min ?? null}
            peaksMax={candPeaks?.max ?? null}
            dataStart={candPeaks?.dataStart ?? viewStart}
            dataEnd={candPeaks?.dataEnd ?? viewStart}
            cursor={displayCursor}
            onSeek={handleWaveformSeek}
            highlights={removedHighlight}
            label="Piste corrigée (originale)"
            className="waveform-candidate"
          />
          <Waveform
            viewStart={viewStart}
            viewDuration={viewDuration}
            peaksMin={finalPeaks?.min ?? null}
            peaksMax={finalPeaks?.max ?? null}
            dataStart={finalPeaks?.dataStart ?? viewStart}
            dataEnd={finalPeaks?.dataEnd ?? viewStart}
            cursor={displayCursor}
            onSeek={handleWaveformSeek}
            highlights={addedHighlight}
            label="Résultat final"
            className="waveform-final"
          />
        </div>
      )}

      <div className="preview-controls">
        <label>
          Position (s) :
          <input
            type="number"
            step="0.5"
            min="0"
            value={previewStart}
            onChange={(e) => setPreviewStart(Number(e.target.value))}
          />
        </label>
        <button className="small-button" onClick={() => loadAndPlay()} disabled={loading}>
          {loading ? "Chargement..." : "Écouter"}
        </button>
        <button className="small-button" onClick={stop} disabled={!refUrl && !candUrl}>
          Arrêter
        </button>
        <label>
          <input type="checkbox" checked={!refMuted} onChange={() => setRefMuted((m) => !m)} />
          Référence
        </label>
        <label>
          <input type="checkbox" checked={!candMuted} onChange={() => setCandMuted((m) => !m)} />
          Piste corrigée
        </label>
        <span className="preview-offset">
          Décalage appliqué : {appliedOffset.toFixed(3)} s (piste corrigée lue à partir de{" "}
          {Math.max(0, previewStart + appliedOffset).toFixed(3)} s)
        </span>
      </div>
      {error && <p className="error">{error}</p>}
      <audio
        ref={refAudioRef}
        src={refUrl ?? undefined}
        muted={refMuted}
        onPlay={() => startCursorLoop(previewStart)}
        onPause={stopCursorLoop}
        onEnded={stopCursorLoop}
      />
      <audio ref={candAudioRef} src={candUrl ?? undefined} muted={candMuted} />
    </div>
  );
}
