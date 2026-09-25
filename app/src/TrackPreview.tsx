import { useEffect, useMemo, useRef, useState } from "react";
import { fetchClip, fetchWaveform, type SegmentOut } from "./api";
import { formatTime } from "./SegmentChart";
import { Waveform, type HighlightRegion } from "./Waveform";

const PREVIEW_DURATION_S = 12;
const WAVEFORM_BUCKETS = 800;
// Below this, there's nothing more to see: the panel isn't wide enough for
// finer detail to matter, and the diff highlight is computed analytically
// anyway, not read off the waveform pixel by pixel.
const MIN_VIEW_DURATION_S = 20;
// Fetched once per track, whole-file, so every zoom/pan afterwards is a pure
// client-side resample (see resamplePeaks) instead of a network round trip.
// At this bucket count a typical (5-45min) episode stays comfortably sharp
// down to the MIN_VIEW_DURATION_S floor above.
const FULL_TRACK_BUCKETS = 20000;

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

interface FullPeaks {
  min: number[];
  max: number[];
}

/**
 * Derives a `outBuckets`-wide view [viewStart, viewEnd) from a pre-fetched
 * whole-track peaks array (spanning [0, full duration]) -- min-of-mins /
 * max-of-maxes over the source buckets each output bucket covers, which is
 * exact (no precision lost beyond what the original server-side bucketing
 * already introduced). This is what makes zoom/pan instant: no network call,
 * just re-slicing data already in memory. Below the source's own
 * resolution (zoomed in tighter than FULL_TRACK_BUCKETS can resolve), output
 * buckets start repeating the same one or two source buckets -- a known,
 * accepted trade-off (see FULL_TRACK_BUCKETS/MIN_VIEW_DURATION_S) rather
 * than falling back to a live fetch.
 */
function resamplePeaks(full: FullPeaks, fullDuration: number, viewStart: number, viewEnd: number, outBuckets: number): PeaksData {
  const n = full.min.length;
  if (n === 0 || fullDuration <= 0) return { min: [], max: [], dataStart: viewStart, dataEnd: viewStart };

  const srcBucketDur = fullDuration / n;
  const clampedStart = Math.max(0, viewStart);
  const clampedEnd = Math.min(fullDuration, viewEnd);
  if (clampedEnd <= clampedStart) return { min: [], max: [], dataStart: viewStart, dataEnd: viewStart };

  const startIdx = Math.max(0, Math.floor(clampedStart / srcBucketDur));
  const endIdx = Math.min(n, Math.ceil(clampedEnd / srcBucketDur));
  const sliceLen = endIdx - startIdx;
  if (sliceLen <= 0) return { min: [], max: [], dataStart: viewStart, dataEnd: viewStart };

  const outMin = new Array<number>(outBuckets);
  const outMax = new Array<number>(outBuckets);
  for (let i = 0; i < outBuckets; i++) {
    const a = startIdx + Math.floor((i / outBuckets) * sliceLen);
    const b = startIdx + Math.floor(((i + 1) / outBuckets) * sliceLen);
    const lo = Math.min(a, n - 1);
    const hi = Math.max(lo + 1, Math.min(b, n));
    let mn = full.min[lo];
    let mx = full.max[lo];
    for (let j = lo; j < hi; j++) {
      if (full.min[j] < mn) mn = full.min[j];
      if (full.max[j] > mx) mx = full.max[j];
    }
    outMin[i] = mn;
    outMax[i] = mx;
  }
  return { min: outMin, max: outMax, dataStart: startIdx * srcBucketDur, dataEnd: endIdx * srcBucketDur };
}

/** Fetches+decodes a clip into a ready-to-schedule AudioBuffer. Decoding
 * up front (rather than handing a <audio> element a src and hoping it
 * buffers in time) is what lets playback below start multiple sources at a
 * genuinely identical, sample-accurate instant -- see playSource. */
async function decodeClip(ctx: AudioContext, blob: Blob): Promise<AudioBuffer> {
  const arrayBuffer = await blob.arrayBuffer();
  return ctx.decodeAudioData(arrayBuffer);
}

/** Stops (if playing) and detaches a previous source; a AudioBufferSourceNode
 * can only ever be started once, so every (re)play/seek creates a fresh one. */
function stopSource(ref: React.MutableRefObject<AudioBufferSourceNode | null>) {
  if (ref.current) {
    try {
      ref.current.stop();
    } catch {
      // Already stopped or never started -- fine, that's what we wanted anyway.
    }
    try {
      ref.current.disconnect();
    } catch {
      // noop
    }
    ref.current = null;
  }
}

/** Schedules `buffer` to start at the AudioContext-clock instant `when`,
 * `offset` seconds into the buffer. `when`/`offset` being expressed on the
 * shared audio clock (not JS timers, not per-element readiness) is what
 * guarantees multiple sources started this way are audibly simultaneous. */
function playSource(ctx: AudioContext, buffer: AudioBuffer, gain: GainNode, when: number, offset: number): AudioBufferSourceNode {
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(gain);
  source.start(when, Math.max(0, Math.min(offset, buffer.duration)));
  return source;
}

interface TrackPreviewProps {
  filePath: string;
  referenceIndex: number;
  trackIndex: number;
  segments: SegmentOut[];
  /** Each track's container-level presentation delay (0 if none), purely
   * informational: every extraction here (waveforms, clips, detection) works
   * on each track's own timeline with that delay excluded (see the engine's
   * _seek_args), so no playback/waveform math uses these values. */
  referenceStartTime: number;
  trackStartTime: number;
}

/** Compare the reference and a candidate track together -- always-visible,
 * zoomable waveforms (reference / candidate as-is / corrected result, the
 * last two git-diff-highlighted) plus actual audio playback -- to check a
 * correction by eye and by ear before spending a full render on it.
 *
 * Playback uses the Web Audio API (AudioContext + AudioBufferSourceNode),
 * not plain <audio> elements: three separate <audio>.play() calls have no
 * guaranteed simultaneity (each has its own, variable, buffering-dependent
 * startup latency), which was audible as "Résultat final starts late"
 * whenever there wasn't a deliberate leading-silence wait long enough to
 * absorb that jitter for free. Scheduling all three AudioBufferSourceNodes
 * against the same AudioContext clock (`start(when, offset)`) starts them
 * at a genuinely identical instant instead. */
export function TrackPreview({
  filePath,
  referenceIndex,
  trackIndex,
  segments,
  referenceStartTime,
  trackStartTime,
}: TrackPreviewProps) {
  const [previewStart, setPreviewStart] = useState(() => (segments.length ? segments[0].start_s : 0));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Where the currently-loaded clips actually start, in reference time --
  // diverges from `previewStart` once an in-clip seek moves the marker
  // without reloading (see handleWaveformSeek), so playback-offset math
  // must be anchored on this, not on `previewStart`.
  const [loadedClipStart, setLoadedClipStart] = useState<number | null>(null);
  // The exact candidate-time position actually requested from /clip for
  // "Résultat final" at the last real fetch -- captured at fetch time, not
  // recomputed live, so it can be compared against what offsetAt(segments,
  // previewStart) says *now* to tell a genuinely stale load apart from a
  // wrong computation.
  const [loadedCandidateStart, setLoadedCandidateStart] = useState<number | null>(null);
  const [refMuted, setRefMuted] = useState(false);
  const [candMuted, setCandMuted] = useState(false);
  const [candOriginalMuted, setCandOriginalMuted] = useState(false);
  // Only set while audio is actually playing (see startCursorLoop); when
  // null, the displayed marker falls back to `previewStart` below, so the
  // red line is always shown, not just during playback.
  const [liveCursor, setLiveCursor] = useState<number | null>(null);

  const [refDuration, setRefDuration] = useState<number | null>(null);
  const [candDuration, setCandDuration] = useState<number | null>(null);
  const [viewStart, setViewStart] = useState(0);
  const [viewDuration, setViewDuration] = useState<number | null>(null);
  // Whole-track peaks, fetched once (see the prefetch effect below) -- every
  // zoom/pan re-derives its view from these via resamplePeaks, no refetch.
  const [refFullPeaks, setRefFullPeaks] = useState<FullPeaks | null>(null);
  const [candFullPeaks, setCandFullPeaks] = useState<FullPeaks | null>(null);
  const [waveformLoading, setWaveformLoading] = useState(false);
  const [waveformError, setWaveformError] = useState<string | null>(null);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const refGainRef = useRef<GainNode | null>(null);
  const candGainRef = useRef<GainNode | null>(null);
  const candOriginalGainRef = useRef<GainNode | null>(null);
  const refSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const candSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const candOriginalSourceRef = useRef<AudioBufferSourceNode | null>(null);
  // The currently-loaded, already-decoded buffers -- kept around so an
  // in-clip seek (handleWaveformSeek) can reschedule instantly from memory
  // instead of re-fetching/re-decoding.
  const refBufferRef = useRef<AudioBuffer | null>(null);
  const candBufferRef = useRef<AudioBuffer | null>(null);
  const candOriginalBufferRef = useRef<AudioBuffer | null>(null);
  // When playback last (re)started: the AudioContext-clock instant it began
  // at, and what reference-time that corresponds to -- the cursor loop below
  // derives the live position from `ctx.currentTime - contextTime`.
  const playbackStartRef = useRef<{ contextTime: number; refTime: number } | null>(null);
  const rafRef = useRef<number | null>(null);
  // Bumped on every loadAndPlay call; lets a stale call recognize it's been
  // superseded by a newer one (e.g. a rapid re-seek) and ignore its own
  // late-arriving fetch/decode results instead of clobbering a newer load.
  const loadGenerationRef = useRef(0);

  const appliedOffset = offsetAt(segments, previewStart);
  const displayCursor = liveCursor ?? previewStart;
  // Informational only (see TrackPreviewProps' comment): the offset above is
  // measured on each track's own timeline, container delay excluded, so a
  // normal player (which applies that delay) would see this residual instead.
  const hasContainerDelay = Math.abs(trackStartTime) > 0.001 || Math.abs(referenceStartTime) > 0.001;
  const presentationOffset = appliedOffset + trackStartTime - referenceStartTime;

  function getAudioCtx(): AudioContext {
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
    return audioCtxRef.current;
  }

  function getGain(ref: React.MutableRefObject<GainNode | null>, ctx: AudioContext, muted: boolean): GainNode {
    if (!ref.current) {
      ref.current = ctx.createGain();
      ref.current.connect(ctx.destination);
    }
    ref.current.gain.value = muted ? 0 : 1;
    return ref.current;
  }

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      stopSource(refSourceRef);
      stopSource(candSourceRef);
      stopSource(candOriginalSourceRef);
      audioCtxRef.current?.close().catch(() => {});
    };
  }, []);

  // Fetch each track's whole-file peaks once, at high enough resolution
  // that every zoom/pan afterwards is a pure client-side resample -- this is
  // what makes the waveform "always visible, whole track" *and* instant to
  // navigate, instead of gated behind a play click or a fetch per zoom step.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setWaveformLoading(true);
      setWaveformError(null);
      try {
        const [refWave, candWave] = await Promise.all([
          fetchWaveform(filePath, referenceIndex, 0, null, FULL_TRACK_BUCKETS),
          fetchWaveform(filePath, trackIndex, 0, null, FULL_TRACK_BUCKETS),
        ]);
        if (cancelled) return;
        setRefDuration(refWave.duration);
        setCandDuration(candWave.duration);
        setRefFullPeaks({ min: refWave.peaks_min, max: refWave.peaks_max });
        setCandFullPeaks({ min: candWave.peaks_min, max: candWave.peaks_max });
        setViewStart(0);
        setViewDuration(refWave.duration);
      } catch (err) {
        if (!cancelled) setWaveformError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setWaveformLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [filePath, referenceIndex, trackIndex]);

  // Re-derive all three waveforms' data for the current view from the
  // already-fetched full-track peaks -- synchronous, no network round trip,
  // so zoom/pan feels instant no matter how far in or out.
  const viewData = useMemo(() => {
    if (viewDuration === null || refDuration === null || candDuration === null || !refFullPeaks || !candFullPeaks) {
      return null;
    }
    // Anchored on previewStart (the actual listening position), not the
    // view's center -- these must agree exactly with loadAndPlay's own
    // offsetAt(segments, startAt) call, or "Résultat final" can look
    // correctly aligned in a wide/zoomed-out view while the audio actually
    // played back (computed at a different point) uses a different offset,
    // e.g. across a segment that isn't perfectly flat after a manual merge.
    const offsetForView = offsetAt(segments, previewStart);
    const seg = segmentAt(segments, previewStart);

    const refPeaks = resamplePeaks(refFullPeaks, refDuration, viewStart, viewStart + viewDuration, WAVEFORM_BUCKETS);
    // Track 2 (candidate, as-is): the SAME numeric window as the reference
    // view -- not offset-shifted -- so the correction is directly visible as
    // a spatial shift between the two waveforms, the same way two git diff
    // panes line up by position.
    const candPeaks = resamplePeaks(candFullPeaks, candDuration, viewStart, viewStart + viewDuration, WAVEFORM_BUCKETS);

    // Track 3 (corrected result): candidate content shifted by the offset,
    // i.e. what will actually occupy this reference-time span after
    // correction -- same underlying candidate data, just a different slice.
    const shifted = resamplePeaks(candFullPeaks, candDuration, viewStart + offsetForView, viewStart + offsetForView + viewDuration, WAVEFORM_BUCKETS);
    const finalPeaks: PeaksData = { min: shifted.min, max: shifted.max, dataStart: shifted.dataStart - offsetForView, dataEnd: shifted.dataEnd - offsetForView };

    const diff = seg && !seg.is_drift ? computeDiffRegions(offsetForView, refDuration, candDuration) : { removed: null, added: null };

    return { refPeaks, candPeaks, finalPeaks, diff };
  }, [viewStart, viewDuration, refDuration, candDuration, refFullPeaks, candFullPeaks, segments, previewStart]);

  /** Zoom by `factor` (< 1 zooms in, > 1 zooms out), keeping `centerTime` at
   * the same relative position in the view -- so a wheel-zoom stays anchored
   * under the cursor instead of recentering the whole view. */
  function zoomAt(factor: number, centerTime: number) {
    if (viewDuration === null || refDuration === null) return;
    const newDuration = Math.max(MIN_VIEW_DURATION_S, Math.min(refDuration, viewDuration * factor));
    const frac = viewDuration > 0 ? (centerTime - viewStart) / viewDuration : 0.5;
    let newStart = centerTime - frac * newDuration;
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

  function startCursorLoop() {
    function tick() {
      const ctx = audioCtxRef.current;
      const start = playbackStartRef.current;
      const refBuffer = refBufferRef.current;
      if (ctx && start && refBuffer) {
        const elapsed = ctx.currentTime - start.contextTime;
        // elapsed < 0 just means playback is still in its scheduled lead-in
        // (see the `when = ctx.currentTime + 0.05` in loadAndPlay/
        // handleWaveformSeek) -- keep looping without moving the marker yet,
        // don't treat "hasn't started" the same as "finished".
        if (elapsed <= refBuffer.duration) {
          if (elapsed >= 0) setLiveCursor(start.refTime + elapsed);
          rafRef.current = requestAnimationFrame(tick);
          return;
        }
      }
      stopCursorLoop();
    }
    rafRef.current = requestAnimationFrame(tick);
  }

  /** Loads a fresh 12s clip pair starting at `startAt` (defaults to the
   * current position field), decodes them, and schedules all three to start
   * together. Takes an explicit argument rather than always reading
   * `previewStart` so a click-to-seek during playback (see
   * handleWaveformSeek) can jump straight to the clicked time without
   * waiting for the state update to land first. */
  async function loadAndPlay(startAt: number = previewStart) {
    const generation = ++loadGenerationRef.current;
    setLoading(true);
    setError(null);
    try {
      // Unclamped: when negative, the correction is a leading silence (real
      // candidate content hasn't started yet at this reference position) --
      // clamping this to 0 for the *fetch* is correct (there's nothing
      // before candidate-time 0 to fetch), but "Résultat final" must then
      // actually start `leadingSilenceS` later than the other two, or it
      // ends up starting at the exact same instant as "Piste corrigée
      // (originale)".
      const rawCandidateStart = startAt + offsetAt(segments, startAt);
      const candidateStart = Math.max(0, rawCandidateStart);
      const leadingSilenceS = Math.max(0, -rawCandidateStart);
      const [refBlob, candBlob, candOriginalBlob] = await Promise.all([
        fetchClip(filePath, referenceIndex, startAt, PREVIEW_DURATION_S),
        fetchClip(filePath, trackIndex, candidateStart, Math.max(0.1, PREVIEW_DURATION_S - leadingSilenceS)),
        fetchClip(filePath, trackIndex, startAt, PREVIEW_DURATION_S),
      ]);
      if (loadGenerationRef.current !== generation) return; // superseded while fetching

      const ctx = getAudioCtx();
      if (ctx.state === "suspended") await ctx.resume();
      const [refBuffer, candBuffer, candOriginalBuffer] = await Promise.all([
        decodeClip(ctx, refBlob),
        decodeClip(ctx, candBlob),
        decodeClip(ctx, candOriginalBlob),
      ]);
      if (loadGenerationRef.current !== generation) return; // superseded while decoding

      refBufferRef.current = refBuffer;
      candBufferRef.current = candBuffer;
      candOriginalBufferRef.current = candOriginalBuffer;
      setLoadedClipStart(startAt);
      setLoadedCandidateStart(candidateStart);
      setLoading(false);

      const refGain = getGain(refGainRef, ctx, refMuted);
      const candGain = getGain(candGainRef, ctx, candMuted);
      const candOriginalGain = getGain(candOriginalGainRef, ctx, candOriginalMuted);

      stopSource(refSourceRef);
      stopSource(candSourceRef);
      stopSource(candOriginalSourceRef);

      // A small fixed lead time (not "now") so all three .start() calls --
      // themselves not perfectly instantaneous -- still land before the
      // instant they're scheduled for, guaranteeing they're simultaneous
      // rather than racing each other.
      const when = ctx.currentTime + 0.05;
      refSourceRef.current = playSource(ctx, refBuffer, refGain, when, 0);
      candOriginalSourceRef.current = playSource(ctx, candOriginalBuffer, candOriginalGain, when, 0);
      candSourceRef.current = playSource(ctx, candBuffer, candGain, when + leadingSilenceS, 0);

      playbackStartRef.current = { contextTime: when, refTime: startAt };
      startCursorLoop();
    } catch (err) {
      if (loadGenerationRef.current !== generation) return;
      setError(err instanceof Error ? err.message : String(err));
      setLoading(false);
    }
  }

  /** Click-to-seek on any of the three waveforms. If the clicked time is
   * still within the clip currently loaded for playback, reschedule all
   * three from the already-decoded buffers in memory (instant, no network
   * round trip); otherwise, if something was playing, reload a fresh clip
   * starting there so listening continues uninterrupted instead of silently
   * going stale. Either way, the position field (and thus the always-visible
   * marker) follows the click. */
  function handleWaveformSeek(t: number) {
    const rounded = Math.round(t * 10) / 10;
    const ctx = audioCtxRef.current;
    const wasPlaying = playbackStartRef.current !== null && ctx !== null;
    const withinLoadedClip =
      loadedClipStart !== null && t >= loadedClipStart && t <= loadedClipStart + PREVIEW_DURATION_S;

    setPreviewStart(rounded);

    if (
      withinLoadedClip &&
      wasPlaying &&
      ctx &&
      refBufferRef.current &&
      candBufferRef.current &&
      candOriginalBufferRef.current &&
      refGainRef.current &&
      candGainRef.current &&
      candOriginalGainRef.current
    ) {
      const refOffset = t - loadedClipStart!;
      const candOriginalOffset = t - loadedClipStart!; // same native axis as the reference

      // Must be the *clamped* value actually passed to /clip
      // (loadedCandidateStart), not offsetAt(loadedClipStart) recomputed
      // unclamped: when the clip was loaded with a leading silence, the
      // buffer's own offset-0 is candidateStart (clamped), not the
      // unclamped raw position.
      const rawCandTarget = t + offsetAt(segments, t) - (loadedCandidateStart ?? 0);

      const when = ctx.currentTime + 0.02;
      stopSource(refSourceRef);
      stopSource(candOriginalSourceRef);
      stopSource(candSourceRef);

      refSourceRef.current = playSource(ctx, refBufferRef.current, refGainRef.current, when, refOffset);
      candOriginalSourceRef.current = playSource(ctx, candOriginalBufferRef.current, candOriginalGainRef.current, when, candOriginalOffset);
      if (rawCandTarget < 0) {
        // Still within the silent lead-in: schedule the real content to
        // start `-rawCandTarget` seconds after `when`, at buffer-offset 0.
        candSourceRef.current = playSource(ctx, candBufferRef.current, candGainRef.current, when - rawCandTarget, 0);
      } else {
        candSourceRef.current = playSource(ctx, candBufferRef.current, candGainRef.current, when, rawCandTarget);
      }
      playbackStartRef.current = { contextTime: when, refTime: t };
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
    stopSource(refSourceRef);
    stopSource(candSourceRef);
    stopSource(candOriginalSourceRef);
    playbackStartRef.current = null;
    stopCursorLoop();
  }

  const diff = viewData?.diff ?? { removed: null, added: null };
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
        <button className="small-button" onClick={() => zoomAt(0.5, previewStart)}>
          + (zoomer)
        </button>
        <button className="small-button" onClick={() => zoomAt(2, previewStart)}>
          − (dézoomer)
        </button>
        <button className="small-button" onClick={resetZoom}>
          Piste entière
        </button>
        <span className="preview-offset">molette = zoomer/dézoomer sous le curseur</span>
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

      {waveformLoading && !viewData && <p className="placeholder">Chargement des formes d'onde...</p>}

      {viewDuration !== null && viewData && (
        <div className="waveforms">
          <Waveform
            viewStart={viewStart}
            viewDuration={viewDuration}
            peaksMin={viewData.refPeaks.min}
            peaksMax={viewData.refPeaks.max}
            dataStart={viewData.refPeaks.dataStart}
            dataEnd={viewData.refPeaks.dataEnd}
            cursor={displayCursor}
            onSeek={handleWaveformSeek}
            onZoom={zoomAt}
            label="Référence"
            className="waveform-reference"
          />
          <Waveform
            viewStart={viewStart}
            viewDuration={viewDuration}
            peaksMin={viewData.candPeaks.min}
            peaksMax={viewData.candPeaks.max}
            dataStart={viewData.candPeaks.dataStart}
            dataEnd={viewData.candPeaks.dataEnd}
            cursor={displayCursor}
            onSeek={handleWaveformSeek}
            onZoom={zoomAt}
            highlights={removedHighlight}
            label="Piste corrigée (originale)"
            className="waveform-candidate"
          />
          <Waveform
            viewStart={viewStart}
            viewDuration={viewDuration}
            peaksMin={viewData.finalPeaks.min}
            peaksMax={viewData.finalPeaks.max}
            dataStart={viewData.finalPeaks.dataStart}
            dataEnd={viewData.finalPeaks.dataEnd}
            cursor={displayCursor}
            onSeek={handleWaveformSeek}
            onZoom={zoomAt}
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
        <button className="small-button" onClick={stop} disabled={loadedClipStart === null}>
          Arrêter
        </button>
        <label>
          <input
            type="checkbox"
            checked={!refMuted}
            onChange={() =>
              setRefMuted((m) => {
                const next = !m;
                if (refGainRef.current) refGainRef.current.gain.value = next ? 0 : 1;
                return next;
              })
            }
          />
          Référence
        </label>
        <label>
          <input
            type="checkbox"
            checked={!candOriginalMuted}
            onChange={() =>
              setCandOriginalMuted((m) => {
                const next = !m;
                if (candOriginalGainRef.current) candOriginalGainRef.current.gain.value = next ? 0 : 1;
                return next;
              })
            }
          />
          Piste corrigée (originale)
        </label>
        <label>
          <input
            type="checkbox"
            checked={!candMuted}
            onChange={() =>
              setCandMuted((m) => {
                const next = !m;
                if (candGainRef.current) candGainRef.current.gain.value = next ? 0 : 1;
                return next;
              })
            }
          />
          Résultat final
        </label>
        <span className="preview-offset">
          Décalage appliqué : {appliedOffset.toFixed(3)} s (résultat final lu à partir de{" "}
          {Math.max(0, previewStart + appliedOffset).toFixed(3)} s dans la piste corrigée)
        </span>
        {loadedCandidateStart !== null && (
          <span
            className="preview-offset"
            title="Ce que le dernier clic sur Écouter a réellement chargé -- s'il diffère du nombre ci-dessus, l'audio est resté sur un ancien calcul (reclique sur Écouter)."
          >
            (dernier chargement réel : {loadedCandidateStart.toFixed(3)} s)
          </span>
        )}
        {hasContainerDelay && (
          <span className="preview-offset" title="Le décalage ci-dessus (utilisé pour la lecture et l'export) est mesuré sur la piste brute, sans son délai de conteneur -- ce nombre est juste informatif.">
            (dont {trackStartTime.toFixed(3)} s déjà présents dans le conteneur pour cette piste ; décalage restant dans un lecteur ≈ {presentationOffset.toFixed(3)} s)
          </span>
        )}
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}
