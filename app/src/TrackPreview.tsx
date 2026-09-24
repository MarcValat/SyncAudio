import { useEffect, useRef, useState } from "react";
import { fetchClip, type SegmentOut } from "./api";
import { formatTime } from "./SegmentChart";
import { Waveform } from "./Waveform";

const PREVIEW_DURATION_S = 12;

/**
 * The reference-time offset applying at `t`, per the current segments
 * (linearly interpolated across a drift segment, clamped to the first/last
 * segment's edge offset outside the analyzed range). Used to pick where in
 * the candidate track to start its preview clip so it lines up with the
 * reference clip at the same wall-clock moment.
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

/** Decodes a WAV blob to mono Float32 samples -- purely for drawing the
 * waveform; actual playback still goes through the <audio> elements below,
 * decodeAudioData is only used as a convenient in-browser PCM decoder. */
async function decodeSamples(ctx: AudioContext, blob: Blob): Promise<{ samples: Float32Array; duration: number }> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
  return { samples: audioBuffer.getChannelData(0), duration: audioBuffer.duration };
}

interface TrackPreviewProps {
  filePath: string;
  referenceIndex: number;
  trackIndex: number;
  segments: SegmentOut[];
}

/** Listen to the reference and a candidate track together, pre-aligned per
 * the current segments, to check by ear (and now by eye, via waveforms)
 * whether a correction sounds right before actually rendering it. */
export function TrackPreview({ filePath, referenceIndex, trackIndex, segments }: TrackPreviewProps) {
  const [previewStart, setPreviewStart] = useState(() => (segments.length ? segments[0].start_s : 0));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refUrl, setRefUrl] = useState<string | null>(null);
  const [candUrl, setCandUrl] = useState<string | null>(null);
  const [refSamples, setRefSamples] = useState<Float32Array | null>(null);
  const [candSamples, setCandSamples] = useState<Float32Array | null>(null);
  const [clipDuration, setClipDuration] = useState(0);
  const [refMuted, setRefMuted] = useState(false);
  const [candMuted, setCandMuted] = useState(false);
  const [cursor, setCursor] = useState<number | null>(null);

  const refAudioRef = useRef<HTMLAudioElement>(null);
  const candAudioRef = useRef<HTMLAudioElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);

  // Shown to the user (and useful for debugging): the actual shift being
  // applied to the candidate clip's read position for this preview.
  const appliedOffset = offsetAt(segments, previewStart);

  // Object URLs are per-viewer browser resources, not tracked by React --
  // revoke the previous pair whenever a new one replaces it, and on unmount.
  useEffect(() => {
    return () => {
      if (refUrl) URL.revokeObjectURL(refUrl);
      if (candUrl) URL.revokeObjectURL(candUrl);
    };
  }, [refUrl, candUrl]);

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      audioCtxRef.current?.close().catch(() => {});
    };
  }, []);

  function getAudioCtx(): AudioContext {
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext();
    return audioCtxRef.current;
  }

  function stopCursorLoop() {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setCursor(null);
  }

  function startCursorLoop() {
    function tick() {
      const refEl = refAudioRef.current;
      if (refEl && !refEl.paused && !refEl.ended) {
        setCursor(refEl.currentTime);
        rafRef.current = requestAnimationFrame(tick);
      } else {
        stopCursorLoop();
      }
    }
    rafRef.current = requestAnimationFrame(tick);
  }

  async function loadAndPlay() {
    setLoading(true);
    setError(null);
    try {
      const candidateStart = Math.max(0, previewStart + offsetAt(segments, previewStart));
      const [refBlob, candBlob] = await Promise.all([
        fetchClip(filePath, referenceIndex, previewStart, PREVIEW_DURATION_S),
        fetchClip(filePath, trackIndex, candidateStart, PREVIEW_DURATION_S),
      ]);
      setRefUrl(URL.createObjectURL(refBlob));
      setCandUrl(URL.createObjectURL(candBlob));

      const ctx = getAudioCtx();
      const [refDecoded, candDecoded] = await Promise.all([decodeSamples(ctx, refBlob), decodeSamples(ctx, candBlob)]);
      setRefSamples(refDecoded.samples);
      setCandSamples(candDecoded.samples);
      setClipDuration(Math.min(refDecoded.duration, candDecoded.duration));
      setLoading(false);

      // Give the <audio> elements a tick to pick up their new src before
      // starting both together.
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

  /** Jump the preview position to the middle of `seg` -- avoids the user
   * having to eyeball the chart to land inside the right segment, since a
   * position right next to a boundary can silently fall on the wrong side. */
  function goToSegment(seg: SegmentOut) {
    setPreviewStart(Math.round(((seg.start_s + seg.end_s) / 2) * 10) / 10);
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

  /** Seek both clips together (they're pre-aligned, so the same in-clip time
   * is the same real moment for each) -- from a click on either waveform. */
  function seekTo(time: number) {
    if (refAudioRef.current) refAudioRef.current.currentTime = time;
    if (candAudioRef.current) candAudioRef.current.currentTime = time;
    setCursor(time);
  }

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
        <button className="small-button" onClick={loadAndPlay} disabled={loading}>
          {loading ? "Chargement..." : "Écouter"}
        </button>
        <button className="small-button" onClick={stop} disabled={!refUrl && !candUrl}>
          Arrêter
        </button>
        <label>
          <input
            type="checkbox"
            checked={!refMuted}
            onChange={() => setRefMuted((m) => !m)}
          />
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
      {(refSamples || candSamples) && (
        <div className="waveforms">
          <Waveform
            samples={refSamples}
            duration={clipDuration}
            cursor={cursor}
            onSeek={seekTo}
            label="Référence"
            className="waveform-reference"
          />
          <Waveform
            samples={candSamples}
            duration={clipDuration}
            cursor={cursor}
            onSeek={seekTo}
            label="Piste corrigée"
            className="waveform-candidate"
          />
        </div>
      )}
      <audio
        ref={refAudioRef}
        src={refUrl ?? undefined}
        muted={refMuted}
        onPlay={startCursorLoop}
        onPause={stopCursorLoop}
        onEnded={stopCursorLoop}
      />
      <audio ref={candAudioRef} src={candUrl ?? undefined} muted={candMuted} />
    </div>
  );
}
