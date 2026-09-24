import { useEffect, useRef, useState } from "react";
import type { SegmentOut } from "./api";
import { formatTime } from "./SegmentChart";
import "./SegmentEditor.css";

const WIDTH = 900;
const HEIGHT = 340;
const MARGIN = { top: 20, right: 20, bottom: 32, left: 64 };
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom;
// Must match engine/src/syncaudio/segments.py's _DRIFT_EPS_S: the editor
// recomputes is_drift live as the user edits offset values (rather than
// trusting the segments' original is_drift, which goes stale the moment
// any value changes), and a mismatched threshold here would relabel a
// segment the backend classified as constant the moment it's opened for
// editing -- exactly the bug this comment replaced.
const DRIFT_EPS_S = 0.2;

/** The editable form of a segment list: N+1 boundary times (shared between
 * consecutive segments, so dragging or typing one can never open a gap or
 * an overlap) plus each segment's own two offset values. */
interface EditorState {
  times: number[];
  offsetStarts: number[];
  offsetEnds: number[];
}

function toEditorState(segments: SegmentOut[]): EditorState {
  return {
    times: [segments[0].start_s, ...segments.map((s) => s.end_s)],
    offsetStarts: segments.map((s) => s.offset_start),
    offsetEnds: segments.map((s) => s.offset_end),
  };
}

function toSegments(state: EditorState): SegmentOut[] {
  return state.offsetStarts.map((offset_start, i) => {
    const offset_end = state.offsetEnds[i];
    return {
      start_s: state.times[i],
      end_s: state.times[i + 1],
      offset_start,
      offset_end,
      is_drift: Math.abs(offset_end - offset_start) > DRIFT_EPS_S,
    };
  });
}

/** Merge segment `i` into segment `i + 1`'s slot (or the previous one, if
 * `i` is the last segment) -- used for both "merge with neighbour" and
 * "delete" (deleting is just merging into whichever neighbour exists). */
function mergeSegment(state: EditorState, i: number): EditorState {
  const mergeWithNext = i < state.offsetStarts.length - 1;
  const j = mergeWithNext ? i + 1 : i - 1;
  const lo = Math.min(i, j);
  const hi = Math.max(i, j);
  return {
    times: [...state.times.slice(0, lo + 1), ...state.times.slice(hi + 1)],
    offsetStarts: [...state.offsetStarts.slice(0, lo), state.offsetStarts[lo], ...state.offsetStarts.slice(hi + 1)],
    offsetEnds: [...state.offsetEnds.slice(0, lo), state.offsetEnds[hi], ...state.offsetEnds.slice(hi + 1)],
  };
}

export function SegmentEditor({
  segments,
  onSave,
  onClose,
}: {
  segments: SegmentOut[];
  onSave: (segments: SegmentOut[]) => void;
  onClose: () => void;
}) {
  const [state, setState] = useState<EditorState>(() => toEditorState(segments));
  const [dragging, setDragging] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const totalDuration = state.times[state.times.length - 1];
  const offsetsFlat = [...state.offsetStarts, ...state.offsetEnds];
  let minOffset = Math.min(0, ...offsetsFlat);
  let maxOffset = Math.max(0, ...offsetsFlat);
  if (minOffset === maxOffset) {
    minOffset -= 1;
    maxOffset += 1;
  }
  const pad = (maxOffset - minOffset) * 0.15;
  minOffset -= pad;
  maxOffset += pad;

  const x = (t: number) => (totalDuration > 0 ? (t / totalDuration) * PLOT_W : 0);
  const xInv = (px: number) => (totalDuration > 0 ? (px / PLOT_W) * totalDuration : 0);
  const y = (offset: number) => PLOT_H - ((offset - minOffset) / (maxOffset - minOffset)) * PLOT_H;

  function timeFromClientX(clientX: number): number {
    const rect = svgRef.current!.getBoundingClientRect();
    const svgX = ((clientX - rect.left) / rect.width) * WIDTH;
    return xInv(svgX - MARGIN.left);
  }

  useEffect(() => {
    if (dragging === null) return;
    const minBound = state.times[dragging - 1] + 0.1;
    const maxBound = state.times[dragging + 1] - 0.1;

    function onMove(ev: PointerEvent) {
      const t = Math.min(maxBound, Math.max(minBound, timeFromClientX(ev.clientX)));
      setState((s) => {
        const times = [...s.times];
        times[dragging!] = t;
        return { ...s, times };
      });
    }
    function onUp() {
      setDragging(null);
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dragging]);

  function updateTime(i: number, value: number) {
    setState((s) => {
      const lo = s.times[i - 1] ?? -Infinity;
      const hi = s.times[i + 1] ?? Infinity;
      if (value <= lo || value >= hi) return s;
      const times = [...s.times];
      times[i] = value;
      return { ...s, times };
    });
  }

  function updateOffset(kind: "start" | "end", i: number, value: number) {
    setState((s) => {
      const key = kind === "start" ? "offsetStarts" : "offsetEnds";
      const arr = [...s[key]];
      arr[i] = value;
      return { ...s, [key]: arr };
    });
  }

  const segmentsPreview = toSegments(state);

  return (
    <div className="editor-overlay" role="dialog" aria-modal="true">
      <div className="editor-panel">
        <div className="editor-header">
          <h2>Corriger manuellement les segments</h2>
          <button className="small-button" onClick={onClose}>
            Annuler
          </button>
        </div>

        <svg ref={svgRef} className="editor-chart" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img">
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            <line x1={0} y1={y(0)} x2={PLOT_W} y2={y(0)} className="zero-line" />
            <text x={-8} y={y(0)} className="axis-label" textAnchor="end" dominantBaseline="middle">
              0s
            </text>

            {Array.from({ length: 6 }, (_, i) => (totalDuration * i) / 5).map((t) => (
              <g key={t}>
                <line x1={x(t)} y1={0} x2={x(t)} y2={PLOT_H} className="grid-line" />
                <text x={x(t)} y={PLOT_H + 18} className="axis-label" textAnchor="middle">
                  {formatTime(t)}
                </text>
              </g>
            ))}

            {segmentsPreview.map((seg, i) => {
              const flat = (seg.offset_start + seg.offset_end) / 2;
              const yStart = seg.is_drift ? y(seg.offset_start) : y(flat);
              const yEnd = seg.is_drift ? y(seg.offset_end) : y(flat);
              return (
                <g key={i}>
                  <line
                    x1={x(seg.start_s)}
                    y1={yStart}
                    x2={x(seg.end_s)}
                    y2={yEnd}
                    className={seg.is_drift ? "segment-line drift" : "segment-line constant"}
                  />
                  <text
                    x={(x(seg.start_s) + x(seg.end_s)) / 2}
                    y={(yStart + yEnd) / 2 - 10}
                    className="segment-label"
                    textAnchor="middle"
                  >
                    {seg.is_drift ? `${seg.offset_start.toFixed(2)}s → ${seg.offset_end.toFixed(2)}s` : `${flat.toFixed(2)}s`}
                  </text>
                </g>
              );
            })}

            {/* Draggable handles on every *internal* boundary only -- the
                first (0) and last (total duration) are fixed. */}
            {state.times.slice(1, -1).map((t, idx) => {
              const i = idx + 1;
              return (
                <g key={i} className="boundary-handle" onPointerDown={() => setDragging(i)}>
                  <line x1={x(t)} y1={-4} x2={x(t)} y2={PLOT_H + 4} className="boundary-line" />
                  <circle cx={x(t)} cy={-4} r={7} />
                </g>
              );
            })}
          </g>
        </svg>

        <div className="editor-table-wrap">
          <table className="editor-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Début (s)</th>
                <th>Fin (s)</th>
                <th>Décalage début (s)</th>
                <th>Décalage fin (s)</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {segmentsPreview.map((seg, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>
                    <input
                      type="number"
                      step="0.01"
                      value={seg.start_s.toFixed(2)}
                      disabled={i === 0}
                      onChange={(e) => updateTime(i, Number(e.target.value))}
                    />
                  </td>
                  <td>
                    <input
                      type="number"
                      step="0.01"
                      value={seg.end_s.toFixed(2)}
                      disabled={i === segmentsPreview.length - 1}
                      onChange={(e) => updateTime(i + 1, Number(e.target.value))}
                    />
                  </td>
                  <td>
                    <input
                      type="number"
                      step="0.01"
                      value={seg.offset_start.toFixed(2)}
                      onChange={(e) => updateOffset("start", i, Number(e.target.value))}
                    />
                  </td>
                  <td>
                    <input
                      type="number"
                      step="0.01"
                      value={seg.offset_end.toFixed(2)}
                      onChange={(e) => updateOffset("end", i, Number(e.target.value))}
                    />
                  </td>
                  <td>
                    <button
                      className="small-button"
                      disabled={segmentsPreview.length < 2}
                      title="Fusionne ce segment avec le suivant (ou le précédent si c'est le dernier) -- utile pour retirer un segment parasite."
                      onClick={() => setState((s) => mergeSegment(s, i))}
                    >
                      {i < segmentsPreview.length - 1 ? "Fusionner ↓" : "Fusionner ↑"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="editor-actions">
          <button
            className="primary-button"
            onClick={() => {
              onSave(segmentsPreview);
              onClose();
            }}
          >
            Enregistrer
          </button>
          <button className="small-button" onClick={onClose}>
            Annuler
          </button>
        </div>
      </div>
    </div>
  );
}
