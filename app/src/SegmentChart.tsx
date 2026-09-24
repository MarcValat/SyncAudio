import type { SegmentOut } from "./api";
import "./SegmentChart.css";

const WIDTH = 760;
const HEIGHT = 220;
const MARGIN = { top: 16, right: 16, bottom: 28, left: 56 };
const PLOT_W = WIDTH - MARGIN.left - MARGIN.right;
const PLOT_H = HEIGHT - MARGIN.top - MARGIN.bottom;

function formatTime(seconds: number): string {
  const s = Math.round(seconds);
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${rem.toString().padStart(2, "0")}`;
}

/**
 * A hand-rolled SVG offset-vs-time chart -- no charting library needed for
 * something this simple, and it keeps the GUI's dependency footprint small.
 * Each segment is drawn as a line from (start_s, offset_start) to (end_s,
 * offset_end): flat for a constant-offset segment, sloped for drift -- the
 * same geometry `render.segment_correction_filter` uses to compute the
 * correction, made visible.
 */
export function SegmentChart({ segments }: { segments: SegmentOut[] }) {
  if (segments.length === 0) return null;

  const totalDuration = segments[segments.length - 1].end_s;
  const offsets = segments.flatMap((s) => [s.offset_start, s.offset_end]);
  let minOffset = Math.min(0, ...offsets);
  let maxOffset = Math.max(0, ...offsets);
  if (minOffset === maxOffset) {
    minOffset -= 1;
    maxOffset += 1;
  }
  const pad = (maxOffset - minOffset) * 0.15;
  minOffset -= pad;
  maxOffset += pad;

  const x = (t: number) => (totalDuration > 0 ? (t / totalDuration) * PLOT_W : 0);
  const y = (offset: number) => PLOT_H - ((offset - minOffset) / (maxOffset - minOffset)) * PLOT_H;

  const timeTicks = 5;
  const timeTickValues = Array.from({ length: timeTicks + 1 }, (_, i) => (totalDuration * i) / timeTicks);

  return (
    <div className="segment-chart">
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="Décalage en fonction du temps">
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* zero-offset reference line */}
          <line x1={0} y1={y(0)} x2={PLOT_W} y2={y(0)} className="zero-line" />
          <text x={-8} y={y(0)} className="axis-label" textAnchor="end" dominantBaseline="middle">
            0s
          </text>

          {/* segment boundaries + time axis */}
          {timeTickValues.map((t) => (
            <g key={t}>
              <line x1={x(t)} y1={0} x2={x(t)} y2={PLOT_H} className="grid-line" />
              <text x={x(t)} y={PLOT_H + 16} className="axis-label" textAnchor="middle">
                {formatTime(t)}
              </text>
            </g>
          ))}

          {/* segments -- a "constant" segment is drawn perfectly flat (at the
              mean of its start/end offset) rather than connecting the two
              raw values: they're classified constant because they're close
              enough, not because they're bit-for-bit equal, and a visible
              tilt on a segment labelled "constant" reads as a rendering bug
              rather than the measurement noise it actually is. Drift
              segments keep their real slope -- that's the whole point. */}
          {segments.map((seg, i) => {
            const flatOffset = (seg.offset_start + seg.offset_end) / 2;
            const yStart = seg.is_drift ? y(seg.offset_start) : y(flatOffset);
            const yEnd = seg.is_drift ? y(seg.offset_end) : y(flatOffset);
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
                  y={(yStart + yEnd) / 2 - 8}
                  className="segment-label"
                  textAnchor="middle"
                >
                  {seg.is_drift ? `${seg.offset_start.toFixed(2)}s → ${seg.offset_end.toFixed(2)}s` : `${flatOffset.toFixed(2)}s`}
                </text>
              </g>
            );
          })}
        </g>
      </svg>
      <div className="segment-chart-legend">
        <span className="legend-item">
          <span className="legend-swatch constant" /> constant
        </span>
        <span className="legend-item">
          <span className="legend-swatch drift" /> dérive
        </span>
      </div>
    </div>
  );
}
