import { useRef } from "react";
import { formatTime } from "./SegmentChart";
import "./Waveform.css";

const WIDTH = 860;
const HEIGHT = 56;

export interface HighlightRegion {
  start: number;
  end: number;
  kind: "removed" | "added";
}

interface WaveformProps {
  viewStart: number;
  viewDuration: number;
  peaksMin: number[] | null;
  peaksMax: number[] | null;
  /** The [dataStart, dataEnd] sub-range (same axis as viewStart) that
   * peaksMin/peaksMax actually cover -- can be narrower than the full view
   * when part of it falls outside the track's real bounds (e.g. the padded
   * side of an offset-shifted view). Outside this range, nothing is drawn. */
  dataStart: number;
  dataEnd: number;
  cursor: number | null;
  onSeek: (time: number) => void;
  highlights?: HighlightRegion[];
  label: string;
  className?: string;
}

function peaksToPath(min: number[], max: number[], x0: number, x1: number, height: number): string {
  const n = min.length;
  if (n === 0 || x1 <= x0) return "";
  const xStep = (x1 - x0) / n;
  const mid = height / 2;
  const top: string[] = [];
  const bottom: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = (x0 + (i + 0.5) * xStep).toFixed(1);
    top.push(`${x},${(mid - max[i] * mid).toFixed(1)}`);
    bottom.push(`${x},${(mid - min[i] * mid).toFixed(1)}`);
  }
  bottom.reverse();
  return `M ${top.join(" L ")} L ${bottom.join(" L ")} Z`;
}

/** A hand-rolled peak waveform (no charting library) over a caller-controlled
 * [viewStart, viewStart+viewDuration] window -- click-to-seek, an optional
 * moving playhead, and optional red/green diff-style highlight regions. */
export function Waveform({
  viewStart,
  viewDuration,
  peaksMin,
  peaksMax,
  dataStart,
  dataEnd,
  cursor,
  onSeek,
  highlights,
  label,
  className,
}: WaveformProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const timeToX = (t: number) => (viewDuration > 0 ? ((t - viewStart) / viewDuration) * WIDTH : 0);

  function handleClick(e: React.MouseEvent<SVGSVGElement>) {
    if (!svgRef.current || viewDuration <= 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    onSeek(viewStart + Math.max(0, Math.min(1, frac)) * viewDuration);
  }

  const dataX0 = Math.max(0, timeToX(dataStart));
  const dataX1 = Math.min(WIDTH, timeToX(dataEnd));
  const path = peaksMin && peaksMax ? peaksToPath(peaksMin, peaksMax, dataX0, dataX1, HEIGHT) : "";

  const cursorX = cursor !== null ? timeToX(cursor) : null;
  const showCursor = cursorX !== null && cursorX >= 0 && cursorX <= WIDTH;

  const timeTicks = 5;
  const tickValues = Array.from({ length: timeTicks + 1 }, (_, i) => viewStart + (viewDuration * i) / timeTicks);

  return (
    <div className={`waveform ${className ?? ""}`}>
      <div className="waveform-label">{label}</div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        className="waveform-svg"
        onClick={handleClick}
        role="img"
        aria-label={`Forme d'onde -- ${label}`}
      >
        <line x1={0} y1={HEIGHT / 2} x2={WIDTH} y2={HEIGHT / 2} className="waveform-zero" />
        {highlights?.map((h, i) => {
          const x0 = Math.max(0, timeToX(h.start));
          const x1 = Math.min(WIDTH, timeToX(h.end));
          if (x1 <= x0) return null;
          return <rect key={i} x={x0} y={0} width={x1 - x0} height={HEIGHT} className={`waveform-highlight ${h.kind}`} />;
        })}
        {path && <path d={path} className="waveform-path" />}
        {showCursor && <line x1={cursorX} y1={0} x2={cursorX} y2={HEIGHT} className="waveform-cursor" />}
      </svg>
      <div className="waveform-ticks">
        {tickValues.map((t, i) => (
          <span key={i}>{formatTime(t)}</span>
        ))}
      </div>
    </div>
  );
}
