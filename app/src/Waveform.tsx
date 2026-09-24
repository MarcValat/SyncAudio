import { useMemo, useRef } from "react";
import "./Waveform.css";

const WIDTH = 860;
const HEIGHT = 90;
const BUCKETS = 400;

/** Per-pixel-column min/max amplitude -- the standard way to draw a waveform
 * without plotting every individual sample (a 12s clip at 44.1kHz is
 * ~530k samples, far more than any useful pixel resolution). */
function computePeaks(samples: Float32Array, buckets: number): { min: Float32Array; max: Float32Array } {
  const min = new Float32Array(buckets);
  const max = new Float32Array(buckets);
  const bucketSize = samples.length / buckets;
  for (let i = 0; i < buckets; i++) {
    const start = Math.floor(i * bucketSize);
    const end = Math.floor((i + 1) * bucketSize);
    let mn = 0;
    let mx = 0;
    for (let j = start; j < end && j < samples.length; j++) {
      const v = samples[j];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    min[i] = mn;
    max[i] = mx;
  }
  return { min, max };
}

function peaksToPath(min: Float32Array, max: Float32Array, width: number, height: number): string {
  const n = min.length;
  if (n === 0) return "";
  const xStep = width / n;
  const mid = height / 2;
  const top: string[] = [];
  const bottom: string[] = [];
  for (let i = 0; i < n; i++) {
    const x = ((i + 0.5) * xStep).toFixed(1);
    top.push(`${x},${(mid - max[i] * mid).toFixed(1)}`);
    bottom.push(`${x},${(mid - min[i] * mid).toFixed(1)}`);
  }
  bottom.reverse();
  return `M ${top.join(" L ")} L ${bottom.join(" L ")} Z`;
}

interface WaveformProps {
  samples: Float32Array | null;
  duration: number;
  /** Playhead position in seconds within [0, duration], or null when not playing. */
  cursor: number | null;
  onSeek: (time: number) => void;
  label: string;
  className?: string;
}

/** A hand-rolled peak waveform (no charting library), click-to-seek, with an
 * optional moving playhead line during playback. */
export function Waveform({ samples, duration, cursor, onSeek, label, className }: WaveformProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  const path = useMemo(() => {
    if (!samples) return "";
    const { min, max } = computePeaks(samples, BUCKETS);
    return peaksToPath(min, max, WIDTH, HEIGHT);
  }, [samples]);

  function handleClick(e: React.MouseEvent<SVGSVGElement>) {
    if (!svgRef.current || duration <= 0) return;
    const rect = svgRef.current.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    onSeek(Math.max(0, Math.min(duration, frac * duration)));
  }

  const cursorX = cursor !== null && duration > 0 ? (cursor / duration) * WIDTH : null;

  return (
    <div className={`waveform ${className ?? ""}`}>
      <div className="waveform-label">{label}</div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="waveform-svg"
        onClick={handleClick}
        role="img"
        aria-label={`Forme d'onde -- ${label}`}
      >
        <line x1={0} y1={HEIGHT / 2} x2={WIDTH} y2={HEIGHT / 2} className="waveform-zero" />
        {samples ? <path d={path} className="waveform-path" /> : (
          <text x={WIDTH / 2} y={HEIGHT / 2} textAnchor="middle" dominantBaseline="middle" className="waveform-empty">
            (charge...)
          </text>
        )}
        {cursorX !== null && <line x1={cursorX} y1={0} x2={cursorX} y2={HEIGHT} className="waveform-cursor" />}
      </svg>
    </div>
  );
}
