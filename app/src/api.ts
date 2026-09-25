// Client for the syncaudio FastAPI sidecar (engine/src/syncaudio/server.py).
// Dev-time only: the sidecar is spawned by src-tauri/src/lib.rs via `uv run`.
// Phase 6 packaging will need this base URL/port to stay in sync with
// whatever bundled sidecar binary replaces that dev-time spawn.
const BASE_URL = "http://127.0.0.1:8756";

export interface TrackInfo {
  index: number;
  codec: string | null;
  language: string | null;
  channels: number | null;
  sample_rate: number | null;
  // Container-level presentation delay (e.g. from mkvtoolnix's --sync), if
  // any -- display-only, see the engine's probe_stream_start_time docstring
  // for why detection/render intentionally ignore it.
  start_time: number;
}

export interface ProbeResponse {
  path: string;
  tracks: TrackInfo[];
}

export interface SegmentOut {
  start_s: number;
  end_s: number;
  offset_start: number;
  offset_end: number;
  is_drift: boolean;
}

export interface SegmentsResponse {
  reference: string;
  track: string;
  segments: SegmentOut[];
}

export interface RenderedTrack {
  track: string;
  language: string | null;
  offset_seconds: number | null; // null for a segmented (non-constant) correction
  segments: SegmentOut[] | null;
}

export interface RenderResponse {
  written: string[];
  corrections: RenderedTrack[];
}

async function readErrorDetail(resp: Response): Promise<string> {
  try {
    const body = await resp.json();
    return body.detail ?? `Erreur ${resp.status}`;
  } catch {
    return `Erreur ${resp.status}`;
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const resp = await fetch(`${BASE_URL}/health`);
    return resp.ok;
  } catch {
    return false;
  }
}

export async function probe(path: string): Promise<ProbeResponse> {
  const resp = await fetch(`${BASE_URL}/probe?path=${encodeURIComponent(path)}`);
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  return resp.json();
}

export interface PrefetchResponse {
  cached: number;
}

/**
 * Warms the engine's analysis cache for every track in the background --
 * fire right after `probe` succeeds so that by the time the user picks a
 * reference and clicks a detection button, the ~7s-per-track
 * extraction+envelope cost (the actual bottleneck, not ffmpeg decoding) is
 * already paid. Fire-and-forget: a failure here just means the next
 * detection redoes the work itself, so callers aren't required to await
 * the job's completion or handle its errors specially.
 */
export async function startPrefetchJob(path: string, trackIndices: number[]): Promise<string> {
  const resp = await fetch(`${BASE_URL}/jobs/prefetch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tracks: trackIndices.map((index) => ({ path, index })) }),
  });
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  const data = await resp.json();
  return data.job_id as string;
}

/**
 * A short playable WAV clip of one track, for the "listen before you
 * render" preview -- not the 16kHz analysis PCM, a normal-rate clip meant
 * to actually be played back in an <audio> element.
 */
export async function fetchClip(path: string, index: number, start: number, duration: number): Promise<Blob> {
  const params = new URLSearchParams({ path, index: String(index), start: String(start), duration: String(duration) });
  // no-store: this is re-fetched with a genuinely different `start` every
  // time the user seeks, and must never come back stale from the browser's
  // HTTP cache (the sidecar's plain Response doesn't set any cache headers
  // of its own to prevent that).
  const resp = await fetch(`${BASE_URL}/clip?${params.toString()}`, { cache: "no-store" });
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  return resp.blob();
}

export interface WaveformResponse {
  duration: number;
  peaks_min: number[];
  peaks_max: number[];
}

/**
 * A downsampled (min, max) amplitude envelope for a track window -- never
 * ships raw audio, so it stays cheap even for a whole multi-minute track at
 * once (`duration` omitted), unlike `fetchClip`. Used to draw the
 * always-visible, zoomable comparison waveforms (see TrackPreview.tsx).
 */
export async function fetchWaveform(
  path: string,
  index: number,
  start: number,
  duration: number | null,
  buckets: number,
): Promise<WaveformResponse> {
  const params = new URLSearchParams({ path, index: String(index), start: String(start), buckets: String(buckets) });
  if (duration !== null) params.set("duration", String(duration));
  const resp = await fetch(`${BASE_URL}/waveform?${params.toString()}`);
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  return resp.json();
}

export async function startSegmentsJob(
  referencePath: string,
  referenceIndex: number,
  trackPath: string,
  trackIndex: number,
  options?: { windowS?: number; hopS?: number; marginS?: number },
): Promise<string> {
  const resp = await fetch(`${BASE_URL}/jobs/segments`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      reference: { path: referencePath, index: referenceIndex },
      track: { path: trackPath, index: trackIndex },
      ...(options?.windowS !== undefined ? { window_s: options.windowS } : {}),
      ...(options?.hopS !== undefined ? { hop_s: options.hopS } : {}),
      ...(options?.marginS !== undefined ? { margin_s: options.marginS } : {}),
    }),
  });
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  const data = await resp.json();
  return data.job_id as string;
}

/**
 * Segmented (drift/jump-aware) render of exactly one track, using `segments`
 * as-is instead of letting the server re-run detection -- so a render after
 * "Analyser" + manual edits in SegmentEditor produces what was actually
 * reviewed, not a silently recomputed result that discards the edits.
 */
export async function startSegmentedRenderJob(
  inputPath: string,
  referenceIndex: number,
  trackIndex: number,
  segments: SegmentOut[],
): Promise<string> {
  const resp = await fetch(`${BASE_URL}/jobs/render`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input_path: inputPath,
      reference_index: referenceIndex,
      track_indices: [trackIndex],
      segmented: true,
      segment_overrides: [{ track: { path: inputPath, index: trackIndex }, segments }],
    }),
  });
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  const data = await resp.json();
  return data.job_id as string;
}

export type JobEvent<TResult> =
  | { type: "log"; message: string }
  | { type: "done"; result: TResult }
  | { type: "error"; message: string };

/**
 * Connects to a job's progress WebSocket; returns a function to close it early.
 *
 * Guards against the sidecar dying (or any other reason the socket just
 * closes) mid-job without ever sending a "done"/"error" event -- without
 * this, the caller's UI would stay stuck in "in progress" forever with no
 * way to know something went wrong.
 */
export function connectJobWS<TResult>(jobId: string, onEvent: (event: JobEvent<TResult>) => void): () => void {
  const ws = new WebSocket(`ws://127.0.0.1:8756/jobs/${jobId}/ws`);
  let settled = false;

  ws.onmessage = (ev) => {
    const event: JobEvent<TResult> = JSON.parse(ev.data);
    if (event.type === "done" || event.type === "error") settled = true;
    onEvent(event);
  };
  ws.onerror = () => {
    if (!settled) {
      settled = true;
      onEvent({ type: "error", message: "Connexion WebSocket perdue." });
    }
  };
  ws.onclose = () => {
    if (!settled) {
      settled = true;
      onEvent({ type: "error", message: "Connexion interrompue avant la fin du traitement (le moteur a-t-il planté ?)." });
    }
  };
  return () => ws.close();
}
