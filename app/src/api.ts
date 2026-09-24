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
}

export interface ProbeResponse {
  path: string;
  tracks: TrackInfo[];
}

export interface AlignResult {
  track: string;
  language: string | null;
  offset_seconds: number;
  confidence: number;
  ambiguous: boolean;
}

export interface AlignResponse {
  reference: string;
  results: AlignResult[];
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

export async function startAlignJob(
  referencePath: string,
  referenceIndex: number,
  candidatePath: string,
  candidateIndex: number,
): Promise<string> {
  const resp = await fetch(`${BASE_URL}/jobs/align`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      reference: { path: referencePath, index: referenceIndex },
      candidates: [{ path: candidatePath, index: candidateIndex }],
    }),
  });
  if (!resp.ok) throw new Error(await readErrorDetail(resp));
  const data = await resp.json();
  return data.job_id as string;
}

export type JobEvent =
  | { type: "log"; message: string }
  | { type: "done"; result: AlignResponse }
  | { type: "error"; message: string };

/** Connects to a job's progress WebSocket; returns a function to close it early. */
export function connectJobWS(jobId: string, onEvent: (event: JobEvent) => void): () => void {
  const ws = new WebSocket(`ws://127.0.0.1:8756/jobs/${jobId}/ws`);
  ws.onmessage = (ev) => onEvent(JSON.parse(ev.data));
  ws.onerror = () => onEvent({ type: "error", message: "Connexion WebSocket perdue." });
  return () => ws.close();
}
