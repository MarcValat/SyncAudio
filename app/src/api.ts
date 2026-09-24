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
  candidateIndices: number[],
): Promise<string> {
  const resp = await fetch(`${BASE_URL}/jobs/align`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      reference: { path: referencePath, index: referenceIndex },
      candidates: candidateIndices.map((index) => ({ path: candidatePath, index })),
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

/**
 * Connects to a job's progress WebSocket; returns a function to close it early.
 *
 * Guards against the sidecar dying (or any other reason the socket just
 * closes) mid-job without ever sending a "done"/"error" event -- without
 * this, the caller's UI would stay stuck in "in progress" forever with no
 * way to know something went wrong.
 */
export function connectJobWS(jobId: string, onEvent: (event: JobEvent) => void): () => void {
  const ws = new WebSocket(`ws://127.0.0.1:8756/jobs/${jobId}/ws`);
  let settled = false;

  ws.onmessage = (ev) => {
    const event: JobEvent = JSON.parse(ev.data);
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
