import { useCallback, useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  checkHealth,
  probe,
  startSegmentsJob,
  startPrefetchJob,
  startSegmentedRenderJob,
  connectJobWS,
  type TrackInfo,
  type SegmentsResponse,
  type PrefetchResponse,
  type RenderResponse,
} from "./api";
import { SegmentChart } from "./SegmentChart";
import { LogPanel } from "./LogPanel";
import { SegmentEditor } from "./SegmentEditor";
import { TrackPreview } from "./TrackPreview";
import { basename } from "./paths";
import "./App.css";

type EngineStatus = "starting" | "ready" | "unreachable";

const HEALTH_POLL_ATTEMPTS = 40; // 40 * 500ms = 20s before giving up

/**
 * Per-track analysis + export state, keyed by track index. There used to be
 * a separate "quick" flat-offset flow (align) alongside this one, but once
 * the analysis cache made both equally fast, the flat flow was strictly
 * weaker (no drift/jump detection, and its "confidence" value was already
 * known to be unreliable) except for launching several tracks at once --
 * so that's folded in here instead: checking several boxes below fires one
 * of these per track.
 */
interface TrackAnalysis {
  status: "running" | "done" | "error";
  referenceIndex: number;
  log: string[];
  result: SegmentsResponse | null;
  error: string | null;
  rendering: boolean;
  renderLog: string[];
  renderResult: RenderResponse | null;
  renderError: string | null;
}

function App() {
  const [engineStatus, setEngineStatus] = useState<EngineStatus>("starting");
  const [filePath, setFilePath] = useState<string | null>(null);
  const [tracks, setTracks] = useState<TrackInfo[] | null>(null);
  const [referenceIndex, setReferenceIndex] = useState<number | null>(null);
  const [targetIndices, setTargetIndices] = useState<number[]>([]);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [prefetching, setPrefetching] = useState(false);

  const [analyses, setAnalyses] = useState<Record<number, TrackAnalysis>>({});
  const [editingTrack, setEditingTrack] = useState<number | null>(null);
  // Which analyzed track's tab is showing -- only one card is ever rendered
  // at a time (see field-analysis below), so an arbitrary number of
  // analyzed tracks never needs the panel itself to scroll.
  const [activeAnalysisTab, setActiveAnalysisTab] = useState<number | null>(null);

  const pollHealth = useCallback(() => {
    let cancelled = false;
    let attempts = 0;
    setEngineStatus("starting");
    async function poll() {
      if (await checkHealth()) {
        if (!cancelled) setEngineStatus("ready");
        return;
      }
      attempts += 1;
      if (attempts > HEALTH_POLL_ATTEMPTS) {
        if (!cancelled) setEngineStatus("unreachable");
        return;
      }
      if (!cancelled) setTimeout(poll, 500);
    }
    poll();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => pollHealth(), [pollHealth]);

  async function handleOpenFile() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Vidéo/Audio", extensions: ["mkv", "mp4", "wav", "flac", "aac", "mp3"] }],
    });
    if (!selected || Array.isArray(selected)) return;

    setFilePath(selected);
    setTracks(null);
    setReferenceIndex(null);
    setTargetIndices([]);
    setProbeError(null);
    setAnalyses({});
    setEditingTrack(null);
    setActiveAnalysisTab(null);

    try {
      const res = await probe(selected);
      setTracks(res.tracks);
      if (res.tracks.length >= 2) {
        setReferenceIndex(res.tracks[0].index);
        setTargetIndices(res.tracks.slice(1).map((t) => t.index));
        prefetchTracks(selected, res.tracks.map((t) => t.index));
      }
    } catch (err) {
      setProbeError(err instanceof Error ? err.message : String(err));
    }
  }

  /** Fire-and-forget: warms the engine's cache so the first "Analyser" click
   * doesn't pay the ~7s-per-track extraction cost that's otherwise
   * unavoidable on a cold cache (see api.ts's startPrefetchJob). */
  function prefetchTracks(path: string, trackIndices: number[]) {
    setPrefetching(true);
    startPrefetchJob(path, trackIndices)
      .then((jobId) => {
        connectJobWS<PrefetchResponse>(jobId, (event) => {
          if (event.type !== "log") setPrefetching(false);
        });
      })
      .catch(() => setPrefetching(false));
  }

  function handleReferenceChange(index: number) {
    setReferenceIndex(index);
    // A track can't be both the reference and something to correct.
    setTargetIndices((current) => current.filter((i) => i !== index));
  }

  function toggleTarget(index: number) {
    setTargetIndices((current) =>
      current.includes(index) ? current.filter((i) => i !== index) : [...current, index],
    );
  }

  function updateAnalysis(trackIndex: number, patch: Partial<TrackAnalysis> | ((entry: TrackAnalysis) => Partial<TrackAnalysis>)) {
    setAnalyses((current) => {
      const entry = current[trackIndex];
      if (!entry) return current;
      const nextPatch = typeof patch === "function" ? patch(entry) : patch;
      return { ...current, [trackIndex]: { ...entry, ...nextPatch } };
    });
  }

  async function analyzeTrack(trackIndex: number, refIndex: number) {
    if (!filePath) return;
    setAnalyses((current) => ({
      ...current,
      [trackIndex]: {
        status: "running",
        referenceIndex: refIndex,
        log: [],
        result: null,
        error: null,
        rendering: false,
        renderLog: [],
        renderResult: null,
        renderError: null,
      },
    }));
    try {
      const jobId = await startSegmentsJob(filePath, refIndex, filePath, trackIndex);
      connectJobWS<SegmentsResponse>(jobId, (event) => {
        if (event.type === "log") {
          updateAnalysis(trackIndex, (e) => ({ log: [...e.log, event.message] }));
        } else if (event.type === "done") {
          updateAnalysis(trackIndex, { status: "done", result: event.result });
        } else if (event.type === "error") {
          updateAnalysis(trackIndex, { status: "error", error: event.message });
        }
      });
    } catch (err) {
      updateAnalysis(trackIndex, { status: "error", error: err instanceof Error ? err.message : String(err) });
    }
  }

  function handleAnalyzeSelected() {
    if (referenceIndex === null) return;
    for (const idx of targetIndices) {
      analyzeTrack(idx, referenceIndex);
    }
    if (targetIndices.length > 0) {
      // Keep whatever tab the user's already looking at if it's still part
      // of this run; otherwise default to the first newly-analyzed track.
      setActiveAnalysisTab((current) => (current !== null && targetIndices.includes(current) ? current : targetIndices[0]));
    }
  }

  async function renderTrack(trackIndex: number) {
    const entry = analyses[trackIndex];
    if (!filePath || !entry || !entry.result) return;
    updateAnalysis(trackIndex, { rendering: true, renderLog: [], renderResult: null, renderError: null });
    try {
      const jobId = await startSegmentedRenderJob(filePath, entry.referenceIndex, trackIndex, entry.result.segments);
      connectJobWS<RenderResponse>(jobId, (event) => {
        if (event.type === "log") {
          updateAnalysis(trackIndex, (e) => ({ renderLog: [...e.renderLog, event.message] }));
        } else if (event.type === "done") {
          updateAnalysis(trackIndex, { rendering: false, renderResult: event.result });
        } else if (event.type === "error") {
          updateAnalysis(trackIndex, { rendering: false, renderError: event.message });
        }
      });
    } catch (err) {
      updateAnalysis(trackIndex, { rendering: false, renderError: err instanceof Error ? err.message : String(err) });
    }
  }

  const anySelectedRunning = targetIndices.some((i) => analyses[i]?.status === "running");
  const analyzedTracks = (tracks ?? []).filter((t) => analyses[t.index]);
  const editingEntry = editingTrack !== null ? analyses[editingTrack] : null;

  // Nothing in the app is usable before the sidecar answers -- a full-screen
  // splash instead of a text banner over an inert shell makes that obvious
  // and stops the user from clicking around a UI that can't do anything yet.
  if (engineStatus !== "ready") {
    return (
      <div className="container startup-screen">
        {engineStatus === "starting" ? (
          <>
            <div className="spinner" aria-hidden="true" />
            <p className="startup-text">Démarrage du moteur...</p>
          </>
        ) : (
          <>
            <p className="startup-text error">Moteur injoignable — le sidecar a-t-il démarré ? (voir la console)</p>
            <button onClick={pollHealth}>Réessayer</button>
          </>
        )}
      </div>
    );
  }

  return (
    <div className="container">
      <header className="app-header">
        <h1>SyncAudio</h1>
      </header>

      <div className="file-bar">
        <button className="primary-button" onClick={handleOpenFile}>
          Ouvrir un fichier
        </button>
        {filePath && (
          <span className="file-path" title={filePath}>
            {basename(filePath)}
          </span>
        )}
        {prefetching && (
          <span className="prefetch-status" title="Analyse des pistes en arrière-plan pour accélérer le premier clic sur Analyser.">
            Analyse audio en cours...
          </span>
        )}
      </div>

      <main className="app-main">
        <section className="panel field-tracks">
          <h2>Pistes</h2>
          {!tracks && !probeError && <p className="placeholder">Ouvre un fichier pour voir ses pistes.</p>}
          {probeError && <p className="error">{probeError}</p>}
          {tracks && tracks.length < 2 && (
            <p className="error">Ce fichier n'a qu'une seule piste audio : rien à comparer.</p>
          )}
          {tracks && tracks.length >= 2 && (
            <>
              <div className="tracks-table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Piste</th>
                      <th>Langue</th>
                      <th>Codec</th>
                      <th>Réf.</th>
                      <th>Analyser</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tracks.map((t) => (
                      <tr key={t.index}>
                        <td>@{t.index}</td>
                        <td>{t.language ?? "?"}</td>
                        <td>{t.codec ?? "?"}</td>
                        <td>
                          <input
                            type="radio"
                            name="reference"
                            checked={referenceIndex === t.index}
                            onChange={() => handleReferenceChange(t.index)}
                          />
                        </td>
                        <td>
                          <input
                            type="checkbox"
                            disabled={referenceIndex === t.index}
                            checked={targetIndices.includes(t.index)}
                            onChange={() => toggleTarget(t.index)}
                          />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="tracks-actions">
                <button
                  className="primary-button"
                  onClick={handleAnalyzeSelected}
                  disabled={referenceIndex === null || targetIndices.length === 0 || anySelectedRunning}
                  title="Détecte le décalage de chaque piste cochée par rapport à la référence (dérive et sauts nets inclus), avec la courbe correspondante."
                >
                  {anySelectedRunning ? "Analyse en cours..." : "Analyser"}
                </button>
              </div>
            </>
          )}
        </section>

        <section className="panel field-analysis">
          <h2>Analyse</h2>
          {analyzedTracks.length === 0 && (
            <p className="placeholder">Coche une ou plusieurs pistes à corriger, puis clique sur « Analyser ».</p>
          )}
          {analyzedTracks.length > 0 && (
            <div className="analysis-tabs">
              {analyzedTracks.map((t) => {
                const entry = analyses[t.index];
                return (
                  <button
                    key={t.index}
                    className={`analysis-tab status-${entry.status}${activeAnalysisTab === t.index ? " active" : ""}`}
                    onClick={() => setActiveAnalysisTab(t.index)}
                  >
                    Piste @{t.index}
                  </button>
                );
              })}
            </div>
          )}
          {analyzedTracks
            .filter((t) => t.index === activeAnalysisTab)
            .map((t) => {
              const entry = analyses[t.index];
              return (
                <div className={`analysis-card status-${entry.status}`} key={t.index}>
                  <div className="analysis-top">
                    <div className="analysis-summary">
                      <h3>
                        Piste @{t.index} ({t.language ?? "?"})
                      </h3>
                      <LogPanel lines={entry.log} />
                      {entry.error && <p className="error">{entry.error}</p>}
                      {entry.status === "running" && !entry.result && (
                        <p className="placeholder">Analyse en cours...</p>
                      )}
                      {entry.result && (
                        <p>
                          {entry.result.segments.length} segment
                          {entry.result.segments.length > 1 ? "s" : ""}
                          <button className="small-button edit-button" onClick={() => setEditingTrack(t.index)}>
                            Modifier
                          </button>
                          <button
                            className="small-button edit-button"
                            onClick={() => renderTrack(t.index)}
                            disabled={entry.rendering}
                          >
                            {entry.rendering ? "Export en cours..." : "Exporter cette piste"}
                          </button>
                        </p>
                      )}
                    </div>
                    {entry.result && <SegmentChart segments={entry.result.segments} />}
                  </div>
                  {entry.result && (
                    <div className="segments-result">
                      {filePath && (
                        <TrackPreview
                          filePath={filePath}
                          referenceIndex={entry.referenceIndex}
                          trackIndex={t.index}
                          segments={entry.result.segments}
                          referenceStartTime={tracks?.find((tr) => tr.index === entry.referenceIndex)?.start_time ?? 0}
                          trackStartTime={t.start_time}
                        />
                      )}
                      <LogPanel lines={entry.renderLog} />
                      {entry.renderError && <p className="error">{entry.renderError}</p>}
                      {entry.renderResult && (
                        <p className="render-success" title={entry.renderResult.written.join(", ")}>
                          Fichier écrit : {entry.renderResult.written.map(basename).join(", ")}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
        </section>
      </main>

      {editingTrack !== null && editingEntry?.result && (
        <SegmentEditor
          segments={editingEntry.result.segments}
          onClose={() => setEditingTrack(null)}
          onSave={(edited) =>
            updateAnalysis(editingTrack, (e) => ({ result: e.result ? { ...e.result, segments: edited } : e.result }))
          }
        />
      )}
    </div>
  );
}

export default App;
