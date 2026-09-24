import { useCallback, useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  checkHealth,
  probe,
  startAlignJob,
  startSegmentsJob,
  startPrefetchJob,
  connectJobWS,
  type TrackInfo,
  type AlignResponse,
  type SegmentsResponse,
  type SegmentOut,
  type PrefetchResponse,
} from "./api";
import { SegmentChart } from "./SegmentChart";
import { LogPanel } from "./LogPanel";
import { SegmentEditor } from "./SegmentEditor";
import "./App.css";

type EngineStatus = "starting" | "ready" | "unreachable";

const HEALTH_POLL_ATTEMPTS = 40; // 40 * 500ms = 20s before giving up

function App() {
  const [engineStatus, setEngineStatus] = useState<EngineStatus>("starting");
  const [filePath, setFilePath] = useState<string | null>(null);
  const [tracks, setTracks] = useState<TrackInfo[] | null>(null);
  const [referenceIndex, setReferenceIndex] = useState<number | null>(null);
  const [targetIndices, setTargetIndices] = useState<number[]>([]);
  const [probeError, setProbeError] = useState<string | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [result, setResult] = useState<AlignResponse | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [detectError, setDetectError] = useState<string | null>(null);

  const [analyzingTrack, setAnalyzingTrack] = useState<number | null>(null);
  const [segmentsResult, setSegmentsResult] = useState<SegmentsResponse | null>(null);
  const [segmentsLog, setSegmentsLog] = useState<string[]>([]);
  const [segmentsError, setSegmentsError] = useState<string | null>(null);
  const [editingSegments, setEditingSegments] = useState(false);
  const [prefetching, setPrefetching] = useState(false);

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
    setResult(null);
    setLogLines([]);
    setDetectError(null);
    setProbeError(null);
    setAnalyzingTrack(null);
    setSegmentsResult(null);
    setSegmentsLog([]);
    setSegmentsError(null);
    setEditingSegments(false);

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

  /** Fire-and-forget: warms the engine's cache so the first "Détecter"/"Analyser"
   * click doesn't pay the ~7s-per-track extraction cost that's otherwise
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

  async function handleDetect() {
    if (!filePath || referenceIndex === null || targetIndices.length === 0) return;
    setDetecting(true);
    setDetectError(null);
    setResult(null);
    setLogLines([]);
    try {
      const jobId = await startAlignJob(filePath, referenceIndex, filePath, targetIndices);
      connectJobWS<AlignResponse>(jobId, (event) => {
        if (event.type === "log") {
          setLogLines((lines) => [...lines, event.message]);
        } else if (event.type === "done") {
          setResult(event.result);
          setDetecting(false);
        } else if (event.type === "error") {
          setDetectError(event.message);
          setDetecting(false);
        }
      });
    } catch (err) {
      setDetectError(err instanceof Error ? err.message : String(err));
      setDetecting(false);
    }
  }

  async function handleAnalyzeSegments(trackIndex: number) {
    if (!filePath || referenceIndex === null) return;
    setAnalyzingTrack(trackIndex);
    setSegmentsError(null);
    setSegmentsResult(null);
    setSegmentsLog([]);
    setEditingSegments(false);
    try {
      const jobId = await startSegmentsJob(filePath, referenceIndex, filePath, trackIndex);
      connectJobWS<SegmentsResponse>(jobId, (event) => {
        if (event.type === "log") {
          setSegmentsLog((lines) => [...lines, event.message]);
        } else if (event.type === "done") {
          setSegmentsResult(event.result);
          setAnalyzingTrack(null);
        } else if (event.type === "error") {
          setSegmentsError(event.message);
          setAnalyzingTrack(null);
        }
      });
    } catch (err) {
      setSegmentsError(err instanceof Error ? err.message : String(err));
      setAnalyzingTrack(null);
    }
  }

  return (
    <div className="container">
      <header className="app-header">
        <h1>SyncAudio</h1>
        {engineStatus !== "ready" && (
          <p className="engine-status">
            {engineStatus === "starting" ? (
              "Démarrage du moteur..."
            ) : (
              <>
                Moteur injoignable — le sidecar a-t-il démarré ? (voir la console){" "}
                <button className="small-button" onClick={pollHealth}>
                  Réessayer
                </button>
              </>
            )}
          </p>
        )}
      </header>

      <div className="file-bar">
        <button onClick={handleOpenFile} disabled={engineStatus !== "ready"}>
          Ouvrir un fichier
        </button>
        {filePath && <span className="file-path">{filePath}</span>}
        {prefetching && (
          <span className="prefetch-status" title="Analyse des pistes en arrière-plan pour accélérer la première détection.">
            Pré-analyse en cours...
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
              <table>
                <thead>
                  <tr>
                    <th>Piste</th>
                    <th>Langue</th>
                    <th>Codec</th>
                    <th>Réf.</th>
                    <th>Corriger</th>
                    <th>Détail</th>
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
                      <td>
                        <button
                          className="small-button"
                          title="Analyse détaillée de cette piste seule : détecte aussi une dérive progressive ou des sauts nets (pas juste un décalage constant), et affiche la courbe ci-dessous."
                          disabled={referenceIndex === t.index || analyzingTrack !== null}
                          onClick={() => handleAnalyzeSegments(t.index)}
                        >
                          {analyzingTrack === t.index ? "Analyse..." : "Analyser"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="tracks-actions">
                <button
                  onClick={handleDetect}
                  disabled={detecting || referenceIndex === null || targetIndices.length === 0}
                  title="Décalage global unique (suppose qu'il est constant sur toute la piste) ; peut traiter plusieurs pistes cochées à la fois."
                >
                  {detecting ? "Détection en cours..." : "Détecter le décalage"}
                </button>

                <details className="hint">
                  <summary>Quelle différence avec « Analyser » ?</summary>
                  <p>
                    « Détecter le décalage » donne un seul décalage global (en supposant qu'il est constant) et peut
                    traiter plusieurs pistes à la fois. « Analyser » (par piste) détecte aussi une dérive
                    progressive ou des sauts, avec la courbe correspondante — pas plus lent en pratique (l'essentiel
                    du temps est pris par l'extraction/l'analyse audio, identique dans les deux cas), juste plus
                    détaillé et limité à une piste.
                  </p>
                </details>
              </div>
            </>
          )}
        </section>

        <section className="panel field-quick">
          <h2>Détection rapide</h2>
          {logLines.length === 0 && !detectError && !result && (
            <p className="placeholder">Résultat de « Détecter le décalage » ici.</p>
          )}
          <LogPanel lines={logLines} />
          {detectError && <p className="error">{detectError}</p>}
          {result && (
            <div className="result">
              {result.results.map((r) => (
                <p key={r.track}>
                  {r.track} : décalage <strong>{r.offset_seconds.toFixed(3)}s</strong>
                  {r.ambiguous ? " (ambigu)" : ""}
                </p>
              ))}
            </div>
          )}
        </section>

        <section className="panel field-detail">
          <h2>Analyse détaillée</h2>
          {segmentsLog.length === 0 && !segmentsError && !segmentsResult && (
            <p className="placeholder">Résultat de « Analyser » (par piste) ici, avec la courbe de décalage.</p>
          )}
          <LogPanel lines={segmentsLog} />
          {segmentsError && <p className="error">{segmentsError}</p>}
          {segmentsResult && (
            <div className="segments-result">
              <p>
                {segmentsResult.track} vs {segmentsResult.reference} — {segmentsResult.segments.length} segment
                {segmentsResult.segments.length > 1 ? "s" : ""}
                <button className="small-button edit-button" onClick={() => setEditingSegments(true)}>
                  Modifier
                </button>
              </p>
              <SegmentChart segments={segmentsResult.segments} />
            </div>
          )}
        </section>
      </main>

      {editingSegments && segmentsResult && (
        <SegmentEditor
          segments={segmentsResult.segments}
          onClose={() => setEditingSegments(false)}
          onSave={(edited: SegmentOut[]) => setSegmentsResult((r) => (r ? { ...r, segments: edited } : r))}
        />
      )}
    </div>
  );
}

export default App;
