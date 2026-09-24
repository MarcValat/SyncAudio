import { useCallback, useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import {
  checkHealth,
  probe,
  startAlignJob,
  connectJobWS,
  type TrackInfo,
  type AlignResponse,
} from "./api";
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

    try {
      const res = await probe(selected);
      setTracks(res.tracks);
      if (res.tracks.length >= 2) {
        setReferenceIndex(res.tracks[0].index);
        setTargetIndices(res.tracks.slice(1).map((t) => t.index));
      }
    } catch (err) {
      setProbeError(err instanceof Error ? err.message : String(err));
    }
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
      connectJobWS(jobId, (event) => {
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

  return (
    <main className="container">
      <h1>SyncAudio</h1>

      {engineStatus !== "ready" && (
        <p className="engine-status">
          {engineStatus === "starting" ? (
            "Démarrage du moteur..."
          ) : (
            <>
              Moteur injoignable — le sidecar a-t-il démarré ? (voir la console)
              <br />
              <button onClick={pollHealth}>Réessayer</button>
            </>
          )}
        </p>
      )}

      <button onClick={handleOpenFile} disabled={engineStatus !== "ready"}>
        Ouvrir un fichier
      </button>
      {filePath && <p className="file-path">{filePath}</p>}
      {probeError && <p className="error">{probeError}</p>}

      {tracks && tracks.length < 2 && (
        <p className="error">Ce fichier n'a qu'une seule piste audio : rien à comparer.</p>
      )}

      {tracks && tracks.length >= 2 && (
        <div className="tracks">
          <table>
            <thead>
              <tr>
                <th>Piste</th>
                <th>Langue</th>
                <th>Codec</th>
                <th>Référence</th>
                <th>À corriger</th>
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

          <button onClick={handleDetect} disabled={detecting || referenceIndex === null || targetIndices.length === 0}>
            {detecting ? "Détection en cours..." : "Détecter le décalage"}
          </button>
        </div>
      )}

      {logLines.length > 0 && <pre className="log">{logLines.join("\n")}</pre>}

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
    </main>
  );
}

export default App;
