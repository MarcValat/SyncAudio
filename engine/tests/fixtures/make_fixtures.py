"""Genere des variantes volontairement desynchronisees de testfile.mkv.

Usage (depuis engine/) :
    uv run python tests/fixtures/make_fixtures.py

Prend un extrait de testfile.mkv (piste 0 = reference, jamais modifiee ;
piste 1 = candidate) et produit plusieurs MKV sous generated/, chacun avec
la candidate desynchronisee d'une facon differente et documentee au sample
pres dans MANIFEST.json. Rien de ce que ce script lit ou ecrit n'est commite
(voir .gitignore) a part ce fichier et le manifeste.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from syncaudio.ffmpeg_backend import resolve_ffmpeg

FIXTURES_DIR = Path(__file__).parent
GENERATED_DIR = FIXTURES_DIR / "generated"
PROJECT_ROOT = FIXTURES_DIR.parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "testfile.mkv"

REFERENCE_TRACK = 0  # jpn, ne doit jamais etre modifiee
CANDIDATE_TRACK = 1  # fre, c'est elle qu'on desynchronise


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Echec: {' '.join(cmd)}\n{proc.stderr.decode(errors='replace')}")


def extract_audio_wav(ffmpeg: str, source: Path, track: int, start: float, duration: float, out: Path) -> None:
    run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-i", str(source), "-t", str(duration),
            "-map", f"0:a:{track}", "-c:a", "pcm_s16le", str(out),
        ]
    )


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype="<i2").reshape(-1, n_channels)
    return data, sr


def write_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(data.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.ascontiguousarray(data, dtype="<i2").tobytes())


@dataclass
class Breakpoint:
    time_s: float
    delta_s: float


def apply_piecewise_offset(cand: np.ndarray, sr: int, breakpoints: list[Breakpoint]) -> np.ndarray:
    """Apply a schedule of offset changes to a candidate track.

    At each breakpoint's ``time_s``, the candidate's lag behind the reference
    changes by ``delta_s`` from that point onward: a positive delta inserts
    that many seconds of silence (candidate now lags more), a negative delta
    removes that many seconds of candidate content (candidate now lags less
    / leads more).
    """
    chunks = []
    prev_sample = 0
    for bp in sorted(breakpoints, key=lambda b: b.time_s):
        idx = int(round(bp.time_s * sr))
        idx = max(idx, prev_sample)
        chunks.append(cand[prev_sample:idx])
        if bp.delta_s > 0:
            silence = np.zeros((int(round(bp.delta_s * sr)), cand.shape[1]), dtype=cand.dtype)
            chunks.append(silence)
            prev_sample = idx
        else:
            cut = int(round(-bp.delta_s * sr))
            prev_sample = idx + cut
    chunks.append(cand[prev_sample:])
    return np.concatenate(chunks, axis=0)


def time_stretch(cand: np.ndarray, factor: float) -> np.ndarray:
    """Uniformly stretch (factor > 1) or compress (factor < 1) the whole track.

    Simulates a candidate running at a slightly different speed than the
    reference from the very first sample (progressive drift). Pitch is not
    preserved -- irrelevant here since detection works on the onset envelope,
    not pitch, and these are synthetic ground-truth fixtures, not deliverables.
    """
    n = len(cand)
    new_n = int(round(n * factor))
    old_idx = np.linspace(0, n - 1, new_n)
    base_idx = np.arange(n)
    out = np.stack(
        [np.interp(old_idx, base_idx, cand[:, ch]) for ch in range(cand.shape[1])],
        axis=1,
    )
    return out.astype(cand.dtype)


def mux(ffmpeg: str, source: Path, start: float, duration: float, ref_wav: Path, cand_wav: Path, out: Path) -> None:
    ref_flac = ref_wav.with_suffix(".flac")
    cand_flac = cand_wav.with_suffix(".flac")
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(ref_wav), str(ref_flac)])
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(cand_wav), str(cand_flac)])
    run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-t", str(duration), "-i", str(source),
            "-i", str(ref_flac), "-i", str(cand_flac),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-c:v", "copy", "-c:a", "flac",
            "-metadata:s:a:0", "language=jpn",
            "-metadata:s:a:1", "language=fre",
            str(out),
        ]
    )
    ref_flac.unlink()
    cand_flac.unlink()


@dataclass
class Fixture:
    name: str
    description: str
    breakpoints: list[Breakpoint] = field(default_factory=list)
    stretch_factor: float | None = None

    def to_manifest(self) -> dict:
        return {
            "file": f"{self.name}.mkv",
            "description": self.description,
            "breakpoints": [{"time_s": b.time_s, "delta_s": b.delta_s} for b in self.breakpoints],
            "stretch_factor": self.stretch_factor,
        }


def build_fixtures(seed: int) -> list[Fixture]:
    rng = random.Random(seed)
    multi_times = sorted(rng.uniform(45.0, 315.0) for _ in range(3))
    multi_deltas = [rng.choice([-1, 1]) * rng.uniform(1.0, 3.5) for _ in multi_times]

    return [
        Fixture("sync_base", "Temoin : aucune desynchronisation (sanity check, offset attendu = 0)."),
        Fixture(
            "offset_const_plus",
            "Decalage constant : la candidate est en retard de 5s des le debut.",
            breakpoints=[Breakpoint(0.0, 5.0)],
        ),
        Fixture(
            "offset_const_minus",
            "Decalage constant : la candidate est en avance de 3s des le debut.",
            breakpoints=[Breakpoint(0.0, -3.0)],
        ),
        Fixture(
            "jump_single",
            "Synchro au debut, puis un saut net de +4s a t=150s.",
            breakpoints=[Breakpoint(150.0, 4.0)],
        ),
        Fixture(
            "jump_multi",
            f"Plusieurs sauts a des instants pseudo-aleatoires (seed={seed}).",
            breakpoints=[Breakpoint(t, d) for t, d in zip(multi_times, multi_deltas)],
        ),
        Fixture(
            "drift_slower",
            "Derive progressive : la candidate tourne 1% plus lentement (retard croissant).",
            stretch_factor=1.01,
        ),
        Fixture(
            "drift_faster",
            "Derive progressive : la candidate tourne 1% plus vite (avance croissante).",
            stretch_factor=0.99,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--start", type=float, default=0.0, help="Debut de l'extrait (s).")
    parser.add_argument("--duration", type=float, default=360.0, help="Duree de l'extrait (s).")
    parser.add_argument("--seed", type=int, default=1234, help="Seed pour jump_multi.")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source introuvable : {args.source}")

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg()

    print(f"[extraction] reference (piste {REFERENCE_TRACK}) ...")
    ref_wav = GENERATED_DIR / "_reference.wav"
    extract_audio_wav(ffmpeg, args.source, REFERENCE_TRACK, args.start, args.duration, ref_wav)

    print(f"[extraction] candidate (piste {CANDIDATE_TRACK}) ...")
    cand_wav_src = GENERATED_DIR / "_candidate_source.wav"
    extract_audio_wav(ffmpeg, args.source, CANDIDATE_TRACK, args.start, args.duration, cand_wav_src)

    cand_data, cand_sr = read_wav(cand_wav_src)

    fixtures = build_fixtures(args.seed)
    resolved_source = args.source.resolve()
    try:
        source_display = resolved_source.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_display = resolved_source.name  # hors du repo : ne pas exposer le chemin local

    manifest = {
        "source": source_display,
        "window": {"start_s": args.start, "duration_s": args.duration},
        "reference_track": REFERENCE_TRACK,
        "candidate_track": CANDIDATE_TRACK,
        "offset_sign_convention": "positif = la candidate est en retard sur la reference",
        "fixtures": [],
    }

    for fx in fixtures:
        print(f"[fixture] {fx.name} : {fx.description}")
        if fx.stretch_factor is not None:
            variant = time_stretch(cand_data, fx.stretch_factor)
        elif fx.breakpoints:
            variant = apply_piecewise_offset(cand_data, cand_sr, fx.breakpoints)
        else:
            variant = cand_data

        variant_wav = GENERATED_DIR / f"_{fx.name}_candidate.wav"
        write_wav(variant_wav, variant, cand_sr)

        out_mkv = GENERATED_DIR / f"{fx.name}.mkv"
        mux(ffmpeg, args.source, args.start, args.duration, ref_wav, variant_wav, out_mkv)
        variant_wav.unlink()

        manifest["fixtures"].append(fx.to_manifest())

    ref_wav.unlink()
    cand_wav_src.unlink()

    manifest_path = FIXTURES_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTermine. Fixtures dans {GENERATED_DIR}, verite-terrain dans {manifest_path}")


if __name__ == "__main__":
    main()
