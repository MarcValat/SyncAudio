from __future__ import annotations

import json
import sys
import time

import click

from syncaudio.align import estimate_offset
from syncaudio.features import extract_envelope
from syncaudio.ffmpeg_backend import FFmpegError, extract_pcm, parse_track_spec, probe_audio_streams


def _log(message: str, quiet: bool) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


@click.group()
def cli() -> None:
    """SyncAudio — aligne des pistes audio multilingues sur leur musique/bruitages communs."""


@cli.command()
@click.argument("path")
def probe(path: str) -> None:
    """Liste les pistes audio d'un fichier (index utilisable avec chemin@INDEX)."""
    try:
        streams = probe_audio_streams(path)
    except FFmpegError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"{path}")
    for s in streams:
        lang = s.language or "?"
        rate = f"{s.sample_rate} Hz" if s.sample_rate else "?"
        channels = f"{s.channels}ch" if s.channels else "?"
        click.echo(f"  @{s.index}  langue={lang:<5} codec={s.codec:<10} {rate:<10} {channels}")


@cli.command()
@click.argument("reference")
@click.argument("candidates", nargs=-1, required=True)
@click.option("--json", "as_json", is_flag=True, help="Sortie au format JSON.")
@click.option("--sample-rate", default=16000, show_default=True, help="Fréquence d'échantillonnage d'analyse (Hz).")
@click.option("--n-fft", default=1024, show_default=True, help="Taille de fenêtre STFT.")
@click.option("--hop", default=256, show_default=True, help="Décalage entre fenêtres STFT (échantillons).")
@click.option(
    "--start",
    default=0.0,
    show_default=True,
    help="Ignore les START premières secondes de chaque piste (utile pour sauter un générique/silence).",
)
@click.option(
    "--duration",
    default=None,
    type=float,
    help=(
        "N'analyse que les DURATION premières secondes (après --start) de chaque piste, "
        "au lieu du fichier entier. Accélère énormément le traitement de gros fichiers. "
        "L'extrait doit rester plus long que le décalage réel attendu (voir README)."
    ),
)
@click.option("-q", "--quiet", is_flag=True, help="Ne pas afficher la progression sur stderr.")
def align(
    reference: str,
    candidates: tuple[str, ...],
    as_json: bool,
    sample_rate: int,
    n_fft: int,
    hop: int,
    start: float,
    duration: float | None,
    quiet: bool,
) -> None:
    """Calcule le décalage temporel de CANDIDATES par rapport à REFERENCE.

    Chaque piste s'écrit `chemin` ou `chemin@INDEX` (INDEX = numéro de piste
    audio dans le fichier, voir `syncaudio probe`). Décalage positif = la
    piste candidate est en retard par rapport à la référence.
    """

    def load_track(raw: str):
        spec = parse_track_spec(raw)
        t0 = time.perf_counter()
        window = f" (fenêtre {start:.0f}s+{duration:.0f}s)" if duration else ""
        _log(f"[extraction] {raw}{window} ...", quiet)
        pcm = extract_pcm(spec, sample_rate=sample_rate, start=start or None, duration=duration)
        t1 = time.perf_counter()
        _log(
            f"[extraction] {raw} -> {len(pcm) / sample_rate:.1f}s audio décodés en {t1 - t0:.1f}s",
            quiet,
        )
        _log(f"[analyse]    {raw} : calcul du spectrogramme et de l'enveloppe...", quiet)
        env, frame_rate = extract_envelope(pcm, sample_rate, n_fft=n_fft, hop=hop)
        t2 = time.perf_counter()
        _log(f"[analyse]    {raw} terminée en {t2 - t1:.1f}s", quiet)
        return spec, env, frame_rate

    try:
        _, ref_env, frame_rate = load_track(reference)

        results = []
        for raw in candidates:
            spec, env, _ = load_track(raw)
            _log(f"[alignement] comparaison de {raw} avec la référence...", quiet)
            t0 = time.perf_counter()
            estimate = estimate_offset(ref_env, env, frame_rate)
            _log(f"[alignement] {raw} terminé en {time.perf_counter() - t0:.1f}s", quiet)
            results.append((spec, estimate))
    except FFmpegError as exc:
        raise click.ClickException(str(exc)) from exc

    if as_json:
        payload = {
            "reference": reference,
            "tracks": [
                {
                    "track": spec.raw,
                    "offset_seconds": round(est.offset_seconds, 4),
                    "confidence": round(est.confidence, 4),
                    "ambiguous": est.ambiguous,
                }
                for spec, est in results
            ],
        }
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    click.echo(f"Référence : {reference}")
    for spec, est in results:
        flag = "  [!] ambigu (contenu répétitif ?)" if est.ambiguous else ""
        click.echo(
            f"  {spec.raw:<30} décalage={est.offset_seconds:+8.3f}s  confiance={est.confidence:5.2f}{flag}"
        )


def main() -> None:
    try:
        cli()
    except FFmpegError as exc:
        click.echo(f"Erreur : {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
