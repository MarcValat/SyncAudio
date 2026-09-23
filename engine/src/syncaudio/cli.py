from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click

from syncaudio.align import estimate_offset
from syncaudio.features import extract_envelope
from syncaudio.ffmpeg_backend import FFmpegError, extract_pcm, parse_track_spec, probe_audio_streams
from syncaudio.render import plan_corrections, render as render_tracks


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


@cli.command()
@click.argument("input_path", metavar="INPUT")
@click.option(
    "--reference",
    "reference_index",
    required=True,
    type=int,
    help="Index (voir `probe`) de la piste de référence, jamais modifiée.",
)
@click.option(
    "--track",
    "track_indices",
    multiple=True,
    type=int,
    help="Index d'une piste à resynchroniser (répétable). Défaut : toutes les pistes sauf la référence.",
)
@click.option("-o", "--output", "output_path", default=None, help="Fichier de sortie. Défaut : <INPUT>.synced.mkv")
@click.option(
    "--audio-only",
    is_flag=True,
    help="N'exporte que la/les piste(s) corrigée(s) en .flac, sans remuxer de MKV.",
)
@click.option("--dry-run", is_flag=True, help="Détecte et affiche la correction prévue sans rien écrire.")
@click.option(
    "--start",
    default=0.0,
    show_default=True,
    help="Ignore les START premières secondes lors de la détection (n'affecte pas le rendu).",
)
@click.option(
    "--duration",
    default=None,
    type=float,
    help="Limite la détection aux DURATION secondes suivant --start (n'affecte pas le rendu).",
)
@click.option("-q", "--quiet", is_flag=True, help="Ne pas afficher la progression sur stderr.")
def render(
    input_path: str,
    reference_index: int,
    track_indices: tuple[int, ...],
    output_path: str | None,
    audio_only: bool,
    dry_run: bool,
    start: float,
    duration: float | None,
    quiet: bool,
) -> None:
    """Corrige le décalage constant de pistes de INPUT par rapport à --reference.

    INPUT est un seul fichier conteneur (ex. mkv) avec plusieurs pistes
    audio. Ne gère que le cas décalage constant (pas de dérive ni de sauts) ;
    voir le README pour les limites.
    """
    try:
        streams = probe_audio_streams(input_path)
    except FFmpegError as exc:
        raise click.ClickException(str(exc)) from exc

    all_indices = [s.index for s in streams]
    if reference_index not in all_indices:
        raise click.ClickException(
            f"Index de référence {reference_index} absent de {input_path!r} (pistes : {all_indices})."
        )

    targets = list(track_indices) if track_indices else [i for i in all_indices if i != reference_index]
    if reference_index in targets:
        raise click.ClickException("La piste de référence ne peut pas aussi être une piste à corriger.")
    unknown = [i for i in targets if i not in all_indices]
    if unknown:
        raise click.ClickException(f"Index(es) inconnu(s) : {unknown} (pistes disponibles : {all_indices}).")

    try:
        corrections = plan_corrections(
            input_path, reference_index, targets, start=start, duration=duration, log=lambda m: _log(m, quiet)
        )
    except FFmpegError as exc:
        raise click.ClickException(str(exc)) from exc

    for corr in corrections:
        if corr.needs_correction:
            action = "trim début + pad" if corr.offset_seconds > 0 else "pad début + trim"
            action = f"décalage={corr.offset_seconds:+.3f}s -> {action}"
        else:
            action = "déjà synchro, aucune correction"
        flag = "  [!] ambigu (contenu répétitif ?)" if corr.ambiguous else ""
        click.echo(f"  @{corr.index} (langue={corr.language or '?'})  {action}  confiance={corr.confidence:.2f}{flag}")

    if dry_run:
        click.echo("[dry-run] rien écrit.")
        return

    if output_path is None:
        stem = Path(input_path)
        while stem.suffix:
            stem = stem.with_suffix("")
        output_path = str(stem) + ".synced.mkv"

    try:
        written = render_tracks(input_path, reference_index, corrections, output_path, audio_only=audio_only)
    except FFmpegError as exc:
        raise click.ClickException(str(exc)) from exc

    for path in written:
        click.echo(f"Écrit : {path}")


def main() -> None:
    try:
        cli()
    except FFmpegError as exc:
        click.echo(f"Erreur : {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
