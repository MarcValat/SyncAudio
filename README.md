# SyncAudio

Calcule le décalage temporel entre des pistes audio de langues différentes (doublages) d'un même contenu, en se basant sur la musique et les bruitages (M&E) qu'elles ont en commun plutôt que sur les dialogues, qui diffèrent d'une langue à l'autre.

Entrées supportées : fichiers audio seuls (wav, mp3, flac…) et fichiers vidéo multi-pistes (mkv, mp4…) — tout ce que `ffmpeg` sait décoder. Cette v1 se limite au **calcul** du décalage ; elle ne génère pas de fichiers réalignés.

## Comment ça marche

1. Chaque piste est décodée en mono 16 kHz via `ffmpeg`.
2. Un spectrogramme (STFT) est calculé, puis séparé en composantes harmonique/percussive par filtrage médian : la composante percussive capte surtout les transitoires (percussions, impacts, effets sonores), largement indépendants de la langue parlée.
3. Une enveloppe d'onset (flux spectral) est extraite de cette composante percussive pour chaque piste.
4. Les enveloppes sont corrélées par FFT pour trouver le décalage qui les aligne le mieux, avec interpolation sous-trame pour affiner la précision.
5. Un score de confiance mesure à quel point ce décalage se distingue statistiquement du reste des décalages testés (voir limites ci-dessous).

Hypothèse : le décalage entre deux pistes est **constant** (pas de dérive de vitesse ni de montage différent). Par défaut, seuls les décalages allant jusqu'à environ 10 % de la durée de la piste la plus courte sont recherchés — largement suffisant pour un désynchronisme de quelques secondes à quelques minutes sur un contenu de plusieurs dizaines de minutes.

### Limites connues

- Sans musique/bruitage réellement partagé (ex. deux pistes de dialogue seul), le score de confiance peut occasionnellement rester modéré plutôt que nul, surtout sur du contenu très long et rythmiquement régulier. Le score de confiance est une indication utile, pas une garantie statistique absolue — croisez-le avec un contrôle d'écoute si le résultat est déterminant.
- Un décalage supérieur à ~10 % de la durée de la piste la plus courte ne sera pas trouvé (ajustable via la constante `_MIN_OVERLAP_FRACTION` dans `align.py`, pas encore exposée en option CLI).
- Pas de gestion du montage différent (scènes coupées/ajoutées) ni du time-stretch.

## Installation

Nécessite [uv](https://docs.astral.sh/uv/).

```
uv sync
```

`ffmpeg` n'a pas besoin d'être installé séparément : le paquet `imageio-ffmpeg` embarque un binaire statique utilisé automatiquement si aucun `ffmpeg` n'est trouvé sur le PATH.

## Usage

Lister les pistes audio d'un fichier (utile pour les conteneurs multi-langues) :

```
uv run syncaudio probe film.mkv
```

Calculer le décalage de chaque candidat par rapport à une référence :

```
uv run syncaudio align film.mkv@0 vf.wav film.mkv@1
```

Chaque piste s'indique par `chemin` (piste unique) ou `chemin@INDEX` (INDEX = numéro de piste audio dans le fichier, 0-based, donné par `probe`). Le `@` est utilisé plutôt que `:` pour ne pas entrer en conflit avec les lettres de lecteur Windows (`C:\...`).

Décalage positif = la piste candidate est en retard par rapport à la référence.

Sortie JSON pour un usage scripté :

```
uv run syncaudio align ref.wav candidate.wav --json
```

### Accélérer sur de gros fichiers

Le temps d'extraction (ffmpeg) et surtout d'analyse spectrale (HPSS) croît avec la durée traitée. Comme l'algorithme cherche un décalage **constant**, il n'a pas besoin de toute la piste : un extrait représentatif suffit. `--start` et `--duration` (en secondes) limitent l'extraction/analyse à une fenêtre de chaque piste, ce qui accélère le traitement dans les mêmes proportions :

```
uv run syncaudio align film.mkv@0 vf.wav --start 300 --duration 600
```

N'analyse ici que les 10 minutes commençant à 5 minutes (utile pour sauter un générique/logo initial, souvent peu riche en musique/bruitages). Deux points à garder en tête :

- L'extrait doit contenir un minimum d'activité musique/bruitages (une scène uniquement silencieuse ou dialoguée donnera une confiance faible).
- Par défaut, seuls les décalages allant jusqu'à ~10 % de la durée de l'extrait sont recherchés (voir « Limites connues » ci-dessus) : avec `--duration 600`, un décalage réel de plus d'environ 60s ne sera pas trouvé. Si le décalage attendu est plus grand, augmentez `--duration` en conséquence.

## Développement

```
uv run pytest
```

`tests/test_align.py` valide l'algorithme sur des signaux synthétiques (sans ffmpeg). `tests/test_ffmpeg_backend.py` valide l'extraction/probe de bout en bout avec le ffmpeg embarqué.
