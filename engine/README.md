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

Nécessite [uv](https://docs.astral.sh/uv/). Toutes les commandes ci-dessous s'exécutent depuis ce dossier (`engine/`).

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

## Corriger un décalage constant (`render`)

`align` ne fait que mesurer ; `render` corrige, sur un seul fichier conteneur (ex. mkv) avec plusieurs pistes audio :

```
uv run syncaudio render film.mkv --reference 0 --track 1
```

Détecte le décalage de la piste `@1` par rapport à la référence `@0`, puis écrit `film.synced.mkv` : vidéo, sous-titres et pièces jointes copiés tels quels, piste de référence copiée telle quelle, piste corrigée réencodée et casée sur la durée de la référence (silence ajouté si elle devient trop courte, coupée si trop longue). Sans `--track`, toutes les pistes sauf la référence sont corrigées.

La piste corrigée est réencodée dans le même codec que sa source (même bitrate quand il est connu), pas toujours en flac — un flac systématique produisait un fichier final nettement plus gros qu'utile pour une source déjà compressée avec perte. Exception délibérée : une source AAC est réencodée en Opus plutôt qu'en AAC, car l'unique encodeur AAC libre de ffmpeg est mono-thread et nettement plus lent (~15x temps réel, contre ~300x+ pour les autres codecs gérés) — sur un film entier, ça peut ajouter plusieurs minutes pour un gain de fidélité marginal. Un codec source sans encodeur libre disponible (DTS(-HD), TrueHD/MLP...) retombe sur flac (sans perte, juste plus lourd).

Autres options utiles :

- `--dry-run` : affiche le décalage détecté et la correction prévue sans rien écrire.
- `--audio-only` : exporte uniquement la/les piste(s) corrigée(s) (`<sortie>.<fichier>.trackN.<ext>`, extension et codec selon la source — voir ci-dessus) au lieu de remuxer un MKV complet.
- `-o/--output` : chemin de sortie (défaut : `<INPUT>.synced.mkv`).
- `--only-imports` : n'inclut aucune piste de INPUT à part la référence (utile avec `--import-audio`/`--subs` ci-dessous, pour ne pas dupliquer une piste déjà présente dans INPUT).
- `--start`/`--duration` : comme pour `align`, limite la fenêtre utilisée pour la *détection* (le rendu, lui, s'applique toujours au fichier entier).

Par défaut, `render` suppose un décalage **constant** sur toute la piste — pas de dérive de vitesse ni de montage différent. Pour ces cas, voir `segments` (détection) et `render --segmented` (correction) plus bas.

### Injecter des pistes d'un autre fichier

Pour ajouter, resynchronisée, une piste venant d'un **second** fichier (ex. injecter l'audio + les sous-titres d'un mkv VF dans un mkv VO complet) :

```
uv run syncaudio render vo.mkv --reference 0 --only-imports \
  --import-audio vf.mkv@1 \
  --subs vf.mkv@2=vf.mkv@1
```

`--import-audio` (répétable) ajoute une piste audio d'un autre fichier, décalage détecté et corrigé automatiquement comme pour `--track`.

### Associer des sous-titres à une piste audio précise (`--subs`)

`--subs SPEC=AUDIO_SPEC` (répétable) ajoute une piste de sous-titres décalée **exactement comme** la piste audio `AUDIO_SPEC` désignée — utile quand on sait qu'une piste audio et ses sous-titres sont déjà synchro entre eux (même si mal synchro avec la référence), pour leur appliquer le même traitement plutôt que de deviner. `SPEC` peut venir de INPUT ou d'un autre fichier ; `AUDIO_SPEC` doit désigner exactement une piste déjà corrigée via `--track` ou `--import-audio` (même `fichier@index`) — sinon erreur claire. Plusieurs `--subs` peuvent pointer vers des pistes audio différentes.

Exemple avec une piste et ses sous-titres tous deux déjà dans INPUT :

```
uv run syncaudio render film.mkv --reference 0 --track 1 --subs film.mkv@2=film.mkv@1
```

Contrairement à l'audio, on ne peut pas « couper »/« combler » du texte : en mode non-segmenté, les horodatages sont simplement translatés du même montant que le décalage audio détecté ; en mode `--segmented`, ils sont individuellement réécrits selon le segment auquel chaque réplique appartient (voir plus bas).

## Détecter une dérive ou des sauts (`segments`)

`align` et `render` supposent un décalage **constant** sur toute la piste. `segments` lève cette hypothèse : il fait glisser une fenêtre d'analyse sur toute la piste et rapporte comment le décalage évolue dans le temps, sans rien corriger ni écrire :

```
uv run syncaudio segments film.mkv --reference 0 --track 1
```

Affiche un ou plusieurs segments, chacun avec un décalage de début/fin :
- même valeur aux deux bouts → décalage **constant** sur ce segment.
- valeurs différentes → **dérive** progressive sur ce segment (vitesse légèrement différente).
- plusieurs segments avec un saut net entre eux → montage différent à cet instant.

Quand plusieurs segments sont détectés, chaque frontière est automatiquement raffinée par une seconde passe locale (fenêtre bien plus petite, uniquement autour de la transition) pour la localiser plus précisément que la passe grossière seule.

Options : `--window`/`--hop` (taille/pas de la fenêtre glissante, secondes), `--margin` (décalage local max recherché par fenêtre), `--json` (sortie machine, fenêtres brutes + segments classifiés), `--start`/`--duration` comme pour `align`.

Pour corriger ce que `segments` a détecté (pas juste le visualiser), voir `render --segmented` ci-dessous.

Limites connues à ce stade :
- La précision de localisation d'un saut dépend de la richesse en musique/bruitages du contenu *juste autour* de la transition, pas seulement de la taille de fenêtre : sur une zone plutôt silencieuse/dialoguée à cet instant précis, même la passe de raffinement peut rester à plusieurs secondes de l'instant réel (vu sur `jump_single`, un cas par ailleurs propre — la correction finale reste malgré tout très bonne, voir ci-dessous).
- Sur des cas avec plusieurs sauts rapprochés, un segment isolé parasite peut occasionnellement apparaître près d'une transition (vu sur `jump_multi`).

## Corriger une dérive ou des sauts (`render --segmented`)

`render` sans `--segmented` suppose un décalage constant partout (voir plus haut). Avec `--segmented`, il détecte d'abord les segments (même pipeline que `segments`, raffinage compris), puis corrige **chaque segment indépendamment** :

```
uv run syncaudio render film.mkv --reference 0 --track 1 --segmented
```

Pour chaque segment, la portion correspondante de la piste candidate (son propre intervalle, décalé par l'offset de *ce* segment) est extraite, puis retimée avec `atempo` pour occuper exactement la durée du segment de référence — un facteur de 1 (aucun effet) quand le segment est à décalage constant, un facteur différent de 1 quand c'est une dérive : c'est la même formule dans les deux cas, pas un traitement séparé. Les segments corrigés sont ensuite concaténés bout à bout, ce qui reconstitue exactement la timeline de la référence. Options `--window`/`--hop`/`--margin` comme pour `segments`.

Sur nos fixtures de test : une dérive de +3.4s en fin de piste retombe à un résidu quasi constant (~0.2s) après correction ; un saut nettement détecté (même avec une frontière imprécise de quelques secondes) redonne un flux parfaitement synchro après correction, l'imprécision de frontière n'affectant qu'une poignée de secondes autour de la transition elle-même.

`--segmented` se combine aussi avec `--subs` : les horodatages de chaque réplique sont individuellement réécrits selon le segment auquel ils appartiennent (pas un simple décalage global comme en mode non-segmenté), donc une réplique après un saut ou en pleine dérive atterrit correctement. Formats de sous-titres supportés : SRT et ASS/SSA (les plus courants) ; un autre format donne une erreur claire plutôt qu'un résultat silencieusement faux.

### Accélérer sur de gros fichiers

Le temps d'extraction (ffmpeg) et surtout d'analyse spectrale (HPSS) croît avec la durée traitée. Le filtrage médian de cette analyse (l'essentiel du temps) est parallélisé sur tous les cœurs disponibles automatiquement (~7s → ~1.5s sur une piste de 6 min avec 8 cœurs) ; au-delà, comme l'algorithme cherche un décalage **constant**, il n'a pas besoin de toute la piste : un extrait représentatif suffit. `--start` et `--duration` (en secondes) limitent l'extraction/analyse à une fenêtre de chaque piste, ce qui accélère le traitement dans les mêmes proportions :

```
uv run syncaudio align film.mkv@0 vf.wav --start 300 --duration 600
```

N'analyse ici que les 10 minutes commençant à 5 minutes (utile pour sauter un générique/logo initial, souvent peu riche en musique/bruitages). Deux points à garder en tête :

- L'extrait doit contenir un minimum d'activité musique/bruitages (une scène uniquement silencieuse ou dialoguée donnera une confiance faible).
- Par défaut, seuls les décalages allant jusqu'à ~10 % de la durée de l'extrait sont recherchés (voir « Limites connues » ci-dessus) : avec `--duration 600`, un décalage réel de plus d'environ 60s ne sera pas trouvé. Si le décalage attendu est plus grand, augmentez `--duration` en conséquence.

## Sidecar HTTP (`serve`)

Expose le même moteur (`probe`/`align`/`segments`/`render`) en HTTP, pour un client autre que le CLI — c'est ce que l'app GUI (`app/`) lance et utilise en arrière-plan :

```
uv run syncaudio serve
```

Démarre sur `http://127.0.0.1:8756` par défaut (`--host`/`--port` pour changer). Docs interactives (Swagger) sur `/docs` une fois lancé — pratique pour explorer les endpoints à la main. `POST /render` reprend les mêmes options que la commande `render` (dont `segmented`, `import_audio`, `subs`) en JSON plutôt qu'en flags.

Deux façons d'appeler `align`/`segments`/`render` :
- **Direct** (`POST /align`, `POST /segments`, `POST /render`) : bloque jusqu'à la fin, simple pour un script ou une vérification rapide.
- **En job** (`POST /jobs/align`, `POST /jobs/segments`, `POST /jobs/render`) : retourne immédiatement un `job_id`, le traitement tourne en arrière-plan. `WS /jobs/{job_id}/ws` diffuse en direct les mêmes messages de progression que ceux affichés par le CLI (`[analyse] ...`), puis un message final `done` (avec le résultat) ou `error`. `GET /jobs/{job_id}` permet aussi d'interroger l'état à tout moment (utile en complément ou à la place de la WebSocket). C'est le mode à utiliser pour un GUI sur un vrai fichier (dizaines de secondes) : progression en direct plutôt qu'un bouton figé.

Endpoints synchrones complémentaires, pensés pour le GUI :
- `GET /probe?path=...` : liste les pistes audio (codec, langue, canaux, sample rate, délai de conteneur éventuel) — voir `probe` ci-dessus.
- `GET /waveform?path=...&index=...&start=...&duration=...&buckets=...` : enveloppe d'amplitude (min/max) sous-échantillonnée d'une fenêtre de piste, pour dessiner une forme d'onde sans envoyer l'audio brut (`duration` omis = piste entière).
- `GET /clip?path=...&index=...&start=...&duration=...` : court extrait audio jouable (WAV), pour une écoute avant/après correction — distinct de l'extraction d'analyse (mono 16 kHz), celui-ci garde le nombre de canaux et un sample rate normal.
- `POST /jobs/prefetch` : lance en arrière-plan, pour une liste de pistes, le calcul coûteux (extraction + spectrogramme/enveloppe) que `segments`/`render` referaient sinon à chaque appel — un résultat déjà en cache (même fichier, piste, fenêtre) est réutilisé tel quel. À appeler juste après `/probe` pour que le premier `segments`/`render` sur ces pistes soit quasi instantané.

## Développement

```
uv run pytest
```

`tests/test_align.py` valide l'algorithme sur des signaux synthétiques (sans ffmpeg). `tests/test_ffmpeg_backend.py` valide l'extraction/probe de bout en bout avec le ffmpeg embarqué. `tests/test_server.py` valide le sidecar HTTP avec `fastapi.testclient`.
