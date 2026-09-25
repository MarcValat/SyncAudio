# SyncAudio

Resynchronise les pistes audio d'un contenu multi-langues (ex. un MKV avec plusieurs doublages) en se basant sur la musique et les bruitages communs, plutôt que sur les dialogues. Gère aussi bien un simple décalage constant qu'une dérive progressive ou des sauts (montage différent), avec aperçu visuel et sonore avant export.

## Structure du repo

- [`engine/`](engine/README.md) — moteur Python (détection, correction/rendu, sidecar HTTP) ; utilisable seul en CLI ou comme serveur local pour l'app.
- `app/` — GUI standalone (Tauri + React/TypeScript) : ouverture de fichier, sélection des pistes, aperçu waveform/audio avant export, édition manuelle des segments détectés.

## Statut

En cours de développement par phases. Le moteur (détection constante/dérive/sauts, rendu, sidecar HTTP) et l'essentiel du GUI (analyse, aperçu, édition, export) sont fonctionnels ; empaquetage/installateur et mode batch multi-fichiers restent à venir. Voir `engine/README.md` pour l'usage détaillé du CLI et de l'API HTTP.
