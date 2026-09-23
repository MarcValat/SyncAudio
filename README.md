# SyncAudio

Resynchronise les pistes audio d'un contenu multi-langues (ex. un MKV avec plusieurs doublages) en se basant sur la musique et les bruitages communs, plutôt que sur les dialogues.

## Structure du repo

- [`engine/`](engine/README.md) — moteur Python (détection de décalage, et bientôt rendu/remux) ; utilisable seul en CLI.
- `app/` — GUI standalone (à venir).

## Statut

En cours de développement par phases. Voir `engine/README.md` pour l'usage actuel du CLI.
