# SyncAudio (app)

GUI standalone de [SyncAudio](../README.md) (Tauri v2 + React/TypeScript) : ouvrir un fichier, choisir la piste de référence et les pistes à corriger, lancer l'analyse (décalage constant, dérive ou sauts), vérifier le résultat par un aperçu visuel (formes d'onde, style diff) et sonore avant d'exporter, éditer manuellement les segments détectés si besoin.

Ne contient aucune logique de détection/correction elle-même : elle pilote le moteur Python (`../engine/`), lancé au démarrage comme process séparé (« sidecar ») exposant une API HTTP + WebSocket locale (`127.0.0.1:8756`). Voir [`engine/README.md`](../engine/README.md) pour le détail du moteur et de son API.

## Prérequis

- [Node.js](https://nodejs.org/) (npm) pour le frontend.
- [Rust](https://www.rust-lang.org/tools/install) (via [rustup](https://rustup.rs/)) pour la coquille Tauri.
- [uv](https://docs.astral.sh/uv/), avec les dépendances du moteur synchronisées (`uv sync` dans `../engine/`) — le sidecar est lancé en dev via `uv run syncaudio serve`, donc `uv` doit être sur le PATH et `../engine` doit exister relativement à ce dossier.

## Développement

```
npm install
npm run tauri dev
```

Ouvre la fenêtre de l'app avec rechargement à chaud du frontend ; le sidecar moteur est démarré et arrêté automatiquement avec la fenêtre (voir `src-tauri/src/lib.rs`).

`npm run dev` (sans Tauri, juste `vite`) lance le frontend seul dans un navigateur — utile pour itérer vite sur l'UI, mais sans le sidecar ni les API natives (`@tauri-apps/*`), donc rien ne fonctionne au-delà de l'affichage statique.

## Build

```
npm run build
```

Vérifie les types (`tsc`) puis construit le frontend (`vite build`) dans `dist/`. Pour un exécutable installable, voir `npm run tauri build` — packaging encore en cours (voir le statut du projet dans le README racine) : à ce stade le sidecar est lancé depuis les sources via `uv run`, pas encore embarqué comme binaire autonome dans le paquet final.

## Structure

- `src/App.tsx` — écran principal : liste des pistes, lancement de l'analyse, onglets par piste analysée.
- `src/SegmentChart.tsx` / `SegmentEditor.tsx` — graphe décalage-vs-temps et éditeur manuel des segments (glisser une frontière, fusionner/supprimer, saisie numérique).
- `src/TrackPreview.tsx` — aperçu avant export : formes d'onde zoomables (référence / candidate / résultat, surlignage façon diff) et lecture audio synchronisée (Web Audio API, pour un démarrage simultané précis des pistes comparées).
- `src/api.ts` — client du sidecar HTTP/WebSocket.
- `src-tauri/` — coquille Rust : démarrage/arrêt du sidecar, config de la fenêtre (`tauri.conf.json`).
