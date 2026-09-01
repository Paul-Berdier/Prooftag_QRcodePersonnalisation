# E042 — Decoder failure localization

## Pourquoi

E041 a réduit fortement les erreurs de modules mesurées sans obtenir de décodage QR-Verify. E042 ne cherche donc pas un nouveau `gamma`. Elle localise le point de rupture entre la représentation différentiable et un vrai décodeur.

## Données réutilisées

E042 réutilise **uniquement** neuf latents E041 Phase A :

- parent `gamma=50, i0` ;
- `gamma=500` aux checkpoints `i1, i2, i4, i8` ;
- `gamma=1000` aux checkpoints `i2, i3` ;
- `gamma=2000` aux checkpoints `i2, i4`.

Aucun Stage 1, Stage 2 ou SR-MPGD n'est recalculé.

## Phase 1 — VAE re-decode

La phase GPU ne fait qu'un re-decode VAE des neuf latents pour récupérer le raster **brut 736×736**, avant `prepare_scan_ready_image`.

Pour chaque état :

1. raster E041 scan-ready actuel ;
2. raster VAE brut ;
3. quiet zone exacte sur `padding=78`, couleur adaptive ;
4. quiet zone exacte blanche.

Cela permet de mesurer si l'ancien traitement proportionnel 37 cellules a écrasé une partie du cœur exact `29×29×20 = 580 px`.

## Phase 2 — diagnostic CPU

Pour chaque état :

- OpenCV `detect` séparé de `detectAndDecode` ;
- OpenCV, zbar, ZXing-C++, WeChat direct ;
- QR-Verify 37 presets, **one-shot diagnostic** ;
- Otsu global, adaptive threshold, blur+Otsu ;
- sampling de la grille exacte 29×29 ;
- reconstruction canonique depuis la moyenne cellule ;
- reconstruction canonique depuis le centre cellule ;
- recherche target-assisted du meilleur seuil global, uniquement pour diagnostic ;
- erreurs par finder, séparateurs, timing, alignment, format info, fixed dark module et data ;
- variance intra-module et marges au seuil.

Un QR binaire exact est d'abord exigé à `37/37` sur trois répétitions QR-Verify. Sinon E042 échoue fermée.

## Interprétation

- **exact quiet-zone rescue** : géométrie/marge extérieure prioritaire ;
- **Otsu/adaptive rescue** : texture/binarisation prioritaire ;
- **grid reconstruction rescue** : bits présents, mais grille/texture difficile à récupérer ;
- **pas de rescue + erreurs format** : pondérer fortement les bits de format ;
- **pas de rescue + erreurs data** : loss bits/data plus proche de l'ECC.

Les reconstructions target-assisted ne sont jamais des candidats production. Elles ne servent qu'à localiser le verrou scientifique.

## Sortie

`/data/e042-decoder-failure-localization-v1`

Le verdict final reste :

- `diagnostic_only = true` ;
- `production_ready = false` ;
- `generalization_authorized = false`.
