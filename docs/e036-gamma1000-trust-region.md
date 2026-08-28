# E036 — γ=1000 perceptual trust-region SR-MPGD

## Question

E035 a validé la fidélité de la loss SRL officielle DiffQRCoder : elle réduit fortement
les erreurs QR et obtient des décodages QR-Verify, mais le pas latent non contraint
`z <- z - 1000 * grad` dégrade trop l'image.

E036 teste une seule idée : **ne pas réduire γ**. Le pas brut reste toujours calculé avec
`γ = 1000`, puis il est projeté/backtracké dans une région de confiance autour du parent
Stage 2 immuable.

## Parent et comparaisons

- Parent : `/data/e035-parent-v1`.
- Référence E035 : `/data/e035-loss-fidelity-gate-v1`.
- Sortie E036 : `/data/e036-gamma1000-trust-region-v1`.
- Même latent parent pour toutes les branches E036.
- VAE FP32, 4 itérations enregistrées, SRL officielle DiffQRCoder épinglée à
  `e24ea73ee2e13c7e6e87cb422e8b11784e70ae00`.
- LPIPS soft conservé à `0.01`.

## Branches

### `e036_gamma1000_global_trust`

- γ brut : 1000.
- rayon latent RMS : 0.050.
- budget LPIPS global : 0.050.
- budget MAE du core normalisé : 0.050.

### `e036_gamma1000_strict_trust`

- γ brut : 1000.
- rayon latent RMS : 0.025.
- budget LPIPS global : 0.020.
- budget MAE du core normalisé : 0.030.

### `e036_gamma1000_local_preserve`

- γ brut : 1000.
- rayon latent RMS : 0.050.
- budget LPIPS global : 0.050.
- budget MAE du core normalisé : 0.050.
- budget MAE hors modules QR actifs (+ voisinage d'un module) : 0.010.

Cette troisième branche teste directement l'hypothèse « corriger le QR sans modifier le
reste de l'image ».

## Règle d'update

À chaque itération :

1. calcul de la loss SRL officielle et de `0.01 * LPIPS` ;
2. rétropropagation jusqu'au latent ;
3. proposition brute `delta_raw = -1000 * grad` ;
4. projection de `z + delta_raw` dans la boule RMS autour du parent ;
5. backtracking `alpha = 1, 1/2, 1/4, ...` ;
6. acceptation du premier candidat satisfaisant tous les budgets et ne dégradant pas la
   SRL au-delà de la tolérance numérique ;
7. si la SRL officielle vaut déjà zéro, l'état est conservé au lieu de continuer à
   déplacer inutilement l'image.

Le paramètre γ n'est jamais remplacé par `alpha * gamma` dans le calcul du pas brut :
`alpha` fait partie de la projection/acceptation de la région de confiance.

## Comparaison visuelle obligatoire

Le notebook `31_e036_gamma1000_trust_region.ipynb` affiche :

1. parent FP32 ;
2. E035 paper ;
3. E035 upstream non contraint ;
4. E036 global trust ;
5. E036 strict trust ;
6. E036 local preserve.

Il affiche également `e036-final-contact-sheet.png`, les images finales individuelles en
736 px, les traces de chaque update et QR-Verify.

## Décision

Aucune branche E036 n'est production-ready automatiquement. Le verdict peut seulement
identifier une branche de recherche gagnante et préparer un mini-holdout ultérieur si
elle obtient au moins un preset QR-Verify exact tout en restant dans ses contraintes.
