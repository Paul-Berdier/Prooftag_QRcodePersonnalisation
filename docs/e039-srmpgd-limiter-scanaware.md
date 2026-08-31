# E039 — SR-MPGD limiter diagnostic + scan-aware optimization

## But

E039 reste volontairement mono-parent. Il part du gagnant E038 `e038_hybrid_r150` et cherche à augmenter le SSR sans relâcher gamma ni la garde esthétique.

## Invariants

- même parent figé E035/E036/E038 ;
- gamma = **1000** ;
- LPIPS weight = **0.01** ;
- VAE FP32 pour la trajectoire ;
- QR-Verify 37 presets ;
- pas de holdout ni d'autorisation de généralisation.

## Grille

### Exact objectif E038 hybrid, rayon 0.150

- 4 updates (contrôle reproduit)
- 6 updates
- 8 updates
- 12 updates

### Scan-aware v2

Le profil v2 conserve tous les termes robustes E038 et ajoute :

- blur 3x3 appliqué deux fois ;
- downscale 50% puis restauration ;
- luminosité ±15% ;
- contraste ±15%.

Recettes :

- r=.150, 4 updates
- r=.150, 6 updates
- r=.150, 8 updates
- r=.200, 8 updates
- r=.300, 8 updates
- r=.300, 12 updates

## Diagnostic du verrou

Chaque tentative de backtracking enregistre :

- alpha ;
- objective ;
- SRL upstream ;
- full-module loss ;
- robust loss ;
- LPIPS ;
- core MAE ;
- latent RMS ;
- checks `latent_radius`, `lpips_budget`, `core_mae_budget`, `objective_nonincrease` ;
- raison(s) de rejet.

Le fichier `blocker-summary.csv/json` agrège ces informations et indique le `dominant_blocker` par recette.

## Sélection

1. garde esthétique PASS ;
2. SSR QR-Verify maximal ;
3. `original_exact=True` ;
4. erreurs modules minimales ;
5. LPIPS minimal.

E039 ne rend rien production-ready et n'autorise pas de holdout automatique.
