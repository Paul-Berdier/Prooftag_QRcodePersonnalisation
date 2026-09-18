# E048 V4 — SR-MPGD sur les 36 Stage2 E047

**Révision du pack : `v4-cross-platform-canary-tests`**

V4 conserve **strictement le chemin scientifique V3 E039/E040** et corrige uniquement le harness de tests poste développeur : les tests de marqueurs du canary utilisent désormais un writer JSON portable au lieu de déclencher `qrnight.common.write`, dont la garantie d'atomicité `O_DIRECTORY` est volontairement POSIX/Linux. Le code de production continue d'utiliser le writer durable Linux dans les Jobs pcIA.

Le script `scripts/e048-install-pc.ps1` sait aussi réparer un overlay V3 échoué **uniquement si les fichiers sales sont tous des fichiers E048 gérés par le pack** ; il refuse toute autre modification locale.

# E048 — SR-MPGD adaptatif sur les 36 Stage2 E047

Version de release : `1.2.0-e048-e040-canary`.
Pack : `v3-e040-validated-trajectory`.

## Pourquoi cette V3 existe

La V2 a correctement bloqué la campagne longue au canary GPU, mais son worker mélangeait deux
chemins SR-MPGD différents : le helper générique `run_srmpgd` et la politique d'offload issue des
campagnes E035/E046. Cette combinaison n'était pas le chemin déjà validé historiquement sur le
projet.

La V3 supprime ce mélange. Elle réutilise directement le moteur de trajectoire déjà employé par
E046/E040 : `e040_checkpoint_frontier._run_trajectory`, avec `E039Config`, l'offload E035, VAE en
float32, SRL upstream officielle et checkpoints persistés.

Aucun Stage1/Stage2 E047 n'est régénéré. Les 36 artefacts source restent montés en lecture seule.

## Objectif

Pour chaque sortie E047 :

1. conserver le Stage2 brut comme baseline ;
2. charger son latent Stage2 original ;
3. appliquer plusieurs trajectoires SR-MPGD bornées ;
4. scorer **chaque checkpoint** avec le vrai QR-Verify ;
5. arrêter les profils suivants dès qu'un checkpoint strict `37/37` + payload exact est retenu ;
6. à robustesse QR égale, préférer le checkpoint qui modifie le moins l'image.

Ordre de priorité :

1. `37/37` + payload original exact ;
2. nombre de presets exacts ;
3. décodage original exact ;
4. LPIPS / changement moyen / delta latent minimum.

## Profils

Les quatre premières zones reprennent les recettes E046 :

- `catalog_g250_r100_i04` ;
- `catalog_g500_r200_i08` ;
- `catalog_g1000_r150_i08` ;
- `catalog_g500_r150_i08_visual`.

Puis, si nécessaire :

- `extended_g500_i16` ;
- `extended_g1000_i24` ;
- `robust_i32`.

Les paramètres V3 correspondent directement au moteur E039/E040 : `gamma`, `latent_radius_rms`,
`max_iterations`, `lpips_weight`, `lpips_budget`, `core_mae_budget`, `full_module_weight` et
`max_backtracks`.

## Garde-fous

- source E047 `qrn-20260917-075337` obligatoire ;
- commit scientifique E047 attendu : `7de284647bc526d1b206e2be21614d724f51e29e` ;
- 36 images + 36 latents + 36 résultats requis ;
- source E047 montée read-only ;
- blobs Git des dépendances SR-MPGD/E038/E039/E040 vérifiés dans le worker ;
- image Docker additive, aucun `pip install` au lancement ;
- `TORCH_HOME=/cache/torch` explicite ;
- préflight CPU avant toute coupure vLLM ;
- snapshot et health-check vLLM avant coupure ;
- canary GPU réel avant la campagne longue ;
- si le canary échoue, `GPU_CANARY_FAILED.json` contient le traceback et le run long ne démarre pas ;
- guard systemd indépendant toutes les 60 s ;
- deadline dure ;
- restauration vLLM contrôlée par `/health`, `/v1/models` et petite inférence ;
- aucune promotion automatique en production.

## Canary V3

Le canary exécute réellement :

`Stage2 latent E047 -> pipeline DiffQRCoder -> offload E035 -> VAE float32 -> E040/E039 SR-MPGD -> checkpoint -> QR-Verify`.

Le Job long n'est autorisé que si `GPU_CANARY_PASS.json` est écrit.

## Résultats

Sous :

`/home/paul/qr-night-runs/e048-.../artifacts/`

Principaux fichiers :

- `GPU_CANARY_PASS.json` ou `GPU_CANARY_FAILED.json` ;
- `progress.json` ;
- `optimized/<task>/best.png` ;
- `optimized/<task>/result.json` ;
- `optimized/<task>/profiles/<profil>/result.json` ;
- `optimized/<task>/profiles/<profil>/checkpoints.json` ;
- trajectoires complètes et latents conservés sur le serveur ;
- `report/comparison.csv` ;
- `report/summary.json` ;
- `report/contact-sheet.png` ;
- GIF des meilleurs candidats.

L'export final exclut les gros latents, trajectoires complètes et caches, mais garde les meilleurs
PNG, les JSON de décision, les tableaux et les GIF.

## Validation hors GPU

Le pack V3 passe :

- 41 tests E048 ;
- 82 tests `nightops` au total avec la base E047 ;
- compilation Python de `e048_host.py`, `e048_worker.py` et `test_e048.py` ;
- vérification du manifeste SHA-256 lors de l'installation.

Ces tests ne remplacent pas le canary CUDA réel. Le canary existe précisément pour fermer le
risque restant lié à la RTX, CUDA, aux caches modèles et au runtime K3s de pcIA.
