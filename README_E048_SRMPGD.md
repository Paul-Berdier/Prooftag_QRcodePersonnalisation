# E048 — SR-MPGD nocturne sur les 36 Stage2 E047

Version de release : `1.1.0-e048-canary`.

## Objectif

E048 ne régénère ni Stage1 ni Stage2. Elle prend comme source le run E047 terminé
`qrn-20260917-075337`, monte ses artefacts en **lecture seule**, charge les 36
`stage2-latent.safetensors`, puis cherche une amélioration SR-MPGD de la robustesse QR.

La priorité de sélection est :

1. QR strict `37/37` + payload exact sur l'image originale ;
2. nombre de presets exacts ;
3. décodage original exact ;
4. à égalité QR, modification visuelle minimale (LPIPS / MAE / delta latent).

Le Stage2 brut reste toujours candidat. Une optimisation qui n'améliore pas le QR ne peut donc
pas remplacer arbitrairement l'image source.

## Stratégie SR-MPGD

Les quatre premières branches reprennent les zones de paramètres déjà présentes dans le catalogue
E046 :

- `catalog_g250_r100_i04` ;
- `catalog_g500_r200_i08` ;
- `catalog_g1000_r150_i08` ;
- `catalog_g500_r150_i08_visual`.

Si aucun 37/37 n'est trouvé, trois branches plus longues utilisent le même moteur SR-MPGD gardé :

- `extended_g500_i16` ;
- `extended_g1000_i24` ;
- `robust_i32` (pertes robustes blur/downscale/brightness/contrast).

Chaque itération reçoit un vrai QR-Verify. Les gardes visuels du moteur `guarded_production`
restent actifs. Un 37/37 arrête les profils suivants pour le candidat concerné.

## Garde-fous opérationnels

- le cœur E047/E046 est réutilisé sans modification ;
- les 36 sources E047 sont montées read-only ;
- le commit E047 source doit être exactement `7de284647bc526d1b206e2be21614d724f51e29e` ;
- les blobs Git des modules scientifiques SR-MPGD sont vérifiés dans le conteneur ;
- `prepare` effectue un préflight CPU réel, vérifie les 36 latents, les imports et l'API SR-MPGD ;
- vLLM reste actif pendant `prepare` ;
- au démarrage nocturne, le snapshot vLLM est pris puis le gardien indépendant est déjà actif ;
- après la coupure de vLLM, **un canary GPU réel d'une itération** teste DiffQRCoder + latent
  + SR-MPGD + LPIPS + QR-Verify ;
- si le canary échoue, la campagne s'arrête et la restauration vLLM est engagée ;
- le Job long est borné par la deadline ;
- 30 minutes sont réservées au rapport et 1 heure entière à la restauration de production ;
- le template/UID du déploiement vLLM est vérifié avant toute remise à l'échelle ;
- la restauration valide `/health`, `/v1/models` et une petite inférence ;
- aucune promotion automatique en production.

## Fenêtre recommandée cette nuit

```text
Départ calcul : 2026-09-17 22:00 Europe/Paris
Arrêt calcul  : 2026-09-18 07:00 Europe/Paris
Objectif vLLM : 2026-09-18 08:00 Europe/Paris
```

Le pipeline peut finir plus tôt. S'il atteint la deadline, les résultats déjà écrits sont conservés
et la restauration de vLLM est prioritaire.

## Résultats

Les sorties sont écrites hors du run E047 :

```text
/home/paul/qr-night-runs/e048-.../artifacts/
```

On y trouve notamment :

- `GPU_CANARY_PASS.json` ;
- `progress.json` ;
- `optimized/<task>/best.png` ;
- `optimized/<task>/result.json` ;
- profils/checkpoints et latents intermédiaires ;
- `report/comparison.csv` ;
- `report/summary.json` ;
- `report/contact-sheet.png` ;
- GIF des meilleurs candidats.

L'export final exclut les gros latents et previews, mais conserve les preuves JSON, meilleurs PNG,
comparaison, rapport et GIF.
