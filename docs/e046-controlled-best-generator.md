# E046 — Controlled Best-Generator Dataset

## But

E046 produit un nouveau corpus de QR esthétiques avec provenance complète. Il ne
réutilise pas en masse les 306 372 PNG génériques ignorés par E045.

La campagne sépare volontairement les calculs coûteux et le scoring :

```text
plan CPU
  ↓
parents Stage 1 + Stage 2 GPU
  ↓
scoring CPU WeChat + qualité
  ↓
sélection diverse
  ↓
SR-MPGD GPU par parent/recette
  ↓
scoring CPU de tous les checkpoints
  ↓
dataset + Pareto + notebooks
```

Si HPS ou QR-Verify échoue, les générations GPU déjà promues restent intactes.
Si un Job GPU échoue, sa tentative reste sous `attempts/` et la relance saute
toutes les tâches possédant déjà `GENERATION_COMPLETE.json`.

## Sélection automatique : valide d'abord, beau ensuite

E046 ne réduit plus le problème à un score unique. Pour chaque prompt, plusieurs
QR sont générés avec des seeds, masques, paramètres Stage 1/Stage 2 et recettes
SR-MPGD différents.

La sélection automatique se fait en deux niveaux :

```text
1. GATE DE VALIDITÉ LOGICIELLE
   - garde visuelle passée
   - raster brut uniquement
   - payload exact sur le preset original WeChat
   - au moins 34/37 presets exacts

2. CLASSEMENT MULTIOBJECTIF DANS LE PALIER VALIDE
   - 40 % robustesse WeChat /37
   - 25 % CLIPScore : respect du prompt
   - 20 % HPSv2 : préférence visuelle
   - 15 % CLIP-Aesthetic : esthétique globale
   - MER croissant utilisé en départage/diagnostic
```

Un QR très beau mais invalide ne peut jamais gagner. À l'inverse, parmi les QR
valides, celui qui correspond le mieux au prompt et obtient les meilleurs scores
visuels gagne automatiquement. Aucune validation manuelle n'intervient dans la
sélection.

Le pont du projet utilise `qr-scanner-wechat` via `qr-verify@0.2.0`, avec payload
exact, 37 presets et trois répétitions conservatrices. OpenCV, ZBar et ZXing
restent des diagnostics et ne votent pas.

La recette `m0_balanced_public`, qui a donné le premier parent smoke à 36/37, est
réutilisée comme ancre pour **chaque prompt**, puis comparée à des recettes plus
scan-oriented, paper-noise et aesthetic-oriented.

## Profils

### smoke tournoi

- 2 prompts ;
- 3 générations Stage1/Stage2 par prompt ;
- 6 parents au total ;
- 1 parent sélectionné par prompt ;
- 2 trajectoires SR-MPGD ;
- valide la logique de meilleur QR **par prompt**.

### pilot — campagne réelle

- 8 prompts ;
- 6 recettes Stage1/Stage2 par prompt ;
- 48 parents ;
- 2 parents sélectionnés par prompt ;
- 3 recettes SR-MPGD par parent ;
- 48 trajectoires SR-MPGD ;
- un `FINAL-QR.png` automatique par prompt si le gate WeChat est satisfait.

### full — recherche profonde

- 8 prompts ;
- 8 recettes/masques × 2 seeds par prompt ;
- 128 parents ;
- 2 parents sélectionnés par prompt ;
- 4 recettes SR-MPGD par parent ;
- 64 trajectoires SR-MPGD.

Le profil est intégré au hash du plan. Une relance doit réutiliser le même
`PROOFTAG_E046_PROFILE`.

## Sorties finales par prompt

```text
pipeline/by-prompt/<prompt-id>/FINAL-QR.png
pipeline/by-prompt/<prompt-id>/FINAL-metadata.json
```

Si aucun candidat ne satisfait automatiquement le gate de validité, E046 écrit
`NO-VALID-FINAL.json` pour ce prompt au lieu de présenter un QR invalide comme
un résultat final. Le verdict liste alors les prompts à approfondir.

## Espace couvert

Le pilot couvre huit familles visuelles, six recettes distinctes par prompt,
les initialisations `public_random` et `paper_stage1_noise`, ECC M/Q, plusieurs
masques, paramètres Stage1/Stage2 et trois régimes SR-MPGD (gamma 250/500/1000,
rayons 0.10/0.15/0.20, 4 ou 8 itérations).

## Quiet zone sans effacement uniforme

E046 interdit comme sortie finale :

```text
white
adaptive_light uniforme
repaint fonctionnel
projection pixel binaire
```

Pour chaque état, il garde :

1. `raw` : raster intégral de diffusion ;
2. `scene_preserving` : même canvas, cœur 580×580 octet-identique, périphérie
   obtenue par lissage/désaturation/éclaircissement de l'œuvre elle-même.

La composition ne coupe rien et ne remplace pas toute la périphérie par une
couleur plate. Chaque variante possède des métriques de luminance, texture,
dark ratio et hash du cœur.

## Arborescence

```text
/data/e046-controlled-best-generator-v1/
├── LATEST.json
└── <plan-id>/
    ├── plan.json
    ├── catalog.json
    ├── parents/<candidate-id>/
    │   ├── images/
    │   ├── stage2-latent.safetensors
    │   ├── scoring/
    │   ├── GENERATION_COMPLETE.json
    │   └── SCORING_COMPLETE.json
    ├── refinements/<candidate-id>/<recipe-id>/
    │   ├── trajectory/<recipe-id>/images/
    │   ├── trajectory/<recipe-id>/latents/
    │   ├── scene-qz/
    │   ├── scoring/
    │   ├── GENERATION_COMPLETE.json
    │   └── SCORING_COMPLETE.json
    ├── attempts/
    ├── failures/
    ├── dataset/
    ├── pipeline/
    ├── verdict.json
    ├── artifact-manifest.json
    └── COMPLETE.json
```

## Classement et Pareto

Le classement est calculé séparément pour chaque prompt. La validité WeChat est
un gate dur, puis les métriques WeChat, CLIPScore, HPSv2 et CLIP-Aesthetic sont
normalisées dans le prompt et combinées selon les poids enregistrés dans
`plan.json`. Le front de Pareto ne contient que des candidats logiciels valides.

Stage1 et les variantes `scene_preserving` restent dans le dataset pour l'analyse,
mais ne peuvent pas devenir des sorties finales. Seuls `stage2_raw` et les
checkpoints `srmpgd_raw` sont éligibles.

Aucun gagnant n'est déclaré `production_ready`; cette mention est distincte de la
validité automatique WeChat utilisée ici.

## Échantillon téléphone

L'agrégation crée également :

```text
dataset/phone-sample-pending.csv
dataset/phone-sample-pending.json
pipeline/phone-sample-contact-sheet.png
```

Le lot contient les meilleurs candidats par prompt, le Pareto et quelques hard
negatives esthétiques. Les champs téléphone restent à zéro avec le statut
`pending_physical_capture` : aucun échec physique n'est inventé.

## Préparation E047

E046 fournit :

- un CSV ;
- un JSON ;
- un JSONL ;
- hashes de pixels ;
- configuration complète ;
- provenance commit/image/modèles ;
- labels WeChat ;
- qualité ;
- erreurs techniques ;
- doublons et no-op.

Le verdict expose `software_advisor_training_candidate`, mais laisse
`automatic_advisor_training_authorized=false`. Les splits E047 doivent être
gelés après revue du notebook.
