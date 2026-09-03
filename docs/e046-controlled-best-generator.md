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

## Source de vérité QR logicielle

Le pont du projet utilise :

```javascript
import { scan } from "qr-scanner-wechat"
```

dans `qr_verify_bridge/bridge.mjs`.

Le label principal est donc uniquement :

```text
wechat_exact_presets / 37
wechat_exact_rate
wechat_original_exact
```

Le payload doit être exact. Les autres décodeurs ne contribuent pas au score
principal. Le téléphone reste la vérité produit finale et n'est pas considéré
comme disponible dans E046.

## Profils

### smoke

- 2 parents ;
- 1 parent sélectionné ;
- 1 recette SR-MPGD ;
- à lancer obligatoirement avant `pilot`.

### pilot

- 8 prompts ;
- 8 masques QR ;
- 8 parents ;
- 4 parents sélectionnés ;
- 3 recettes SR-MPGD ;
- 12 trajectoires SR-MPGD ;
- tous les checkpoints conservés.

### full

- 8 prompts × 3 formulations = 24 parents ;
- 8 parents sélectionnés ;
- 4 recettes SR-MPGD ;
- 32 trajectoires.

Le profil est intégré au hash du plan. Une relance doit réutiliser le même
`PROOFTAG_E046_PROFILE`.

## Espace couvert

Le pilot couvre :

- huit familles visuelles ;
- huit masques QR légaux ;
- ECC M et Q ;
- plusieurs seeds ;
- Stage 1 : steps, CFG, ControlNet, control start/end ;
- Stage 2 : `public_random` et `paper_stage1_noise`, strength, steps,
  ControlNet scale, poids SRG et perceptuel ;
- SR-MPGD : gamma 250/500/1000, rayons .10/.15/.20, 4 ou 8 itérations,
  projection et backtracking ;
- brut vs composition périphérique `scene_preserving`.

La première campagne verrouille version 3, modules de 20 px, padding 78 px et
canvas 736 px pour ne pas mélanger une modification géométrique avec la recherche
de paramètres. Ces axes seront ouverts dans une campagne ultérieure une fois le
dataset E046 audité.

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

## Classement

On filtre d'abord les gardes visuelles. Puis le classement maximise :

1. WeChat exact / 37 ;
2. exactitude du preset original ;
3. raster brut si le score WeChat est identique ;
4. sécurité de la composition périphérique ;
5. CLIP-Aesthetic ;
6. HPSv2 ;
7. CLIPScore ;
8. baisse de MER en diagnostic.

Le front de Pareto est également exporté. Aucun gagnant n'est déclaré
`production_ready`.

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
