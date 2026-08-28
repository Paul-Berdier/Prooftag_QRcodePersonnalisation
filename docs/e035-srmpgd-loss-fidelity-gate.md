# E035 — SR-MPGD loss fidelity gate

## Statut

**Implémentation source complète ; exécution GPU non réalisée dans le paquet.** E035 est
bloquante tant qu’un parent Stage 2 canonique n’est pas disponible sous forme de PNG,
latent `safetensors` et contrat de provenance vérifié.

L’archive E034 fournie contient les images et les hashes, mais pas le tenseur latent.
Le code ne tente jamais de reconstruire ce latent depuis le PNG. Deux origines de parent
sont autorisées :

1. export exact du latent Stage 2 E034 encore présent sur le PVC ;
2. nouveau Stage 2 capturé une seule fois depuis le PNG Stage 1 exact observé dans E034.

La deuxième voie reste une comparaison appariée valide, mais n’est pas une reproduction
bit-à-bit du latent E034.

## Hypothèse

E034 a montré que `paper_v3` peut atteindre zéro erreur centrale à 0,5 tout en gardant
des modules erronés sur leur surface complète et zéro décodage QR-Verify. Le code public
DiffQRCoder épinglé utilise une condition plus exigeante :

- cible noire active tant que la moyenne du centre est `> 0.45` ;
- cible blanche active tant que la moyenne du centre est `< 0.65` ;
- centre `6:14 × 6:14`, soit 8×8 pixels pour un module de 20 pixels ;
- niveaux de gris `0.2999 R + 0.587 G + 0.1114 B` ;
- noyau `cv2.getGaussianKernel(20, 1.5)`, produit extérieur, min-max, puis valeurs
  `< 0.1` mises à zéro, sans renormalisation finale ;
- convolution de stride 20 produisant une somme pondérée par module ;
- aucune pondération additionnelle des motifs fonctionnels.

E035 teste si cette loss retarde l’arrêt prématuré et améliore la décodabilité réelle.

## Révision upstream

```text
e24ea73ee2e13c7e6e87cb422e8b11784e70ae00
```

Le runner charge réellement :

```python
from diffqrcoder.losses.scanning_robust_loss import ScanningRobustLoss
```

La branche upstream utilise cette classe pour son gradient. Une implémentation locale
vectorisée calcule en parallèle la même valeur. Le Job échoue immédiatement si les deux
valeurs ne passent pas `torch.allclose(atol=2e-6, rtol=2e-6)` pendant l’évaluation ou le
gradient.

## Invariants appariés

Les deux branches utilisent :

- le même contrat parent, le même tenseur source et la même conversion FP32 ;
- le même VAE en FP32 ;
- la même référence LPIPS flottante `x0 = D(z0)` ;
- LPIPS VGG sur CPU, poids `0.01` ;
- quatre mises à jour imposées, sortie `i4` ;
- `gamma = 1000` ;
- loss scaling `32768`, puis division exacte du gradient ;
- crop `78 px`, cœur 580×580, QR v3 29×29, modules 20 px ;
- quiet zone `adaptive_light` et aucun renforcement des motifs fonctionnels ;
- QR-Verify conservateur, 37 presets et trois répétitions ;
- aucune sélection opportuniste du meilleur état ;
- aucune campagne multi-prompt et aucun entraînement du conseiller.

Le raster source du parent et le raster FP32 `D(z0)` sont archivés séparément. Le fichier
`branch-pairing.json` exige le même hash latent `i0` et le même hash du raster FP32 `i0`
pour les deux branches.

## Branches

### `e035_paper_srl_control`

Contrôle équationnel déjà utilisé par E034 : centre d’un tiers, seuils 0,5/0,5, Gaussian
normalisé par module et `functional_weight=1.0`.

### `e035_upstream_code_srl`

Classe officielle `ScanningRobustLoss` du commit épinglé, centre 8×8 et marges 0,45/0,65.

## Parent immuable

Répertoire par défaut :

```text
/data/e035-parent-v1/
├── parent-stage2.png
├── parent-stage2-latent.safetensors
└── parent-stage2-metadata.json
```

Le contrat lie :

- SHA-256 du fichier PNG et SHA-256 des pixels RGB ;
- SHA-256 du fichier `safetensors` et SHA-256 du tenseur ;
- forme, dtype, clé du tenseur et device source ;
- scaling factor du VAE ;
- modèle, révisions, QR, payload, run, méthode et commit source.

Les noms de fichiers sont canoniques et toute tentative de traversal est refusée.
Aucun `.pt` ou pickle n’est chargé.

Vérification :

```bash
python -m prooftag_qr.e035_parent_artifact /data/e035-parent-v1
```

Export d’un latent E034 existant :

```bash
python scripts/export_e035_parent_artifact.py \
  --image /chemin/parent-stage2.png \
  --latent /chemin/parent-stage2-latent.safetensors \
  --source-json docs/e035-parent-source-template.json \
  --output-dir /data/e035-parent-v1
```

## Capture fallback Stage 2 uniquement

Le paquet contient le PNG exact :

```text
docs/e035-assets/e034-observed-stage1.png
```

Empreintes :

```text
fichier : be2ed76a2d4e3157beb3e3165a4041123ecc05b0f21d8be8c728e9f2fd12fb71
pixels  : ce7066664a9d3fee982841ce30f7fbdf442e4d601818187ed05d0f1301296079
format  : RGB 736×736
```

Le Job `e035-parent-capture-job.yaml` charge ce fichier depuis le PVC, vérifie les deux
hashes, appelle directement `_run_stage2(...)`, désactive SR-MPGD, exporte immédiatement
`backend.export_stage2_state()` et écrit le contrat. Aucun Stage 1 n’est exécuté.

## Déploiement

Après commit/push local puis pull sur le serveur :

```bash
bash scripts/deploy-e035-notebook.sh all
```

Contrôle manuel :

```bash
bash scripts/deploy-e035-notebook.sh prepare
bash scripts/deploy-e035-notebook.sh verify-input
bash scripts/deploy-e035-notebook.sh capture-parent   # seulement si nécessaire
bash scripts/deploy-e035-notebook.sh verify-parent
bash scripts/deploy-e035-notebook.sh run
bash scripts/deploy-e035-notebook.sh status
bash scripts/deploy-e035-notebook.sh logs
bash scripts/deploy-e035-notebook.sh download
bash scripts/deploy-e035-notebook.sh restore
```

Le script libère temporairement le GPU occupé par l’API, lance un Job exclusif, puis
restaure le nombre de répliques précédent même en cas d’échec.

## Sorties

```text
/data/e035-loss-fidelity-gate-v1/
├── plan.json
├── parent-verification.json
├── parent-module-diagnostics.json
├── parent-fp32-redecoded.png
├── branch-pairing.json
├── runtime.json
├── e035_paper_srl_control/
│   ├── images/iteration-000.png ... iteration-004.png
│   ├── diagnostic-maps/
│   ├── trace.json
│   ├── trace.csv
│   ├── final-latent.safetensors
│   └── branch-result.json
├── e035_upstream_code_srl/
├── qr-verify-evidence.json
├── e035-final-contact-sheet.png
├── verdict.json
└── report.md
```

Chaque itération conserve les deux SRL, LPIPS, objectif, compte d’erreurs centrales,
modules actifs upstream, erreurs de moyenne complète, séparation fonctionnelle/données,
quantiles de luminance, gradients, pas demandé/appliqué, delta latent, hashes, mémoire
CUDA et parité locale/officielle.

## Gate gradient corrigé

Le gradient SRL seul n’est requis que si la SRL sélectionnée est supérieure à la
tolérance. Quand `SRL = 0`, un gradient SRL nul est légitime si LPIPS produit encore un
gradient d’objectif, un gradient latent et un pas appliqué non nuls.

## Décision

- au moins un preset exact upstream et garde visuelle passée :
  `GO_MINI_HOLDOUT_8_TO_12_PROMPTS` ;
- aucun décodage mais amélioration des marges ou du MER complet :
  `E036_HYBRID_CENTER_PLUS_FULL_MODULE_LOSS` ;
- aucune amélioration : `STOP_AND_DIAGNOSE_DETECTOR_GEOMETRY`.

Dans tous les cas :

```text
production_ready=false
automatic_expansion_authorized=false
advisor_training_authorized=false
```
