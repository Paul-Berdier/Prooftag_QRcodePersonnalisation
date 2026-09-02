# E043 — scanner-cell SR-MPGD frontier

## Point de départ

E042 localise le verrou principal à `GRID_DETECTION_OR_INTRA_MODULE_TEXTURE` :

- `grid_reconstruction_rescue_count = 7/9`;
- `quiet_zone_rescue_count = 0`;
- `binarization_rescue_count = 0`;
- l'ancienne quiet zone empiète néanmoins sur le cœur exact dans 9/9 états.

E043 ne généralise pas et ne change pas le prompt. Il reste apparié à E041.

## Contrôle et recettes

Toutes les branches utilisent `gamma=500`, rayon latent `0.20`, checkpoints `i0..i8`.

- **A** : contrôle E041 `gamma=500` ; les 9 latents existants sont seulement re-décodés par le VAE.
- **B** : A + `whole_cell_margin` + `intra_module_variance_penalty`.
- **C** : B + `grid_consistency`.
- **D** : C + `format_information_weighted_margin` + `data_module_margin_with_ecc_awareness`.

La composante ECC est un **proxy de risque** : les data modules proches du seuil sont davantage pondérés. Elle n'est pas un décodeur Reed-Solomon différentiable.

## Quiet zone

E043 n'utilise pas le post-traitement historique proportionnel. La sortie est reconstruite sur :

- canvas : 736 px ;
- padding exact : 78 px ;
- cœur : 580 px ;
- 29×29 modules ;
- 20 px/module.

Le cœur VAE n'est jamais écrasé par la quiet zone.

## Critère de réussite

La baisse du MER n'est plus suffisante. Succès scientifique minimal : garde visuelle valide et soit `original_exact=true`, soit SSR QR-Verify supérieur au contrôle apparié.

Production et généralisation restent `false`.
