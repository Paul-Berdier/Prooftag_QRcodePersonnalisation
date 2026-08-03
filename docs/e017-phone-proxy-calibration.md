# E017 — Calibration téléphone et diagnostics structurels

## Pourquoi cette étape existe

La campagne `c2bf3033-f122-4270-b964-01d85d2749fa` a montré un désaccord
important entre les décodeurs logiciels stricts et le téléphone : 13 images
artistiques sur 28 étaient lues physiquement, alors qu'aucune n'était acceptée
automatiquement. Continuer à augmenter SRG ou SR-MPGD dans cette situation
optimiserait le mauvais signal et risquerait de dégrader l'image.

E017 ne change donc ni DiffQRCoder, ni sa règle de livraison. Il ajoute trois
instruments de mesure :

1. `Phone Proxy`, qui essaie sur l'image finale des prétraitements
   déterministes courants (niveaux de gris, agrandissement, CLAHE, netteté,
   Otsu et seuil adaptatif), puis demande aux mêmes décodeurs le payload exact ;
2. des diagnostics sur la géométrie exacte 736/78/580/20 de DiffQRCoder ;
3. des essais téléphone répétés avec nombre de réussites et appareil utilisé.

Le Phone Proxy est marqué `calibration_only=1`. Son score ne modifie pas
`status`, `scan_pass_rate`, la sélection de variante ou la porte de livraison.

## Données enregistrées

Chaque génération expose notamment :

- `phone_proxy_normalized_pass_rate` : taux Phone Proxy limité aux capacités du
  QR témoin propre ;
- `structure_module_center_error_rate` : centres de modules du mauvais côté du
  seuil ;
- `structure_functional_center_error_rate` : même mesure pour les motifs
  fonctionnels ;
- `structure_center_confidence_p10` et `structure_center_ambiguous_ratio` ;
- `structure_intra_module_std_p95` : texture ou bruit interne aux modules ;
- `structure_quiet_zone_dark_pixel_ratio` : contamination de la marge ;
- les SHA-256 des pixels finaux, du Stage 1, du QR témoin et, lorsqu'il existe,
  du latent Stage 2 ;
- le modèle de base, le ControlNet et la révision DiffQRCoder.

Les empreintes sont calculées sur les pixels décodés, avec le mode et la taille.
Deux PNG encodés différemment mais visuellement identiques ont donc la même
empreinte. Cela permet de prouver la réutilisation du Stage 1 et d'identifier
les doublons réels.

## Protocole humain

Pour chaque image, faire idéalement trois essais consécutifs avec le même
téléphone, la même application, la même distance et le même éclairage. Saisir :

- le nombre d'essais ;
- le nombre de lectures exactes ;
- le téléphone et l'application ;
- le verdict esthétique et les notes.

Le verdict de scan est calculé positif à partir de deux réussites sur trois
(seuil de 2/3). Les anciennes évaluations binaires restent compatibles et sont
converties en un essai réussi ou échoué.

## Décision après la campagne

Ne pas promouvoir le Phone Proxy avant au moins 100 images, 12 familles de
prompts et plusieurs appareils. Comparer alors Phone Proxy, SSR, MER et
diagnostics structurels aux répétitions physiques par appareil. Le prochain
seuil de production devra être choisi sur la sensibilité aux vrais scans, tout
en mesurant explicitement les faux positifs.

## Migration

La migration Alembic `0005_e017_phone_calibration` ajoute la provenance aux
générations et les champs de répétition aux évaluations. Le conteneur
`database-migrations` l'applique avant le démarrage de l'API.
