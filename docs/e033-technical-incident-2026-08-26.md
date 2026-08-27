# E033 — incident VRAM du premier microdiagnostic

## Périmètre

Le plan `732665438a6e7382` n'a produit aucun résultat SR-MPGD interprétable. Les quatre
enregistrements en erreur de `732665438a6e7382-e033-failed-trials.csv` correspondent aux deux
branches équations FP16/FP32, chacune exécutée deux fois par l'ancien runner. Les répétitions ont
reproduit la même panne ; elles ne constituent pas quatre observations scientifiques.

## Preuve

| Branche | Échecs | Mémoire processus | Mémoire libre pilote | Allocation refusée |
|---|---:|---:|---:|---:|
| `e033_equation_srmpgd_fp16` | 2/2 | 19,35 Gio | 323–325 Mio | 530 Mio |
| `e033_equation_srmpgd_fp32` | 2/2 | 19,54–19,55 Gio | 117–125 Mio | 134 Mio |

Les valeurs sont presque identiques entre les deux tentatives. L'incident est donc déterministe et
ne doit pas être traité par une nouvelle tentative automatique.

## Cause racine

Après Stage 2, l'UNet, le ControlNet et les encodeurs restaient sur la RTX 4000 Ada. SR-MPGD
ajoutait ensuite le graphe différentiable du VAE et le réseau LPIPS/VGG. La branche FP32 promouvait
en plus le VAE. La rétropropagation VAE + LPIPS ne disposait alors plus de la marge nécessaire.

Le mode `paper_equations` construisait également un objet SRPG amont qui n'était jamais utilisé :
ce protocole calcule sa propre scanning-robust loss et transmet `scanning_loss=None` au moteur
SR-MPGD. Ni QArt, ni QR-Verify, ni CLIP/HPS ne sont responsables de cet arrêt.

## Correctif

- ne plus construire l'objet SRPG amont en mode `paper_equations` ;
- déplacer temporairement vers le CPU l'UNet, le ControlNet, les encodeurs texte/image et un
  éventuel SRPG déjà présent ;
- conserver le VAE sur CUDA pendant Eq. 13–14 ;
- déplacer le LPIPS mis en cache vers le CPU avant de restaurer les modules de diffusion ;
- restaurer les périphériques d'origine même si SR-MPGD lève une exception ;
- mesurer la mémoire allouée et la mémoire libre pilote avant/après l'offload et la restauration ;
- limiter E033 à une tentative et produire une archive technique exploitable en cas d'erreur.

Les tests locaux prouvent la séquence d'offload/restauration, y compris après exception, et
l'absence de construction SRPG inutile. Ils ne remplacent pas la validation CUDA réelle.

## Décision

Le plan `732665438a6e7382` reste une preuve d'incident en lecture seule. Il ne doit pas être repris,
mélangé à la relance ou compté dans une métrique. Le correctif modifie le code runtime et génère
donc un nouveau commit, un nouveau digest OCI et un nouveau `plan_id`.

La prochaine exécution reste le même microdiagnostic d'un prompt et d'un seed. Elle doit d'abord
prouver un gradient fini, un déplacement latent réel et une baisse de SRL. Aucun élargissement
multi-prompt ni aucune revendication de scannabilité n'est autorisé avant cette porte.

## Relance corrective du 27 août 2026

Le plan `82549caa971652bb` a exécuté une seule campagne et une seule tentative, conformément au
runner corrigé. L'archive
`82549caa971652bb-e033-srmpgd-microdiagnostic-v1-technical-failure.tar.gz`, de SHA-256
`3534cf4e22761f7027bf4fd237016a8cdb38a86cdfbf756c18d8250e3cf3c360`, contient dix
fichiers couverts par `checksums.json` ; les dix empreintes correspondent et aucun fichier de
données non manifesté n'est présent. L'état est terminal, sans campagne active : cinq trials ont
été exportés, trois sont `rejected` et deux sont `error`.

| Branche | Statut | QR-Verify | MER | CLIP-AES | HPS | Observation |
|---|---|---:|---:|---:|---:|---|
| `diffqrcoder_stage1` | rejetée | 0/37 | 24,62 % | 7,086 | 0,300 | référence esthétique |
| `diffqrcoder_paper_srpg` | rejetée | 0/37 | 9,04 % | 2,922 | 0,088 | saturation moyenne 0,695 ; garde de divergence déclenchée |
| `e033_public_demo_srpg` | rejetée | 0/37 | 1,90 % | 6,882 | 0,223 | latent parent produit, garde de divergence non déclenchée |
| `e033_equation_srmpgd_fp16` | erreur | — | — | — | — | OOM avant toute sortie SR-MPGD |
| `e033_equation_srmpgd_fp32` | erreur | — | — | — | — | OOM avant toute sortie SR-MPGD |

Le Stage 1 est bien réutilisé par les quatre branches suivantes. La démo publique produit le
latent parent de SHA-256
`215346a50b3223b3250f2fe3af2a35a3937779601db957fa4a1b09bae3565b7b`. En revanche,
les deux erreurs SR-MPGD surviennent avant l'enregistrement de leur provenance Stage 2 : cette
archive ne prouve donc pas encore que ces deux branches ont importé ce latent exact. Cet
appariement reste une porte à satisfaire, pas un résultat acquis.

### Mémoire observée

| Branche | Mémoire processus | PyTorch allouée | Réservée non allouée | Libre pilote | Allocation refusée |
|---|---:|---:|---:|---:|---:|
| FP16 | 19,63 Gio | 18,81 Gio | 615,00 Mio | 39,12 Mio | 266,00 Mio |
| FP32 | 19,57 Gio | 18,99 Gio | 374,19 Mio | 95,12 Mio | 530,00 Mio |

La fragmentation peut contribuer, mais elle n'explique pas à elle seule une exécution qui occupe
déjà environ 19,6 Gio sur 19,67 Gio. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` n'est donc
pas accepté comme correctif scientifique suffisant.

## Cause consolidée

L'offload des modules de diffusion a supprimé le travail redondant et permis de conserver les
trois témoins, mais il n'a pas rendu le backward SR-MPGD viable. Le VAE décode toujours le raster
complet de 736 x 736 px avant le crop QR de 78 px. Le crop donne bien un cœur exact de
`736 - 2 * 78 = 580 = 29 * 20` px, mais il ne réduit pas les activations intermédiaires du
décodeur VAE. Le chemin défaillant gardait simultanément en mémoire :

1. le graphe différentiable du décodeur VAE 736 px ;
2. le graphe de la SRL ;
3. le graphe LPIPS/VGG ;
4. les graphes retenus pour les diagnostics et le gradient latent.

Le second incident n'est donc ni une erreur QArt, ni un échec QR-Verify, ni un nouveau gradient
nul. C'est un dépassement mémoire du graphe combiné de l'objectif image et du VJP VAE, avant que
la première mise à jour puisse être auditée.

## Décision numérique après le second incident

La prochaine implémentation doit appliquer exactement la même règle de chaîne d'Eq. 14, mais en
étapes mémoire bornées :

1. décoder sans graphe pour le raster candidat et les diagnostics scalaires ;
2. calculer séparément `d(SRL)/dx` puis `d(LPIPS)/dx` sur des feuilles image, avec LPIPS/VGG hors
   de la mémoire GPU contrainte lorsque nécessaire ;
3. sommer ces deux gradients en `dL/dx`, puis libérer complètement les graphes SRL et LPIPS ;
4. recalculer un seul décodage VAE avec gradient checkpointing et calculer le produit
   vecteur-jacobienne `(dx/dz)^T dL/dx` ;
5. appliquer ensuite seulement la mise à jour `z <- z - gamma * dL/dz`.

Cette factorisation est la règle de chaîne exacte ; elle ne remplace ni la SRL, ni LPIPS, et ne
change pas leurs poids. Elle évite seulement de garder VAE et LPIPS vivants simultanément.

La reprise est désormais divisée en deux portes immuables. Le premier nouveau plan exécute **une
seule itération** sur le même prompt et le même seed. Il doit prouver : latent parent apparié par
SHA-256, itération zéro identique, gradients image et latent finis et positifs, pas appliqué non
nul, déplacement latent non nul, absence d'OOM et SRL de l'itération 1 inférieure à celle de
l'itération 0. Ce n'est qu'après ce PASS qu'un autre plan pourra exécuter **quatre itérations**.
Tout échec reste un STOP archivé ; aucune extension multi-prompt n'est autorisée.

## Progrès réellement acquis

- plus aucune seconde tentative automatique : une erreur déterministe ne double plus le coût GPU ;
- archive technique complète, checksummée et directement interprétable ;
- Stage 1, témoin PDF et parent public conservés malgré l'arrêt des branches SR-MPGD ;
- séparation claire entre baisse de MER et lecture réelle : la démo publique atteint 1,90 % de
  MER mais reste à 0/37 QR-Verify ;
- cause mémoire localisée au backward VAE 736 px combiné à LPIPS, et non plus seulement à la
  présence des modules de diffusion.

Ces progrès concernent l'observabilité et l'isolation de la panne. Ils ne constituent toujours
pas une validation du SR-MPGD.
