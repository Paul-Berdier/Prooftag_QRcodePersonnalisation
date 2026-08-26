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
