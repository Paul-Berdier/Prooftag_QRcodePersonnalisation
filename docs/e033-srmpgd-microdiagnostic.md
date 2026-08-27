# E033 - microdiagnostic numérique SR-MPGD

## Pourquoi E033 existe

E032 est techniquement complète mais scientifiquement négative. Son Stage 2 dégrade déjà
l'esthétique, puis ses traces SR-MPGD annoncent vingt itérations alors que le gradient, le pas
appliqué et le déplacement latent restent exactement nuls. E033 ne répète donc pas les trente
contextes : il vérifie d'abord que le mécanisme effectue réellement une descente.

## Matrice minimale

Un seul prompt (`e033_simple_greenhouse`), un seed (`51001`) et un payload court produisent cinq
sorties appariées :

1. Stage 1, référence esthétique uniquement ;
2. Stage 2 E032, témoin de la dégradation observée ;
3. Stage 2 de la démo publique DiffQRCoder (`ControlNet=1,05`, `SRG=50`, `PG=20`, cible binaire,
   initialisation aléatoire) ;
4. le même latent Stage 2 avec les équations 13-14 et le VAE dans sa précision modèle ;
5. le même latent Stage 2 avec les équations 13-14 et le VAE temporairement en FP32.

Les branches 4 et 5 prouvent leur parent par l'identifiant de run, l'identifiant de méthode, le
SHA-256 du latent, le SHA-256 du raster et l'identité de l'itération zéro. Le Stage 1 partagé est
également vérifié par son identifiant et son SHA-256.

## Corrections numériques testées

- loss scaling algébriquement neutre, dénormalisé avant l'application de `gamma` ;
- repli déterministe du facteur `32768` vers des facteurs plus faibles en cas de gradient non
  fini ;
- mesure du gradient SRL par rapport au raster pré-clamp et du gradient de l'objectif par rapport
  au latent ;
- arrêt explicite lorsqu'une SRL positive possède un gradient image ou latent nul ;
- promotion FP32 limitée au VAE et restauration garantie, y compris après un échec de conversion ;
- gradient de `SRL + 0,01 LPIPS` calculé en espace image, puis VJP VAE recalculé séparément avec
  checkpointing ; LPIPS/VGG est placé sur CPU pour les deux branches E033 ;
- témoin visuel du même latent redécodé sans aucun pas, afin de séparer l'effet de la précision VAE
  de l'effet de la descente.

## Visuels et portes

Le notebook affiche d'abord une planche des cinq sorties finales, puis une planche FP16/FP32 avec :

- le Stage 2 parent ;
- le latent redécodé sans mise à jour ;
- les itérations directes 0 et 1.

Un arrêt précoce laisse une case explicite `absent / 404` et un enregistrement
`available=False` ; il ne se transforme pas en faux résultat. La branche FP32 ne passe que si :

1. ses témoins et ses deux jalons sont disponibles et intègres ;
2. ses gradients image et latent initiaux sont finis et strictement positifs ;
3. le pas appliqué à l'itération 0 et le déplacement latent à l'itération 1 sont positifs ;
4. la SRL de l'itération 1 est inférieure à la SRL initiale.

Un PASS est seulement une preuve mécanistique locale. Il ne prouve ni la généralisation, ni la
probabilité de scan téléphone, ni une amélioration esthétique. Aucun élargissement automatique
n'est lancé.

## Arrêt technique et reprise

E033 n'autorise qu'une tentative de la campagne atomique. Une campagne encore active est reprise
après une coupure, mais une campagne terminale contenant une erreur n'est jamais régénérée
automatiquement. Le notebook affiche alors l'historique, la méthode fautive et le message conservé
dans chaque CSV, puis crée une archive `*-technical-failure.tar.gz` dans
`/workspace/downloads`. Les cellules scientifiques suivantes s'ignorent proprement : les sorties
existantes et l'état de reprise restent inchangés jusqu'à la correction de la cause.

Le premier plan `732665438a6e7382` a ainsi été arrêté par quatre OOM CUDA reproductibles : deux
branches FP16 et FP32, chacune répétée deux fois par l'ancien runner. La cause et le correctif
d'offload temporaire des modules de diffusion sont consignés dans
`docs/e033-technical-incident-2026-08-26.md`. Cette archive reste une preuve d'incident ; la relance
corrective doit obligatoirement créer un nouveau plan.

Le plan correctif `82549caa971652bb`, exécuté le 27 août, confirme que l'orchestration est réparée
mais que l'offload seul ne suffit pas. Une seule campagne et une seule tentative ont conservé les
trois témoins, puis les deux branches SR-MPGD ont échoué par OOM :

- FP16 : 19,63 Gio utilisés, 39,12 Mio libres, allocation de 266 Mio refusée ;
- FP32 : 19,57 Gio utilisés, 95,12 Mio libres, allocation de 530 Mio refusée.

Le VAE produit un raster complet de 736 x 736 px ; le crop exact de 78 px isole ensuite le cœur
580 px, mais ne réduit pas le graphe du décodeur. Le graphe VAE, la SRL et LPIPS/VGG étaient encore
vivants simultanément. Aucun état SR-MPGD, gradient ou déplacement latent de ce plan n'est donc
interprétable. L'archive technique et ses dix checksums valides prouvent l'incident, pas
l'appariement du latent dans les deux branches interrompues.

## Chemin de gradient mémoire borné

La prochaine relance conserve l'objectif d'Eq. 13, `SRL + 0,01 LPIPS`, mais évalue sa règle de
chaîne en deux niveaux :

1. calcul de `d(SRL)/dx` et de `d(LPIPS)/dx` sur le raster/cœur image, dans deux graphes séparés ;
2. libération de ces graphes, nouveau calcul du seul décodeur VAE avec checkpointing, puis VJP
   `(dx/dz)^T dL/dx` pour obtenir le gradient latent exact d'Eq. 14.

Cette décomposition ne modifie ni la fonction objectif ni `gamma`. Elle empêche seulement le VAE
736 px et LPIPS/VGG d'occuper simultanément la RTX. Les métriques scalaires et le raster de
diagnostic sont calculés sans conserver de graphe.

## Porte séquentielle révisée

E033 ne commence plus directement avec quatre itérations :

1. **Gate 1 itération :** même prompt, seed et latent parent ; SHA-256 d'appariement, raster
   d'itération zéro, gradients image/latent, pas appliqué, déplacement latent, mémoire et baisse
   de SRL entre 0 et 1 doivent tous être prouvés ;
2. **Gate 4 itérations :** uniquement après PASS du premier plan, nouveau plan immutable avec les
   jalons 0, 1, 2 et 4 et les mêmes contrôles ;
3. **Extension :** toujours interdite tant que le gate quatre itérations n'a pas conclu sans OOM,
   gradient nul, divergence ou perte d'appariement.

Le Stage 2 public du plan `82549caa971652bb` réduit le MER de 24,62 % à 1,90 % tout en restant à
0/37 QR-Verify. Il reste un parent expérimental utile, mais ne doit pas être confondu avec un QR
validé ni avec un succès SR-MPGD.

## Résultat du gate à une itération

Le plan `f2b9c0df72cd8cd4`, exécuté le 27 août sous le commit `8f66ac910023`, valide enfin le
mécanisme local : aucune erreur ni OOM, appariement exact, i0 pixel-identique au Stage 2 parent,
gradients image et latent non nuls, déplacement latent `0,0078558`, SRL réduite de `70,64 %` et
MER réduit de `1,7836 %` à `1,4269 %`. Les branches FP16 et FP32 sont presque identiques et ne
présentent pas de saturation destructive.

Le résultat reste cependant `0/37` QR-Verify avant et après la mise à jour. E033 autorise donc le
design du gate quatre itérations, pas une extension multi-prompt ni une revendication de
scannabilité. L'audit complet, les métriques mémoire et les limites d'interprétation sont consignés
dans `docs/e033-results-2026-08-27.md`.

## Exécution

Après avoir commité et poussé le dépôt, exécuter sur le serveur :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git fetch origin
git switch main
git pull --ff-only origin main
bash scripts/deploy-e033-notebook.sh
```

Puis sur le PC :

```powershell
git pull
.\scripts\notebook-remote.ps1 -Notebook 28_e033_srmpgd_microdiagnostic.ipynb
```

Dans Jupyter : **Run > Run All Cells**. Le plan, l'état de reprise, les CSV, les deux planches, les
traces, le rapport et l'archive restent sous `/data/e033-srmpgd-microdiagnostic` et une copie de
l'archive est placée dans `/workspace/downloads`.
