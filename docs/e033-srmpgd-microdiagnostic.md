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
- témoin visuel du même latent redécodé sans aucun pas, afin de séparer l'effet de la précision VAE
  de l'effet de la descente.

## Visuels et portes

Le notebook affiche d'abord une planche des cinq sorties finales, puis une planche FP16/FP32 avec :

- le Stage 2 parent ;
- le latent redécodé sans mise à jour ;
- les itérations directes 0, 1, 2 et 4.

Un arrêt précoce laisse une case explicite `absent / 404` et un enregistrement
`available=False` ; il ne se transforme pas en faux résultat. La branche FP32 ne passe que si :

1. ses témoins et jalons sont disponibles et intègres ;
2. son gradient initial est fini et strictement positif ;
3. le latent s'est déplacé à l'itération 1 ;
4. la SRL minimale des itérations 1 à 4 est inférieure à la SRL initiale.

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
