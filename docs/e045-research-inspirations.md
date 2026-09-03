# E045 — mécanismes externes à intégrer prudemment

Ce document distingue inspiration et implémentation validée.

## Text2QR — blueprint avant diffusion

Text2QR place un QR Aesthetic Blueprint en amont puis affine la scannabilité dans
le latent. Pour Prooftag, l'idée utile est d'optimiser avant génération :

- masque légal ;
- ECC/version ;
- alias ou nonce contrôlé du payload ;
- position du QR ;
- chevauchement avec la saliency du sujet.

Cela ne signifie pas que le code Text2QR est déjà intégré.

## Face2QR — conflits saliency/QR

Face2QR réorganise la compatibilité entre identité du visage et structure QR.
Prooftag ne doit pas permuter arbitrairement les modules. L'équivalent légal est
une recherche sur les matrices QR sémantiquement équivalentes et le placement.

## ArtCoder — simulation d'échantillonnage

ArtCoder utilise une couche de simulation d'échantillonnage. La prochaine loss
surrogate doit modéliser :

- déplacement du point d'échantillonnage dans le module ;
- point-spread et défocus ;
- contraste avec voisins ;
- variance intra-module ;
- finders, timing et alignment.

Elle ne doit pas se réduire à la moyenne du centre.

## DiffQRCoder

DiffQRCoder reste le moteur de référence épinglé, avec les corrections déjà
auditées :

- gradient perceptuel conservé ;
- latent Stage 2 exact ;
- géométrie entière ;
- SRL officielle ;
- SR-MPGD borné et conditionnel ;
- tous les checkpoints.

Le QArt Reed–Solomon exact décrit par le papier n'est pas disponible dans le
dépôt public ; aucune approximation n'est appelée « reproduction exacte ».

## Optimisation bayésienne

L'espace de 98 paramètres E045 n'est pas un produit cartésien. E046 utilisera une
optimisation multiobjectif contrainte et multi-fidélité :

```text
métadonnées/blueprint
→ preview Stage 1
→ Stage 1 complet
→ Stage 2
→ SR-MPGD conditionnel
→ téléphone
```

La politique recommandée est Ax/BoTorch avec qLogNEHVI, puis comparaison à un
échantillonnage Sobol et à la recette fixe.

## Simulation caméra

Les transformations numériques doivent être calibrées sur des captures réelles :

- perspective, rotation et distance ;
- flou de mouvement et défocus ;
- exposition, gamma, balance des blancs ;
- JPEG et bruit ;
- moiré d'écran ;
- reflets et ombres.

Elles servent à réduire le nombre de tests physiques, pas à les remplacer.
