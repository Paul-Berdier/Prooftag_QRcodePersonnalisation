# E018 — Appariement strict du Stage 2 SRPG et de SR-MPGD

Date : 4 août 2026.

## Pourquoi cette phase est nécessaire

La campagne `prooftag-lab-831e74cb-6876-40b4-a055-6da347a8422d.csv`
contient 10 prompts, une seed et 9 méthodes, soit 90 lignes. Après exclusion
des 10 QR témoins, les 80 images artistiques donnent :

- 53 scans téléphone réussis sur 80, soit 66,3 % ;
- 80 jugements esthétiques positifs sur 80 ;
- SR-MPGD : 9/10 scans, CLIP-aesthetic moyen 5,05 ;
- SRPG force 0,80 : 9/10, CLIP-aesthetic moyen 4,59 ;
- SRPG force 0,65 : 8/10, CLIP-aesthetic moyen 4,97.

Ces résultats sont descriptifs : chaque image n'a reçu qu'un essai téléphone
et aucun modèle d'appareil n'a été renseigné. Ils ne constituent donc pas
encore une estimation publiable de la probabilité de scan.

L'audit de provenance a révélé un problème plus fondamental. Les SHA-256 du
latent Stage 2 de `diffqrcoder_srpg` et `diffqrcoder_srmpgd` étaient différents
pour les dix prompts. La comparaison n'était pas une application de SR-MPGD
sur le même résultat SRPG : deux Stage 2 distincts avaient été comparés.

## Contrat expérimental E018

Pour chaque triplet `(prompt, seed, payload)` :

1. `diffqrcoder_srpg` calcule le Stage 2 une seule fois ;
2. son latent propre, son image, sa cible et ses diagnostics sont conservés en
   mémoire pendant la campagne ;
3. l'état reçoit le `run_id`, le `method_id` et le SHA-256 du latent source ;
4. `diffqrcoder_srmpgd` doit importer cet état exact ;
5. le SHA est vérifié avant puis après le transfert vers le GPU ;
6. SR-MPGD applique seulement l'optimisation des équations 13–14 sur ce latent ;
7. toute absence de source ou différence de SHA met l'essai en erreur.

Il n'existe plus de recalcul silencieux du Stage 2 pour SR-MPGD.

## Clé canonique de partage

La clé inclut uniquement ce qui modifie mathématiquement le Stage 2 public :

- Stage 1, payload, prompt, seed et géométrie ;
- CFG, nombre de pas, force de redémarrage et initialisation ;
- poids ControlNet, SRG, perceptuel, ETA et seed dérivée ;
- fenêtre ControlNet et type de cible ;
- paramètres QArt uniquement lorsque ce mode expérimental est demandé.

Les aperçus, l'intervalle de sauvegarde, les gardes de sélection et tous les
paramètres SR-MPGD sont exclus. Ils interviennent après le latent propre et ne
doivent pas empêcher sa réutilisation.

## Preuves enregistrées

Chaque résultat expose maintenant :

- `provenance.stage2_latent_sha256` ;
- `provenance.stage2_source_latent_sha256` ;
- `provenance.stage2_source_run_id` ;
- `provenance.stage2_source_method_id` ;
- `provenance.stage2_pairing_status` (`generated_source` ou `exact_reuse`) ;
- `quality_metrics.diffqrcoder_stage2_pairing_exact`.

Le Web Lab affiche ces valeurs dans le détail de l'image. L'export CSV les
reprend dans les colonnes de provenance.

## Campagne suivante

Les quatre méthodes activées par défaut sont : QR témoin, Stage 1, SRPG 0,65
et SRPG 0,65 + SR-MPGD. L'ordre doit rester celui-ci afin que la source SRPG
existe avant SR-MPGD.

Protocole minimal :

- conserver les dix prompts de la campagne précédente ;
- utiliser au moins trois seeds ;
- faire trois essais téléphone par image ;
- renseigner l'appareil ;
- vérifier que chaque ligne SR-MPGD porte `exact_reuse` et que les deux SHA de
  latent sont identiques ;
- comparer SRPG et SR-MPGD par prompt/seed, jamais seulement par moyenne globale.

Le résultat attendu de cette phase n'est pas encore « 99 % ». Il est d'obtenir
une comparaison causalement valide permettant de décider si SR-MPGD apporte
réellement un gain au même Stage 2.
