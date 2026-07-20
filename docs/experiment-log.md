# Journal des expériences et décisions

Ce fichier est append-only : une hypothèse invalidée ou une erreur n'est pas supprimée. Une
correction reçoit une nouvelle entrée qui référence l'expérience précédente.

## Modèle d'entrée

```text
ID / date / auteur / commit / image Docker
Question et hypothèse
Configuration exacte
Dataset ou hash des cas
Résultats bruts
Anomalies et échecs
Décision : conserver, rejeter ou approfondir
Prochaine expérience
```

## E000 — Baseline avant raffinement perceptuel

- **Date :** 2026-07-20
- **Commit :** `83641dff89ef86d7365cb160e21c63fa36250986`
- **Archive :** `20260720T122033Z-83641dff.tar.gz`
- **GPU :** NVIDIA RTX 4000 SFF Ada Generation, 20 475 MiB
- **Logiciel :** PyTorch 2.4.1+cu121, Diffusers 0.31.0
- **Échantillon :** six cas fixes, une campagne

Résultats constatés :

| Mesure | Résultat |
|---|---:|
| Livraisons acceptées | 5/6, soit 83,33 % |
| Lecture moyenne sur les scénarios | 99,36 % |
| Pixels modifiés par rapport au brut | 58,81 % |
| Pixels écrêtés | 70,15 % |
| Durée totale moyenne | 5,043 s |
| Pic VRAM | 3 866 MiB |

### Échec important

`geometric-packaging` a obtenu un taux d'erreur module égal à zéro mais a échoué à une des 26
validations, avec un taux final de 96,15 %. Conclusion : le seuil moyen par module ne prédit
pas à lui seul la lecture. Les décodeurs et dégradations exactes restent obligatoires.

### Erreurs de démarche identifiées

1. Les premières réparations remplaçaient trop de centres par du noir ou blanc pur. Elles
   augmentaient la lecture mais rendaient la grille QR dominante.
2. Une erreur module nulle a été assimilée trop facilement à une garantie de lecture.
3. Six cas permettent de trouver des régressions, pas d'estimer un taux proche de 100 %.
4. Les sorties finales fortement réparées ne doivent pas devenir les cibles d'un futur
   entraînement, sinon le modèle apprendra ces artefacts.

### Décision

Conserver cette campagne comme baseline historique. Introduire des réparations arrondies pour
le secours immédiat, puis déplacer la correction principale dans le latent avec une loss par
module. Les valeurs complètes retenues sont versionnées dans
`docs/baselines/2026-07-20-83641dff.json`.

## E001 — Implémentation initiale SRL et raffinement latent

- **Date :** 2026-07-20
- **État :** implémenté localement, campagne GPU non encore exécutée
- **Hypothèse :** corriger le latent VAE à partir des seuls modules centraux incorrects doit
  réduire le recours aux superellipses et aux aplats binaires.

Éléments ajoutés :

- carte pixel-module respectant les arrondis de la production ;
- pondération gaussienne dans chaque module ;
- sous-module central d'un tiers ;
- arrêt local des modules déjà corrects ;
- poids renforcé sur les motifs fonctionnels ;
- loss de préservation L1 multi-échelle ;
- conservation du meilleur intermédiaire et retour inchangé en l'absence d'amélioration ;
- métriques Prometheus avant/après, pertes, durée, itérations et résultat ;
- option désactivée par défaut jusqu'au benchmark serveur.

### Paramètres initiaux à tester, pas encore validés

`iterations=8`, `learning_rate=0.20`, `qr_weight=1.0`,
`preservation_weight=0.15`, `functional_weight=4.0`.

### Matrice d'ablation prévue

| Groupe | Valeurs |
|---|---|
| Itérations | 4, 8, 12 |
| Learning rate | 0,05 ; 0,10 ; 0,20 |
| Préservation | 0,05 ; 0,15 ; 0,30 |
| Poids fonctionnel | 2 ; 4 ; 8 |

La prochaine entrée devra contenir les résultats GPU réels, y compris les combinaisons
rejetées et les images montrant les défauts.

### Vérifications locales

- `ruff check .` : réussi ;
- tests : 22 réussis, 1 ignoré ;
- validation des manifests et du dashboard Grafana à 30 panneaux : réussie ;
- test SRL avec PyTorch : présent mais ignoré sur le poste Windows, car l'environnement local
  ne contient pas PyTorch ; il devra obligatoirement s'exécuter dans l'image GPU du serveur ;
- construction Docker locale : non exécutée, le daemon Windows a refusé l'accès au pipe
  Docker. Ce n'est pas une erreur du code, mais cela reporte la validation CUDA au serveur.

La première exécution de pytest a également rencontré un refus d'accès dans le dossier
temporaire global de Windows. Elle a été relancée avec `--basetemp .pytest-tmp/srl-run-4`, ce
qui a produit le résultat complet ci-dessus. Cette option doit rester utilisée sur ce poste.

## E002 — Ablation contrôlée du raffinement latent SRL v1

- **Date :** 2026-07-20
- **Commit :** `374d1698e04274cc0762f9c07bd36e0880355378`
- **Image Docker :** `sha256:f5377eb37a8e5745540f86a731ceaa8875b9afc0aca69ed7d0f83add3d23eda1`
- **Contrôle :** `20260720T142359Z-374d1698.tar.gz`, raffinement désactivé
- **Traitement :** `20260720T142446Z-374d1698.tar.gz`, raffinement activé
- **Hash des six cas :** `7c15bd3ac05537fbd74c0458bea2a223158c6341d1024b00f8751c70c25e7f4e`
- **GPU :** NVIDIA RTX 4000 SFF Ada Generation, 20 475 MiB

Les deux campagnes utilisent exactement le même commit, la même image, les mêmes cas, seeds,
prompts et paramètres de génération. Les six images brutes et les six images finales ont des
hashs identiques entre les campagnes. L'écart observé est donc attribuable au raffinement et à
sa validation supplémentaire.

### Configuration testée

`iterations=8`, `learning_rate=0.20`, `qr_weight=1.0`,
`preservation_weight=0.15`, `functional_weight=4.0`,
`target_module_error_rate=0.0`.

### Résultats

| Mesure | Contrôle | SRL v1 | Écart |
|---|---:|---:|---:|
| Livraisons acceptées | 6/6 | 6/6 | aucune régression grâce au secours |
| Images brutes strictement lisibles | 0/6 | 0/6 | aucun gain strict |
| Erreur module moyenne | 13,152 % | 5,269 % | -52,425 % relatif |
| Durée totale moyenne | 5,185 s | 7,775 s | +2,589 s, soit +49,93 % |
| Pic VRAM | 3 880 MiB | 6 180 MiB | +2 300 MiB |
| Température maximale | 54 °C | 62 °C | +8 °C |
| Puissance moyenne | 38,86 W | 45,55 W | +6,69 W |
| Pixels modifiés par le latent | — | 99,389 % | inacceptable |
| Changement absolu moyen du latent | — | 0,2814 | inacceptable |

Un seul cas, `geometric-packaging`, passe 6 scénarios sur 26 avec ZBar : original, JPEG 90,
flou 3, luminosité haute, contraste bas et réduction à 75 %. OpenCV ne réussit aucun scénario.
Les cinq autres images latentes restent à 0/26.

L'inspection des six couples brut/latent montre le même défaut systématique : perte massive de
netteté, dérive des couleurs vers le gris/cyan, halos noirs et blancs, et destruction des
détails esthétiques. Le gain numérique sur les modules n'est donc pas un gain produit.

### Cause identifiée

La normalisation du gradient par sa moyenne absolue impose une mise à jour de forte amplitude à
chaque itération. Le latent VAE complet peut bouger alors que la loss QR ne concerne que certains
modules, et le poids de préservation 0,15 est trop faible pour retenir l'image. La règle de
sélection ne regardait que l'erreur des centres de modules et acceptait ainsi une image visuellement
détruite dès que cette erreur baissait.

### Décision

**Rejeter la v1** et garder le raffinement désactivé. Ces images ne doivent ni être livrées, ni
servir de cibles au dataset Prooftag. Les résultats numériques complets sont versionnés dans
`docs/baselines/2026-07-20-374d1698-latent-ablation.json`.

### Correction v2 préparée

- learning rate ramené de 0,20 à 0,02 ;
- poids de préservation relevé de 0,15 à 1,0 ;
- gradient limité spatialement aux modules encore incorrects ;
- déplacement du latent borné à ±0,10 autour de l'encodage initial ;
- refus automatique au-delà d'un changement absolu moyen de 0,08 ;
- distinction métrique entre amélioration observée et variante réellement acceptée ;
- aucune variante dupliquée n'est validée lorsque la porte de préservation la refuse.

Ces valeurs sont des paramètres de départ prudents, pas encore des paramètres validés. La
prochaine expérience E003 doit reprendre exactement les mêmes six cas, d'abord avec la v2 seule,
puis élargir à 100 générations uniquement si la porte visuelle est franchie.
