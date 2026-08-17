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

## E003 — Ablation contrôlée du raffinement latent SRL v2

- **Date :** 2026-07-20
- **Commit :** `584bb0cb3a042421da067282c8fabd984919bab4`
- **Image Docker :** `sha256:8f7d2049a0b4673d0e0c16d8d361ae6a57667645ea89c1c3062377228f32a99e`
- **Contrôle :** `20260720T145422Z-584bb0cb.tar.gz`, raffinement désactivé
- **Traitement :** `20260720T145508Z-584bb0cb.tar.gz`, raffinement activé
- **Hash des six cas :** `7c15bd3ac05537fbd74c0458bea2a223158c6341d1024b00f8751c70c25e7f4e`
- **GPU :** NVIDIA RTX 4000 SFF Ada Generation, 20 475 MiB

Les campagnes partagent le commit, l'image Docker, les paramètres, les cas et les seeds. Les
hashs des six images brutes et des six images finales sont identiques. La comparaison mesure donc
uniquement le coût et l'effet du raffinement v2.

### Configuration testée

`iterations=8`, `learning_rate=0.02`, `qr_weight=1.0`,
`preservation_weight=1.0`, `functional_weight=4.0`, `max_latent_delta=0.10`,
`max_mean_absolute_change=0.08`.

### Résultats

| Mesure | Contrôle | SRL v2 | Écart |
|---|---:|---:|---:|
| Livraisons acceptées | 6/6 | 6/6 | aucune régression grâce au secours |
| Images brutes/latentes strictement lisibles | 0/6 | 0/6 | aucun gain de lecture |
| Scénarios de lecture réussis par le latent | — | 0/156 | aucun sauvetage |
| Erreur module moyenne | 13,152 % | 12,306 % | -4,631 % relatif |
| Durée totale moyenne | 5,210 s | 7,689 s | +2,479 s, soit +47,59 % |
| Durée moyenne du raffinement | — | 1,908 s | huit itérations dans tous les cas |
| Pic VRAM | 3 896 MiB | 6 228 MiB | +2 332 MiB |
| Température maximale | 54 °C | 61 °C | +7 °C |
| Puissance moyenne | 38,06 W | 46,26 W | +8,20 W |
| Pixels modifiés par le latent | — | 28,259 % | -71,55 points face à la v1 |
| Changement absolu moyen du latent | — | 0,02786 | -90,10 % face à la v1 |
| Netteté moyenne | 3 278,26 | 3 239,12 | -1,19 % |

La v2 supprime le défaut visuel catastrophique de E002. Les six images restent cohérentes,
nettes et exploitables comme illustrations. Le cas `dense-payload` reste le plus affecté : 70,01 %
des pixels changent, MAE 0,06128 et netteté en baisse de 13,07 %. Il reste sous la porte de 0,08,
mais doit rester un cas sentinelle dans les prochaines ablations.

### Résultats par cas

| Cas | Erreur brute | Erreur latente | Réduction relative | MAE latente | Lecture |
|---|---:|---:|---:|---:|---:|
| botanical-short | 6,864 % | 6,815 % | 0,72 % | 0,02082 | 0/26 |
| wine-label | 8,889 % | 8,346 % | 6,11 % | 0,01724 | 0/26 |
| geometric-packaging | 14,202 % | 13,453 % | 5,28 % | 0,01923 | 0/26 |
| cosmetics-organic | 8,372 % | 8,372 % | 0,00 % | 0,02340 | 0/26 |
| industrial-technical | 13,703 % | 13,203 % | 3,65 % | 0,02517 | 0/26 |
| dense-payload | 26,885 % | 23,651 % | 12,03 % | 0,06128 | 0/26 |

### Décision

**Conserver l'architecture et les garde-fous de la v2, mais rejeter ces paramètres comme réglage
final.** La préservation visuelle est obtenue, mais le signal QR est trop faible pour provoquer un
seul décodage. Ne pas lancer la campagne de 100 images et ne pas utiliser ces sorties comme cibles
d'entraînement positives.

### Anomalie de chaîne révélée par l'inspection des images finales

Les six `final.png` sont identiques bit pour bit au contrôle. Ce résultat est d'abord normal parce
que `latent_srl` échoue seul, mais l'inspection du code révèle une lacune supplémentaire : les
profils `rounded`, `perceptual`, `incorrect` et `uncertain` étaient ensuite appliqués à l'image
brute. L'amélioration latente était donc abandonnée au lieu de devenir la base de la réparation
ciblée. Le latent n'aurait pu changer le final qu'en passant directement les 26 validations.

La chaîne est corrigée après E003 : lorsqu'un latent franchit la porte visuelle, les réparations
ciblées `latent_*` sont évaluées sur celui-ci en premier. La chaîne complète issue du brut reste
ensuite disponible comme secours, et les réparations globales restent issues du brut. Cette
structure permet de mesurer si le latent réduit réellement la quantité de pixels QR visibles sans
sacrifier la fiabilité de livraison.

La prochaine expérience E004 testera d'abord cette chaîne corrigée avec les paramètres v2 sur les
six cas fixes. Une ablation de force ne sera lancée que si ce test confirme que les variantes
`latent_*` apparaissent bien dans le rapport. Elle fera ensuite varier une dimension à la fois
autour de `learning_rate=0.02`, `preservation_weight=1.0` et `max_latent_delta=0.10`, tout en
conservant la porte MAE à 0,08. Le critère de passage exige au minimum une réparation `latent_*`
sélectionnée avec un écart visuel inférieur à la réparation brute correspondante. Les données
complètes sont versionnées dans
`docs/baselines/2026-07-20-584bb0cb-latent-v2.json`.

## E004 — Seconde diffusion guidée localisée

- **Date :** 2026-07-21
- **État :** implémentée localement, campagne GPU non encore exécutée
- **Question :** une seconde diffusion peut-elle intégrer les corrections QR dans le style avant
  les réparations déterministes ?
- **Référence :** pipeline en deux étapes SRPG puis SR-MPGD de DiffQRCoder.

### Erreur de conception corrigée

E003 ajoutait des corrections après la diffusion, puis appliquait éventuellement SR-MPGD. Le
modèle génératif ne voyait donc jamais les points correctifs. E004 reconstruit un guide depuis les
modules incorrects, ajoute de nouveau du bruit à l'image artistique et exécute une passe img2img
ControlNet avant SR-MPGD.

### Implémentation

- seconde passe déterministe avec seed décalée et enregistrée ;
- guide local : motifs fonctionnels verrouillés, centres incorrects ou incertains uniquement ;
- huit étapes effectives, soit 27 timesteps planifiés à strength 0,30, et poids ControlNet 1,75 ;
- projection du résultat dans un masque dérivé du guide, dilaté et adouci de quatre pixels ;
- refus de la seconde passe au-dessus d'une MAE de 0,12 ;
- SR-MPGD appliqué à `guided` et non plus obligatoirement au brut ;
- réparations `guided_latent_*` calculées depuis le meilleur intermédiaire ;
- chaîne brute historique intégralement conservée en secours ;
- artefacts `guided_control`, `guided_mask`, `guided_unprojected`, `guided_candidate`, `guided`
  et `guided_latent_srl` ;
- métriques Prometheus et quatre panneaux Grafana spécifiques ;
- protocole benchmark 2.1 et export `refinements.csv` ;
- `run_id`, tentative et seed ajoutés aux journaux des raffinements.

### Validation locale avant campagne GPU

- Ruff : aucun défaut sur `prooftag_qr`, `scripts`, `tests` et `main.py` ;
- tests : 30 réussis, 1 ignoré ;
- couverture Python totale : 76 % ;
- manifestes : 14 documents Kubernetes chargés sans erreur ;
- observabilité : dashboard JSON valide, 34 panneaux ;
- `git diff --check` : aucune erreur d'espace ou de patch.

La passe GPU n'est pas simulée sur le PC Windows : elle doit être exécutée sur la RTX afin de
mesurer la VRAM, le temps et le comportement réel de Diffusers. La syntaxe du script Bash sera
donc également confirmée par son exécution sur le serveur Linux ; le poste local ne possède pas
de distribution WSL Bash configurée.

### Limite assumée

Cette étape utilise ControlNet comme guidage pendant le débruitage, puis SR-MPGD après la seconde
diffusion. Elle ne calcule pas encore le gradient SRPG exact à travers l'UNet à chaque timestep et
n'implémente pas Qart. Les taux de réussite publiés par DiffQRCoder ne peuvent donc pas être
revendiqués pour E004 avant reproduction complète et mesure locale.

### Porte avant E004b

La campagne courte doit comparer une baseline et E004 sur le même commit et la même image Docker.
On n'élargit les paramètres que si la galerie contient les quatre étapes, que les six finals restent
à 26/26, et qu'au moins une variante `guided_*` réduit la réparation visible. Le protocole et les
commandes sont détaillés dans `docs/e004-guided-rediffusion.md`.

### Résultat GPU E004 — échec confirmé

- **Archives :** `20260721T074704Z-308953d2.tar.gz` et
  `20260721T074752Z-308953d2.tar.gz` ;
- **Contrôle d'identité :** 6/6 images brutes ont le même SHA-256 entre les campagnes ;
- **Lecture :** `raw`, `guided` et `guided_latent_srl` restent à 0/26 pour chaque cas ;
- **Signal :** le guide descend à 5,825 % d'erreur module moyenne, mais la rediffusion remonte à
  15,072 % contre 13,152 % au départ ;
- **Localité :** masque moyen 68,16 %, jusqu'à 83,68 % sur le cas dense ;
- **Final :** toujours 26/26 par réparation déterministe, mais 74,00 % des pixels sont modifiés
  contre 58,75 % au contrôle ;
- **Coût :** 9,62 s contre 5,18 s (+85,7 %) et 6 192 MiB contre 3 860 MiB ;
- **Anomalie rapport :** `guided_candidate` et `guided` sont identiques bit pour bit dans les six
  cas.

**Décision : E004 est rejetée, aucune ablation E004b.** L'acceptation fondée uniquement sur la
MAE était incorrecte. Le code exige désormais une amélioration QR réelle et ne publie plus deux
artefacts identiques. L'ancien mode reste désactivé.

## E005 — SRPG dans chaque timestep DDIM

- **Date :** 2026-07-21
- **État :** implémentation et instrumentation locales terminées ; campagne RTX requise
- **Question :** le gradient SRL + LPIPS injecté dans chaque prédiction de bruit produit-il une
  amélioration QR effectivement mesurée par les décodeurs ?

### Corrections méthodologiques

- boucle DDIM explicite à 40 pas au lieu d'un appel img2img suivi d'une projection ;
- calcul différentiable `z0|t -> VAE -> SRL + LPIPS -> gradient(z_t)` à chaque pas ;
- poids gelés et gradient checkpointing UNet/ControlNet pour la carte 20 Go ;
- cap RMS sur le delta de bruit, rejet des NaN/Inf et suivi du pic CUDA ;
- image SRPG toujours soumise aux 26 validations, même quand sa porte interne la rejette ;
- réutilisation comme base des réparations seulement avec baisse QR réelle et MAE sous la porte ;
- sélection finale parmi toutes les variantes valides par MAE puis proportion de pixels changés,
  au lieu de conserver la première réparation lisible ;
- export sans duplication : `attempt_1_srpg.png`, `srpg-steps.csv`, courbe par cas et journal des
  40 pas ;
- campagne causale baseline/SRPG seul, les raffinements E004 et latent étant désactivés.

### Limite déclarée avant résultat

La cible ControlNet est encore le QR exact ; QArt n'est pas reproduit dans E005a. Le résultat ne
pourra donc être comparé aux 99–100 % de DiffQRCoder que comme une adaptation partielle. Aucun taux
de réussite E005 n'est inscrit ici avant réception des deux archives produites par
`make benchmark-e005`.

### Validation locale avant publication

- Ruff : aucun défaut sur le package, les scripts, les tests et `main.py` ;
- tests : 37 collectés, 34 réussis et 3 ignorés faute de PyTorch dans le venv Windows ;
- couverture : 68 %, la boucle GPU restant volontairement non exécutée sur ce poste ;
- compilation Python : réussie ;
- Kubernetes : 14 documents YAML chargés, dashboard JSON valide et 40 panneaux uniques ;
- PowerShell : analyse syntaxique AST réussie pour le lanceur distant ;
- espaces/patch : `git diff --check` réussi.

Les deux tests différentiables SRPG et un test latent nécessitent PyTorch. Le Dockerfile transforme
cette absence locale en contrôle de build : il importe la pile GPU et instancie LPIPS/AlexNet afin
de télécharger les poids dans l'image avant le déploiement. La syntaxe Bash et l'exécution CUDA
restent des contrôles serveur obligatoires ; Windows ne possède aucune distribution WSL locale.

### Résultat GPU et autopsie E005a

- **Archives :** `20260721T090445Z-0b3c040b.tar.gz` et
  `20260721T090541Z-0b3c040b.tar.gz` ;
- **baseline :** invalide, 5/6 cas en `Connection refused` ;
- **SRPG strict :** 1 lecture réussie sur 156 scénarios, soit 0,641 % ;
- **changement SRPG :** 95,565 % des pixels en moyenne, MAE 0,20547 ;
- **erreur module :** 13,1525 % avant contre 12,2327 % après en moyenne, mais trois cas sur six se
  dégradent ;
- **livraison :** 6/6 à 26/26 uniquement grâce aux variantes déterministes `perceptual_*` ;
- **cause visuelle :** `strength=1.0` détruit l'identité du brut ; SRPG est rejeté ou illisible,
  puis la chaîne repart du brut et superpose la réparation qui rend les modules visibles.

**Décision : E005a rejetée comme configuration.** Il est interdit d'interpréter les 6/6 finales
comme une réussite du modèle. L'expérience suivante commence par l'observation : capture du
contrôle, des `x0` et des cartes d'erreurs aux pas 0/5/10/.../39, puis lecture dans
`notebooks/01_srpg_step_by_step.ipynb`. Une ablation de force ou de loss ne sera définie qu'après
avoir localisé précisément la rupture. Le benchmark retourne maintenant un échec si une campagne
est incomplète, ce qui empêche de répéter la fausse comparaison de cette première baseline.

### Correction de l'outil d'observation

Le premier notebook livré pour cette autopsie ne recalculait aucune image : il ne faisait que
relire les artefacts d'une archive E005. C'était utile pour comparer une campagne, mais insuffisant
pour observer et modifier l'expérience. Cette confusion d'objectif est enregistrée comme une
erreur d'outillage.

Le notebook `02_generate_live_on_gpu.ipynb` corrige ce problème. Son kernel s'exécute dans un pod
Kubernetes qui possède le GPU. Il produit réellement le QR de contrôle, la diffusion brute, les
prédictions `x0` et cartes d'erreurs pendant la boucle DDIM/SRPG, les métriques par pas, toutes les
réparations et leurs validations. Le notebook 01 reste volontairement un lecteur d'archives et ne
doit plus être présenté comme un générateur.

L'accès se fait depuis le navigateur Windows au travers d'un tunnel SSH authentifié. Le lanceur
mémorise les réplicas de l'API et de vLLM, les met à zéro avant d'attribuer la RTX au notebook, puis
restaure exactement cet état à l'arrêt. Cette instrumentation n'est pas encore un résultat E005b :
aucune amélioration n'est revendiquée avant une exécution complète sur la RTX.

## E006 — recherche des paramètres SRPG

- **Date :** 2026-07-21
- **État :** protocole et notebook terminés ; campagne RTX à exécuter
- **Observation initiale :** avec le même exemple, `SRPG_STEPS=100` est lisible sur un téléphone
  alors que la configuration à 40 pas ne l'était pas.
- **Interprétation :** observation encourageante sur un seul cas, pas un taux de réussite.
- **Décision :** criblage causal de 17 profils, puis confirmation des trois meilleurs sur trois
  autres cas et validation physique structurée.

L'audit a ajouté les seuils SRL noir/blanc configurables afin de comparer le comportement Prooftag
0,50/0,50 au comportement officiel 0,45/0,65. Le classement donne la priorité au 100 % strict,
puis au pire cas, avant l'esthétique et le temps. Les paramètres, validations individuelles,
métriques par pas, erreurs et résultats physiques sont persistés. Voir
[`docs/e006-parameter-search.md`](e006-parameter-search.md). Aucun profil n'est promu en production
avant résultats RTX et tests sur plusieurs téléphones.

## E007 - optimiseur contextuel et qualité sous contrainte

- **Date :** 2026-07-21
- **État :** implémentation locale ; campagne RTX requise
- **Déclencheur :** E006 varie de 1/26 à 23/26 selon les contextes et ne produit aucun profil
  universel strict.
- **Décision :** abandon du réglage statique au profit d'une recherche TPE factorielle et d'un
  surrogate contextuel.

La sélection est désormais lexicographique : 26/26 obligatoire, puis CLIP-aesthetic et CLIPScore.
La SRL intègre des vues floues, réduites, éclaircies/assombries et à contraste réduit. Les axes
prompt, seed et payload sont isolés, quatre contextes holdout sont obligatoires, et un groupe
incomplet ne peut pas être déclaré strict. Un garde-fou `nvidia-smi` empêche de répéter la
contamination GPU d'E006. Détails :
[`docs/e007-contextual-optimizer.md`](e007-contextual-optimizer.md).

Avant les holdouts, les huit meilleurs essais sont maintenant recalculés sur les douze contextes
factoriels. Cette promotion à 96 exécutions empêche qu'un résultat chanceux sur un prompt unique
devienne la configuration globale du conseiller.

Un `KeyboardInterrupt` observé le 21 juillet a révélé que le compteur de reprise considérait aussi
les essais `FAIL` comme terminés. La reprise compte désormais uniquement les essais Optuna
`COMPLETE` : une interruption reste auditée mais une exécution de remplacement est obligatoire afin
d'obtenir réellement 72 résultats exploitables.

Le calcul d'importance restait ensuite opaque pendant la fANOVA Optuna par défaut (64 arbres,
profondeur maximale 64). Il est désormais déterministe et borné à 32 arbres de profondeur 16, avec
messages de début/fin et contrôle explicite de la présence des résultats `search/ok`.

La calibration et les holdouts ne produisaient auparavant aucune sortie pendant leurs longues
boucles GPU. Chaque exécution affiche désormais `START`, statut, scan, durée et progression globale.
Les lignes étrangères aux configurations actuellement promues sont filtrées et une campagne
incomplète s'arrête explicitement avant la heatmap ou l'entraînement de l'advisor.

## E008 — bake-off des ControlNet QR

- **Date :** 2026-07-21
- **État :** recherche et protocole implémentés ; campagne RTX à exécuter
- **Déclencheur :** `Nacholmo/controlnet-qr-pattern-v2` pourrait mieux masquer la grille que la
  baseline Dion, mais aucune source ne publie une comparaison quantitative appariée.
- **Décision :** ne changer aucun défaut de production avant comparaison de Dion, Monster v1,
  Monster v2 et Nacholmo v2 sur 192 exécutions SD1.5.

La branche SDXL reste séparée en raison de l'incompatibilité de pipeline/SRPG et du risque VRAM.
Le protocole, les preuves disponibles et leurs limites sont consignés dans
[`docs/e008-controlnet-bakeoff.md`](e008-controlnet-bakeoff.md).

Une réussite automatique E008 produit seulement un `AUTOMATIC_CANDIDATE`. Elle autorise la suite
E007 mais pas la production, qui exige encore les holdouts et la validation physique documentée.

### Résultat E008 et décision de trajectoire

- **Archive :** `e008-controlnet-bakeoff-v1.tar.gz` ;
- **complétude :** 192/192 sorties, aucune erreur ;
- **porte :** aucun profil strict sur les douze contextes, statut `NO_PROMOTION` ;
- **meilleur observé :** Nacholmo v2 à 1,60, moyenne 66,0 %, pire cas 9/26 ;
- **stricts isolés :** Monster v2/1,60/seed-9001 et Nacholmo v2/1,60/seed-2026 ;
- **goulot :** OpenCV 85,9 % contre ZBar 46,2 % pour Nacholmo 1,60 ;
- **décision :** optimisation E009 spécifique à Nacholmo, Monster v2 conservé comme challenger,
  aucun changement de production.

E007/Dion est archivé comme baseline partielle après 72 recherches et 80 calibrations, toutes
non-strictes. Les 16 calibrations manquantes et les holdouts ne justifient pas de temps GPU avant
la nouvelle population Nacholmo. Rapport :
[`docs/e007-e008-results-2026-07-22.md`](e007-e008-results-2026-07-22.md).

## E009a — correction d'intégration Nacholmo

- **Date :** 2026-07-22
- **Déclencheur :** le brut Nacholmo transforme presque chaque module QR en objet carré et dégrade
  fortement le respect du prompt.
- **Cause :** E008 et la première version du notebook 06 utilisaient le QR comme source img2img avec
  une force ControlNet 1,60. La fiche Nacholmo montre au contraire une pipeline ControlNet
  text2img ; 1,60 était le meilleur point de lecture E008, pas un optimum esthétique.
- **Première correction :** Stage-1 text2img sur DreamShaper 8, 30 pas, CFG 6,5, balayage des
  forces 0,80/1,00/1,20 ; pipeline img2img/DDIM séparée pour SRPG 100 pas.
- **Résultat :** toujours fortement quadrillé. La classe de pipeline était corrigée, mais la
  condition binaire complète, appliquée jusqu'au dernier pas, dominait encore la composition.
- **Deuxième correction :** base `Nacholmo/Counterfeit-V2.5-vae-swapped` recommandée par l'auteur,
  condition ternaire `nacholmo_extremes_25`, profils force/fin `0,40/0,55`, `0,55/0,70` et
  `0,75/0,85`. SRPG reste volontairement binaire et séparé.
- **Traçabilité :** images de chaque force, classe de pipeline, scheduler, validations, CLIP,
  paramètres et sélection sont inscrits dans l'archive et `manifest.json`.
- **État :** implémenté, validation RTX et test téléphone requis ; aucun taux de réussite n'est
  annoncé avant cette nouvelle campagne.

## E010 — retour contrôlé à DiffQRCoder officiel

- **Date :** 2026-07-22
- **Déclencheur :** les sorties Nacholmo restent visuellement quadrillées et la pipeline historique
  Prooftag est une approximation img2img, pas l'algorithme à deux stages du dépôt DiffQRCoder.
- **Décision :** figer le dépôt officiel au commit `e24ea73`, isoler sa pile publiée dans l'image
  notebook et en faire la nouvelle baseline observable.
- **Comparaison :** Stage 1, Stage 2 SRPG et Stage 2 SRPG + SR-MPGD, avec état aléatoire Stage 2
  apparié, aperçu tous les cinq pas et validation stricte multi-décodeur/multi-dégradation.
- **Porte :** aucun fichier `DELIVERABLE` sans réussite exacte sur tous les tests ; CLIP-aesthetic
  et CLIPScore interviennent uniquement après la scannabilité.
- **Correctif amont audité :** remplacement de l'agrégation `torch.tensor` par `torch.stack` dans
  `PerceptualLoss` afin de conserver le graphe et le device, sans changer la formule de loss.
- **État :** notebook et protocole implémentés ; exécution RTX requise avant toute revendication de
  taux. Voir [`docs/e010-diffqrcoder-official-reference.md`](e010-diffqrcoder-official-reference.md).

### Premier démarrage E010 — OOM avant génération

- **Symptôme :** `pipe.to('cuda')` échoue avec 19,66 Gio utilisés sur 19,67 Gio ; aucune image n'a
  encore été générée.
- **Diagnostic :** le processus du kernel Jupyter détient lui-même 19,37 Gio alloués. Il ne s'agit
  ni du téléchargement ni d'un autre pod GPU. `empty_cache()` ne libérait pas les pipelines encore
  référencées par le kernel.
- **Correction :** libération CPU et suppression explicite des objets GPU, nettoyage de
  l'historique IPython, garbage collection, porte de propreté à 1 Gio et message demandant un
  redémarrage du kernel si nécessaire.
- **Prévention 20 Gio :** paramètres gelés, attention/VAE slicing et gradient checkpointing UNet +
  ControlNet avant le Stage 2 différentiable.
- **Compatibilité Jupyter :** la première implémentation appelait la magic `%reset_out`, absente du
  serveur. Elle est remplacée par l'effacement direct de `Out`, `_`, `__` et `___` dans le namespace
  IPython.
- **Résultat scientifique :** aucun ; l'échec ayant précédé le Stage 1, il ne compte pas comme un
  essai du modèle.

### Première exécution du Stage 2 — conversion PIL encore attachée

- **Symptôme :** les 40 pas SRPG se terminent en 93 secondes, puis `pt_to_numpy` refuse le tenseur
  final car il requiert encore un gradient.
- **Cause :** le notebook appelle `_run_stage2` directement pour séparer et apparier les variantes,
  alors que seul `DiffQRCoderPipeline.__call__` porte le décorateur externe `@torch.no_grad()`.
- **Correction :** `run_stage2` est maintenant décorée par `@torch.no_grad()`. Les sections SRPG et
  SR-MPGD du code amont utilisent leurs propres blocs `torch.enable_grad()` : leurs gradients sont
  donc conservés, tandis que le décodage final devient détaché comme dans l'appel public officiel.
- **Résultat scientifique :** aucun fichier final n'ayant été produit, cette tentative reste un
  contrôle d'intégration, pas un résultat de modèle.

### Export de l'archive E010 — confusion de chemin

- **Symptôme :** `results/notebook-runs/...` est introuvable depuis le kernel.
- **Cause :** le kernel travaille dans `/workspace/notebooks`, tandis que les expériences sont
  écrites dans le chemin absolu `/data/notebook-runs`. De plus, `/data` appartient au conteneur et
  ne peut pas être récupéré directement par `scp` depuis l'hôte.
- **Correction :** l'archivage utilise `RUN_DIR` absolu ; les instructions finales font un
  `kubectl cp` du pod vers le home Linux, puis un `scp` du serveur vers Windows.

### Résultat E010 — rejet de la baseline visuelle

- **Archive :** `20260722T114948Z-e010-diffqrcoder-official-v1-seed1-Copy1.tar.gz`.
- **Lecture :** Stage 1, SRPG et SRPG + SR-MPGD sont tous à 0/26.
- **Évolution :** la MER baisse de 29,36 % à 18,19 %, puis 16,51 %, mais CLIP-aesthetic chute de
  7,19 à 3,88/3,98 et CLIPScore de 0,844 à 0,567/0,599.
- **Conclusion :** le guidage public améliore la structure moyenne sans restaurer la lecture et
  dégrade fortement l'image. Ces sorties ne constituent pas une baseline acceptable.
- **Écart identifié :** le Stage 2 du dépôt public repart d'un bruit aléatoire alors que le papier
  décrit l'encodage bruité du Stage 1 ; QArt n'est pas fourni dans le dépôt.

## E011 — DiffQRCoder contre reproduction publique QRBTF

- **Date :** 2026-07-22
- **Périmètre :** deux familles seulement, quatre prompts de complexité croissante, même payload,
  même matrice et seeds appariées.
- **Sorties :** DiffQRCoder-paper et QRBTF-public-reproduction, chacune sans puis avec le même
  SR-MPGD, soit 16 évaluations.
- **Observabilité :** chaque pas de diffusion et chaque itération SR-MPGD, GIF, temps par phase,
  SSR exact, SSR original, MER, CLIP-aesthetic, CLIPScore, manifest et validation physique vide.
- **Transparence :** le backend QRBTF est fermé ; la branche publique utilise Monster v2 et
  Brightness ControlNet. Le QArt Reed–Solomon exact manque aussi côté DiffQRCoder ; une cible
  matricielle dérivée du Stage 1 est exportée et documentée.
- **État :** notebook et protocole implémentés ; les taux E011 restent à mesurer sur la RTX.
  Voir [`docs/e011-diffqrcoder-vs-qrbtf.md`](e011-diffqrcoder-vs-qrbtf.md).

### Résultats E011 et diagnostic post-hoc

- **Archive :** `20260722T124932Z-e011-diffqrcoder-vs-qrbtf-public-v1.tar.gz`, 16/16 sorties,
  640 frames, 20 GIF et 16 validations complètes.
- **Porte originale :** 0/16 image lisible sans transformation ; aucune promotion.
- **Esthétique/vitesse :** QRBTF base gagne avec 5,878 CLIP-aesthetic, 0,787 CLIPScore et 18,03 s,
  contre 4,928, 0,719 et 76,53 s pour DiffQRCoder.
- **SR-MPGD :** rejeté dans sa configuration commune 20 × 0,1 ; baisse esthétique d'environ 52 %,
  aucune lecture originale et artefacts dès la première mise à jour.
- **Cause principale :** quiet zone envahie et motifs fonctionnels trop texturés ; la MER moyenne
  masque ce défaut. Trois bases QRBTF n'ont qu'un module de données faux, mais 125 à 433 erreurs de
  quiet zone.
- **Diagnostic hors protocole :** une projection fonctionnelle tonale rend 7/8 bases lisibles par
  au moins un décodeur original ; meilleur cas 21/26 et 2/2 décodeurs.
- **Décision :** E012 doit intégrer la protection fonctionnelle dans la diffusion avant toute
  nouvelle recherche de modèle ou entraînement. Rapport :
  [`docs/e011-results-2026-07-22.md`](e011-results-2026-07-22.md).

Cette dernière numérotation est remplacée par l'audit ci-dessous : E012 mesure d'abord le vrai
SR-MPGD ; la protection fonctionnelle devient E013a afin de ne pas mélanger deux corrections.

## E012 - correction fidèle de SR-MPGD

- **Date :** 2026-07-22.
- **Déclencheur :** l'audit du notebook E011 montre que sa variante `SR-MPGD` réencode un PNG,
  vise le proxy QArt et réutilise la loss SRPG 500/3 avec LR 0,1. Elle ne correspond donc pas aux
  équations 12-14 et son échec ne permet pas de rejeter le vrai SR-MPGD.
- **Correction :** conserver le latent propre exact de la Stage 2, utiliser le QR binaire original,
  la SRL du dépôt public, un vrai LPIPS VGG, `gamma=1000` et `lambda=0,01`.
- **Incident QArt :** le premier brouillon utilisait un proxy visuel présenté à tort comme
  matriciel. `p1_simple` n'était plus décodable avant même Stage 2 : ce proxy ne préservait donc ni
  la matrice utile ni le payload et a été rejeté.
- **Décision v2 :** conditionner Stage 2 avec le QR binaire original valide, comme le chemin
  exécutable du dépôt public. L'absence du transformateur QArt Reed–Solomon est consignée comme
  écart au papier ; aucun faux proxy n'est substitué.
- **Itérations :** le papier ne donne pas leur nombre. Les états 0 à 20 sont persistés et validés,
  avec arrêt au premier 26/26 et sélection lexicographique si la porte n'est jamais atteinte.
- **Comparaison :** quatre prompts appariés, Stage 2 à 40 et 100 pas, chaque base sans puis avec
  SR-MPGD ; 16 résultats attendus.
- **Traçabilité :** latents safetensors, toutes les frames/GIF, validations, MER, CLIP-aesthetic,
  CLIPScore, temps, hash amont/latents, manifest, rapport et grille physique.
- **Limite maintenue :** QArt Reed-Solomon n'est pas publié ; le proxy matriciel reste déclaré et
  interdit l'étiquette de reproduction bit-à-bit de l'ensemble DiffQRCoder.
- **Modèle :** SD1.5/Cetus + QR Monster v2 reste figé pour isoler la correction. SDXL exige une
  autre pipeline et un ControlNet QR compatible ; il sera comparé séparément en E013b, après la
  baseline et l'expérience fonctionnelle E013a.
- **État :** implémenté et testé statiquement ; campagne RTX requise, aucun taux inventé. Voir
  [`docs/e012-faithful-srmpgd.md`](e012-faithful-srmpgd.md).

## E013 - géométrie exacte, SD 2.1 et politique contextuelle

- **Déclencheur :** l'audit E012 a montré une interaction très forte entre prompt et nombre de pas,
  tandis que SR-MPGD sélectionnait le plus souvent l'itération zéro.
- **Erreur géométrique identifiée :** le QR 740 px était ramené à 736 px alors que SRL supposait
  des modules de 20 px. Le cœur v3 n'était donc pas parfaitement aligné.
- **Correction :** cœur 29 × 20 = 580 px placé sans interpolation sur 744 px (padding 82) ou
  768 px (padding 94). Les métriques de modules utilisent ces bornes exactes.
- **Fondations :** DiffQRCoder SD 1.5 / QR Monster v2 contre SD 2.1 /
  `DionTimmer/controlnet_qrcode-control_v11p_sd21`.
- **Corrections séparées :** SR-MPGD papier et comportement du dépôt public ne sont plus agrégés
  sous un même nom.
- **Recherche :** Optuna TPE multiobjectif contraint, confirmation des meilleures recettes sur les
  quatre prompts, puis export d'un dataset de politique.
- **Garde ML :** CatBoost n'est entraîné qu'à partir de 100 observations et 12 succès stricts,
  avec validation groupée par prompt.
- **Livraison :** seule une image 26/26 peut être livrée ; sinon la politique essaie un autre
  candidat dans un budget plafonné, puis rejette.
- **Référence complète :**
  [`docs/e013-exact-geometry-sd21-policy.md`](e013-exact-geometry-sd21-policy.md).
- **Incident E013-01 :** l'essai Optuna 8 a produit un gradient SR-MPGD non fini dès l'itération
  zéro. La diffusion était valide, mais l'exception arrêtait l'étude. SR-MPGD conserve désormais
  l'état zéro, inscrit un `stop_reason` numérique et laisse l'étude continuer. Le quota Optuna
  compte les essais `COMPLETE` plutôt que tous les états, ce qui préserve 32 observations valides.

### Résultat et audit E013

- **Archive :** `20260723T081024Z-e013-exact-geometry-sd15-sd21-policy-v1.tar.gz`.
- **Complétude :** 80 lignes de baseline et 36 recherches. Les 32 recherches SD 1.5 sont
  présentes, mais seulement 4/32 recherches SD 2.1. La confirmation multi-prompt n'a pas été
  exécutée.
- **Lecture :** 1/116 ligne à 26/26 et 4/116 à 2/2 sur l'original. L'unique résultat strict est
  `p1_simple`, DiffQRCoder SD 1.5, 768 px, modules 20, 40 + 40 pas et SR-MPGD du dépôt public.
- **Erreur de porte :** seuls 5/12 QR témoins géométriques au masque 4 passent 26/26. Quarante-sept
  lignes utilisent exactement une combinaison masque 4 inéligible. Trente recherches emploient
  un autre masque sans témoin exact dans l'archive ; seules 39 lignes sont certifiées éligibles.
- **Modèles :** la baseline SD 1.5 obtient 16,03 % de validations en moyenne, contre 0,84 % pour
  SD 2.1. SD 2.1 améliore CLIP-aesthetic mais ne préserve pas le QR.
- **SR-MPGD papier :** amélioration dans 2/32 paires ; l'état initial reste sélectionné dans 30/32
  cas. Le SR-MPGD public produit le seul strict, mais améliore 4 paires, en dégrade 4 et modifie
  78,3 % des pixels en moyenne.
- **MER :** 39 sorties ont une erreur module égale à zéro, mais une seule est stricte. La métrique
  de centres ne prédit donc pas les décodeurs.
- **Politique :** CatBoost n'est pas entraîné, car le dataset ne contient qu'un positif strict.
- **Comparabilité :** le 99 % de l'article mesure 99 images originales lisibles par `qr-verify`
  sur 100 prompts, pas une porte 26/26. L'étape QArt Reed-Solomon décrite dans l'article reste
  absente du dépôt public et d'E013.
- **Décision :** suspendre SD 2.1, Optuna large et l'advisor. Conserver DiffQRCoder SD 1.5,
  recalibrer la porte sur les témoins, reproduire le protocole `qr-verify`, puis implémenter un
  vrai QArt avant toute nouvelle recherche.
- **Rapport complet :**
  [`docs/e013-results-and-project-audit-2026-07-23.md`](e013-results-and-project-audit-2026-07-23.md).

## E014–E016 — plan causal blueprint, latent, esthétique et surrogate

- **E014A :** intégration du QArt public réel au commit
  `6e0e00804a1994db7098432c19fadfc552071e30`. Sa correction L et son fragment `#…` sont
  explicitement séparés des variantes exact-payload. Comparaison appariée avec QR binaire,
  meilleur des huit masques légaux et blueprint Prooftag adaptatif.
- **E014B :** reconstruction déclarée `FreeQR-inspired`, avec ablations successives du canal,
  de la fenêtre temporelle et de la force. La cible bruitée est alignée sur le timestep suivant
  car le callback intervient après le pas DDIM.
- **E015 :** SD 1.5, SDXL et FLUX comparés comme références esthétiques uniquement. Aucun résultat
  ne sera interprété comme une compatibilité ControlNet QR.
- **Incident E015-01 :** le premier chargement FLUX a échoué en HTTP 401 après SD 1.5 et SDXL.
  `FLUX.1-schnell` est visible publiquement mais ses fichiers exigent l'acceptation des conditions
  et un jeton personnel. Le Deployment accepte désormais le secret Kubernetes optionnel
  `prooftag-huggingface`, et E015 contrôle les trois accès avant les chargements lourds.
- **Incident E015-02 :** après authentification et téléchargement des 23 fichiers, la construction
  du tokenizer T5 a échoué faute de `sentencepiece` et `protobuf`. Les versions `0.2.0` et `5.29.3`
  sont maintenant figées dans l'extra notebook, contrôlées pendant le build Docker et importées
  par E015 avant tout téléchargement. Les fichiers FLUX déjà reçus restent dans le PVC de cache.
- **Incident E015-03 :** `enable_model_cpu_offload()` a chargé le Transformer FLUX 12B BF16 entier
  sur la RTX, saturant 19,66/19,67 Gio avant le premier pas. E015 suit désormais le profil officiel
  bas VRAM : `enable_sequential_cpu_offload()`, slicing et tiling VAE. Ce choix plus lent, mais sans
  quantification, est enregistré comme `offload_mode=sequential_cpu`.
- **E016 :** labels demandés à OpenCV, ZBar et ZXing-cpp sur les dégradations réelles, split
  groupé anti-fuite, CNN multi-sorties, calibration et audit du gradient avec les vrais décodeurs.
- **Incident E016-01 :** deux workers `DataLoader` avec des lots 32×3×256×256 ont saturé le
  `/dev/shm` Docker historique et l'un d'eux est mort par `SIGBUS` pendant `optimizer.step()`.
  E016 choisit désormais zéro worker sous 256 Mio libres ou deux workers avec préchargement borné.
  Le pod notebook fournit un `emptyDir` mémoire de 2 Gio monté sur `/dev/shm`, et le mode réellement
  utilisé est conservé dans la carte du surrogate.
- **Résultat E014A :** douze contextes et 48 sorties Stage 2. Tous les blueprints passent 39/39
  avant diffusion. Après diffusion, l'adaptatif exact obtient 1/12 strict, 32,69 % de SSR moyen
  et 2/12 originaux lus par les trois moteurs. Il gagne environ 8 points sur les deux baselines
  exactes, mais peut aussi les dégrader fortement selon la seed.
- **Résultat E014B :** sur l'unique contexte facile déjà à 39/39, le canal latent 1 conserve 39/39
  et fait passer CLIP-aesthetic de 5,645 à 6,397. Cette expérience ne prouve pas encore une
  réparation ; elle doit être confirmée sur les onze contextes difficiles.
- **Résultat E015 :** Cetus SD 1.5 obtient le meilleur CLIP-aesthetic moyen (6,829), SDXL le
  meilleur CLIPScore (0,889) et FLUX 0,886. Ce sont des références esthétiques sans Stage 2 QR.
- **Incident E014A-01 :** deux f-strings échappées écrasaient les frames successives et les quinze
  sorties QArt brutes. Les traces et résultats finaux restent interprétables, mais pas les
  anciennes animations. Les fichiers sont désormais numérotés et `qart-screening.json` conserve
  chaque candidat.
- **Incident E014A-02 :** OpenCV 4.13 peut lever une exception native lorsque les quatre points
  détectés forment un contour d'aire nulle. Cette situation signifie « QR illisible » et ne doit
  pas arrêter la campagne. Le validateur convertit désormais toute exception d'un décodeur en
  lecture échouée et conserve son type/message dans `decoder_error`.
- **Répétition E014A du 27 juillet :** archive complète avec 800 frames JPEG, seize traces,
  seize GIF et soixante PNG QArt. Les seeds sont toutefois restées celles du premier lot. Une
  seule sortie est stricte : `p1_simple / exact_payload_mask_search_m`, 39/39.
- **Incident E014A-03 — appariement non déterministe :** le binaire et le meilleur masque ont la
  même condition PNG dans les quatre contextes, mais produisent des sorties différentes. Deux
  runs avec les mêmes seeds ne sont pas non plus bit-à-bit identiques. Le prochain protocole doit
  réinitialiser tous les RNG avant chaque branche et inclure un contrôle dupliqué ; sinon les
  différences doivent être estimées sur plusieurs répétitions.
- **Correction E014A v2 :** `SEED_OFFSET=30000` fournit automatiquement quatre nouveaux contextes.
  Python, NumPy, PyTorch CPU/CUDA, cuDNN et cuBLAS sont configurés avant chaque branche. Un binaire
  dupliqué, exclu de la sélection, contrôle les hashes de condition, latent initial, latent final
  et image finale dans `determinism-audit.json`. Les versions exactes du runtime sont manifestées.
- **Résultat E014A v2 :** archive
  `20260727T085645Z-e014a-deterministic-blueprint-pairing-v2.tar.gz`, SHA-256
  `E939E53ACD1E3253E276B4F58B29476E6CEE90D92BF8B0E0D2CC3CD37FA344FE`. Les 20 branches et leurs
  artefacts sont complets, mais les quatre contrôles dupliqués échouent : condition et latent
  initial identiques, latents finaux et images finales différents. La divergence commence au
  premier pas de Stage 2. Les MAE finales valent 5,2247, 2,1709, 0,1385 et 2,7721 pixels.
- **Performance E014A v2 :** aucune sortie à 39/39 ni à 3/3 sur l'original. L'adaptatif obtient
  20/156 validations, le meilleur masque et le duplicata 18/156, le binaire 17/156 et QArt 10/156.
  Le gain global de l'adaptatif est trop proche du bruit du contrôle pour être causal. Seul p3
  fournit un signal à confirmer : 10/39 pour l'adaptatif contre 0/39 pour les trois conditions
  exactes classiques.
- **Incident E014A-04 — non-déterminisme interne à Stage 2 :** les RNG et le latent initial sont
  innocentés. `_run_stage2` combine plusieurs chemins CUDA avec gradient en FP16 ; le mode
  déterministe n'était qu'en `warn_only=True`. Avant E014B complet ou E016, un diagnostic de 2 à
  5 pas doit exécuter le mode strict et ablater SRL, LPIPS, gradient checkpointing, précision du
  gradient et réutilisation de la pipeline. Les comparaisons par paire unique restent suspendues.
- **Résultat E014C v1 :** archive
  `20260727T100629Z-e014c-stage2-determinism-isolation-v1.tar.gz`, SHA-256
  `D66811E796BB2575A3493CE82687B52545951343CE20829D690DCE1E4132E9B1`. Les quatre profils sans
  checkpointing sont inconclusifs : OOM dans `torch.baddbmm` de l'attention UNet avant le premier
  pas, avec seulement 75 à 119 Mio libres. Ce n'est pas une erreur de déterminisme.
- **Signal E014C v1 :** la guidance complète avec checkpointing activé exécute deux répétitions de
  cinq pas en mode strict. Le latent initial, chacun des cinq hashes et le latent final sont
  bit-à-bit identiques ; aucune opération CUDA non déterministe n'est signalée.
- **Correction E014C v2 :** toutes les ablations conservent le checkpointing. La campagne ajoute
  deux contrôles complets à 40 pas : callback de hashes minimal, puis callback E014A avec décodage
  VAE, diagnostics et JPEG à chaque pas. Les OOM sont classées comme inconclusives et ne peuvent
  plus devenir automatiquement la « première erreur déterministe ».
- **Résultat E014C v2 :** archive
  `20260727T105243Z-e014c-stage2-determinism-isolation-v2.tar.gz`, SHA-256
  `FD80E48067D89E196996CC3489B43E8C336D12956FD11E4F5E50033525EBAD94`. SRL seule, LPIPS seule et
  leur combinaison sont exactes sur cinq pas. Les deux contrôles à quarante pas divergent tous
  deux à l'étape 7, timestep 801, après sept hashes identiques.
- **Callback E014A innocenté :** le contrôle minimal et le callback avec décodage VAE produisent
  les deux mêmes latents finaux A/B dans un ordre inversé. A et B diffèrent sur 36 282/36 864
  valeurs FP16, avec MAE 0,0361, RMS 0,0969 et maximum 2,4639.
- **Interprétation E014C v2 :** le timestep 801 est déterministe lorsqu'il est le premier pas du
  planning à cinq pas. La bifurcation exige donc l'historique des sept pas précédents ou un état
  interne alternant entre appels. Le mode PyTorch strict ne signale aucune opération fautive.
- **E014C v3 :** conserver le planning quarante pas mais interrompre après l'étape 7. Comparer un
  gradient nul connecté au graphe, SRL seule, LPIPS seule et la combinaison. Cette ablation
  localise le chemin fautif sans calculer les trente-deux pas suivants.
- **Résultat E014C v3 :** archive
  `20260727T120751Z-e014c-stage2-divergence-ablation-v3.tar.gz`, SHA-256
  `E1A8F1A866BD5AE63D9E327EE0300863A6944ED0A9C825B487561F6752526BBC`. Les quatre profils sont
  bit-à-bit identiques sur leurs deux répétitions jusqu'à l'étape 7. La combinaison retrouve deux
  fois l'état A de v2 et jamais l'état B.
- **Cause pratique E014C :** v2 termine le premier run complet avant le second, tandis que v3
  interrompt le premier après l'étape 7. Un état laissé par les étapes 8–39 influence donc l'appel
  suivant. Aucun chemin isolé — cœur, SRL, LPIPS ou combinaison — ne diverge par lui-même dans v3.
- **Décision E014C :** clôturer l'audit bit-à-bit. Les comparaisons ordinaires emploieront au moins
  trois répétitions, un ordre alterné/randomisé, un duplicata de contrôle et des intervalles de
  confiance. Une branche en processus/pod frais reste disponible pour les audits causaux rares.
  E014B peut reprendre selon ce protocole statistique.
- **Protocole E014B v2 :** le notebook 16 fixe quatre recettes — baseline DiffQRCoder, fusion
  canal 1 sur toute la trajectoire, fusion early et fusion avec petit gradient central — sur
  `p3_detailed`. Il exécute quatre répétitions dans un carré latin équilibré de Williams, afin
  d'équilibrer la position et la recette précédente, puis recharge une pipeline fraîche au début
  de chaque répétition. Chaque résultat conserve son latent, ses frames, les 39 validations,
  CLIPScore, CLIP-aesthetic et son temps.
- **Porte E014B v2 :** une fusion doit battre la baseline dans au moins trois répétitions, avoir
  un gain moyen supérieur à l'étendue observée de la baseline, ne jamais réduire son pire SSR et
  perdre au maximum 0,5 point de CLIP-aesthetic. À défaut, aucune confirmation multi-contexte
  n'est lancée. Ceci est un protocole enregistré, pas encore un résultat.
- **Résultat E014B v2 :** archive
  `20260727T123551Z-e014b-statistical-freeqr-confirmation-v2.tar.gz`, SHA-256
  `34A9E3164CAA54AD7AD63C0CB48AC7A731D33D336DE13D2E0EAC9B64AE4CD270`. Les seize runs, seize
  latents, seize validations, seize GIF et 144 frames sont complets. Les quatre répétitions
  partagent le même latent initial.
- **Gain E014B v2 :** sur `p3_detailed`, la baseline reste à 2/39 et 0/3 original. La fusion
  canal 1 alpha 0,15 sur quarante pas obtient 31/39 et 3/3 original dans quatre répétitions sur
  quatre. CLIP-aesthetic progresse de 4,578 à 5,418, tandis que CLIPScore recule de 0,630 à 0,571.
  Aucune sortie ne franchit encore 39/39.
- **Faiblesses E014B v2 :** `print_dot_loss` échoue pour les trois décodeurs ; OpenCV échoue aussi
  sous JPEG 90, luminosité basse/haute et contraste faible, et ZBar sous bruit gaussien. La
  fusion complète atteint 8/13 OpenCV, 11/13 ZBar et 12/13 ZXing-C++.
- **Incident E014B-01 — gradient sans effet :** la variante gradient calcule dix losses et ajoute
  environ 10,3 secondes, mais produit exactement les mêmes pixels que la fusion simple commune.
  Instrumenter norme et delta avant toute nouvelle tentative ; conserver la fusion simple.
- **Incident E014B-02 — porte trop permissive :** `fusion_early` est marquée promue malgré 0/3
  original. Le vainqueur reste correctement `fusion_all`, mais toute prochaine porte doit exiger
  la lecture originale par tous les décodeurs.
- **Historique E014B v2 :** la première `fusion_all`, placée après la baseline, diverge au pas 33
  des trois autres, avec seulement 0,099/255 de MAE finale et aucun changement de métrique. Le
  signal 2/39 vers 31/39 est donc robuste à cette variation.
- **Protocole E014B v3 :** le notebook 17 fige `fusion_all` canal 1, alpha 0,15, quarante pas et
  la compare à la baseline sur `p1_simple`, `p2_medium` et `p4_complex`. Quatre blocs appariés
  par contexte alternent deux fois chaque ordre, avec une pipeline fraîche par bloc, soit
  vingt-quatre diffusions.
- **Portes E014B v3 :** chaque contexte doit gagner dans au moins trois blocs, gagner en moyenne
  au moins 3/39, conserver son pire SSR, passer l'original 3/3 dans les quatre blocs et perdre au
  maximum 0,75 point de CLIP-aesthetic. Le statut production exige 39/39 sur les douze sorties
  fusionnées. Les révisions des modèles, le commit DiffQRCoder et le hash du pipeline sont
  manifestés.
- **Incident E014B-03 — URL single-file :** la première exécution v3 s'arrête avant tout run.
  Une URL `/resolve/<commit>/fichier` est réinterprétée par Diffusers comme un nom de fichier,
  produisant `/resolve/main/resolve/<commit>/fichier` et une 404. Une tentative
  `/blob/<commit>/fichier` échoue de la même façon, car le parseur 0.32 ne retire que
  `blob/main/`. Transmettre le commit avec `revision=` échoue aussi : Diffusers réutilise cette
  révision Cetus pour le dépôt de configuration SD 1.5. La forme finale télécharge séparément le
  checkpoint Cetus avec `hf_hub_download` et la configuration SD 1.5 avec `snapshot_download`,
  chacun à sa propre révision, puis fournit leurs chemins locaux à `from_single_file`. Aucun
  résultat partiel n'a été écrit.
- **Incident E014B-04 — assertion inter-blocs trop stricte :** les vingt-quatre diffusions v3 se
  terminent, puis le contrôle final trouve deux hashes initiaux entre les blocs de `p2_medium`.
  L'appariement baseline/fusion est néanmoins exact dans chacun des douze blocs ; c'est la
  condition requise par l'analyse appariée, tandis que l'ordre alterné traite la variabilité entre
  blocs. Le hard fail inter-blocs est remplacé par `initial-latent-audit.json`. Les avertissements
  FP16/CPU provenaient uniquement du déplacement de la pipeline pendant sa destruction ; la
  destruction libère désormais directement les objets CUDA.
- **Résultat E014B v3 :** archive
  `20260727T143614Z-e014b-multicontext-generalization-v3.tar.gz`, SHA-256
  `859BB6AC02959438532D77EC47C948FDCB6E8280E4D7940D96A9177F26A782A6`. Les vingt-quatre
  diffusions, latents, GIF, traces et 936 validations sont présents. Les trois blueprints passent
  39/39 avant diffusion.
- **Gain E014B v3 :** la fusion canal 1 alpha 0,15 fait passer le SSR global de 74/468
  (15,81 %) à 247/468 (52,78 %). Elle gagne dans les douze blocs : 28/39 sur `p1_simple`,
  17–20/39 sur `p2_medium` et 16/39 sur `p4_complex`.
- **Rejet E014B v3 :** aucune des douze sorties fusionnées ne passe l'original à 3/3 et aucune
  n'atteint 39/39. ZBar lit les douze originaux, ZXing-cpp seulement les quatre `p1_simple` et
  OpenCV aucun. La fusion est conservée comme composant de réparation, pas comme recette
  autonome de production.
- **Qualité E014B v3 :** CLIP-aesthetic progresse de 0,458 sur `p1` et 0,678 sur `p2`, mais baisse
  de 0,496 sur `p4`. CLIPScore baisse dans les trois contextes, jusqu'à -0,168 sur `p4`. Une alpha
  globale plus forte n'est donc pas la prochaine expérience recommandée.
- **Diagnostic E014B v3 :** `p1` présente zéro erreur de modules mesurée tout en échouant sous
  OpenCV. Les centres de modules ne suffisent pas : la prochaine réparation doit protéger
  contours, motifs fonctionnels et quiet zone, puis effectuer une rediffusion tardive faible
  bruit. Ce protocole ciblé devient E014D.
- **Rapport E014B v3 :**
  [`docs/e014b-v3-results-2026-07-28.md`](e014b-v3-results-2026-07-28.md).
- **Protocole E014D :** le notebook 18 reprend les meilleurs latents fusionnés de `p1`, `p2`,
  `p3` et `p4`, puis compare trois forces structurelles figées — 0,15, 0,30 et 0,45 — sur les huit
  derniers timesteps DDIM. Chaque candidat possède une pipeline fraîche et le même bruit tardif
  dans un contexte.
- **Masque E014D :** la quiet zone, les finders/séparateurs, timings, formats/versions et
  alignments sont projetés dans le latent à chaque pas. Les modules de données restent libres.
  La sortie finale est décodée directement du latent ; aucun collage ou correctif de pixels
  post-diffusion n'est autorisé.
- **Porte E014D :** priorité lexicographique à l'original 3/3, puis SSR 39 tests, pire décodeur,
  pire scénario, CLIP-aesthetic, CLIPScore et préservation. Une recette fixe doit réussir dans les
  quatre contextes et ne pas faire régresser `p3`. L'oracle contextuel est uniquement
  diagnostique.
- **Incident E014D-01 — schéma E014B v2 :** le premier chargement de `p3_detailed` échoue avant
  toute diffusion avec `KeyError: original_all`. E014B v2 enregistrait seulement
  `original_passed` et `original_total`, alors qu'E014B v3 ajoutait `original_all`. La sélection
  des sources normalise désormais les deux schémas, exige un total original strictement positif
  et utilise des accès tolérants pour les champs de classement. Aucun résultat E014D partiel
  n'avait encore été produit.
- **Incident E014D-02 — fenêtre DDIM non prise en charge :** la première réparation s'arrête
  avant son premier pas, car `DDIMScheduler.set_timesteps` de Diffusers 0.32.2 n'accepte pas
  l'argument `timesteps` utilisé par `retrieve_timesteps`. Un adaptateur DDIM local accepte
  uniquement le suffixe exact du planning complet à quarante pas, conserve
  `scheduler.num_inference_steps=40` pour que le calcul du pas précédent reste celui du planning
  d'origine, puis expose les huit pas tardifs à la boucle DiffQRCoder. Il refuse toute autre liste
  afin d'éviter de transformer silencieusement la dynamique DDIM. Aucun résultat E014D n'avait
  encore été écrit.
- **Résultat E014D :** archive
  `20260728T080203Z-e014d-functional-late-rediffusion-v1.tar.gz`, SHA-256
  `7CC59CD2D65373070BAC653863B00CA8FB4129D43548242885E2528CC6AD8332`. Les seize sorties,
  seize latents, douze GIF, douze traces et 96 frames sont complets. Les entrées sont appariées
  dans chaque contexte et les traces exécutent le suffixe DDIM `176,151,126,101,76,51,26,1`.
- **Gain E014D :** les douze rediffusions passent l'original 3/3. La recette fixe 0,30 passe
  151/156 validations, soit 96,79 %, contre 95/156 et 60,90 % pour le contrôle E014B. Elle atteint
  39/39 sur `p1` et `p4`, 37/39 sur `p2` et 36/39 sur `p3`.
- **Rejet E014D :** la porte fixe reste fausse. CLIP-aesthetic moyen baisse de 5,165 à 4,017,
  la perte maximale atteint 1,339 pour une limite à 0,75 et la modification absolue atteint
  0,250 pour une limite à 0,18. Visuellement, les sorties sont nettement plus QR et perdent leurs
  cadres et transitions artistiques.
- **Échecs E014D :** les seize échecs restants se concentrent sur `print_dot_loss` (6),
  `downscale_75` (5), `print_dot_gain` (4) et `noise_gaussian` (1). ZXing-C++ réussit tous les
  tests E014D ; ZBar échoue neuf fois et OpenCV sept fois.
- **Décision E014D :** ne pas entraîner le sélecteur contextuel sur quatre prompts. E014E doit
  d'abord séparer rediffusion, fusion globale, masque fonctionnel et longueur de fenêtre avec des
  forces plus faibles. Le but est de conserver le 3/3 original en récupérant l'esthétique.
- **Rapport E014D :**
  [`docs/e014d-results-2026-07-28.md`](e014d-results-2026-07-28.md).
- **Protocole E014E :** le notebook 19 sépare la rediffusion DiffQRCoder, la fusion globale et
  le masque fonctionnel sur `p2`/`p3` avec quatre pas. La référence 0,15/0,15 est forcée dans la
  phase B avec les deux meilleures recettes faibles, puis les fenêtres 2/4/6/8 sont comparées sur
  les quatre contextes. `p1`/`p4` servent de holdout après le screening.
- **Portes E014E :** priorité à l'original 3/3 et au SSR, mais une recette fixe doit aussi limiter
  toute perte CLIP-aesthetic à 0,75 et toute modification absolue à 0,18. La campagne produit
  74 lignes, conserve les entrées appariées, journalise les erreurs et interdit explicitement
  l'entraînement d'un sélecteur sur quatre contextes.
- **Protocole détaillé E014E :**
  [`docs/e014e-protocol-2026-07-28.md`](e014e-protocol-2026-07-28.md).
- **Résultat E014E :** archive
  `20260728T085500Z-e014e-mechanism-window-ablation-v1.tar.gz`, SHA-256
  `34FBB51CEFB1EA58030BA01980590F12909E3DBE7A4E0E74CC0B86B4EE6B8944`. La campagne
  contient 74 candidats complets et aucun événement dans `errors.jsonl`.
- **Mécanisme E014E :** la fusion sous le masque des motifs fonctionnels produit l'essentiel du
  gain. La fusion globale seule apporte peu et peut dégrader la lecture originale. La recette
  douce `combined_a06_s10`, écartée de la phase B par le classement, reste une candidate de
  frontière et doit revenir dans E014F.
- **Fenêtre E014E :** quatre pas maximisent le SSR ; six et huit pas sont dominés et détruisent
  davantage l'image. Deux pas maximisent la préservation. Trois pas, non testés, deviennent le
  point prioritaire d'E014F.
- **Candidat équilibré E014E :** `combined_a10_s15` à deux pas passe 148/156 validations
  (94,87 %), l'original 3/3 dans les quatre contextes, avec CLIP-aesthetic moyen 4,915 et une
  modification maximale de 0,083. Il franchit la porte logicielle de faible dommage.
- **Candidat robuste E014E :** `combined_a15_s15` à quatre pas passe 153/156 validations
  (98,08 %) et l'original 3/3 partout, mais sa perte esthétique maximale de 0,938 dépasse la
  limite 0,75. Il reste expérimental.
- **Décision E014E :** ne pas entraîner de sélecteur. E014F comparera les fenêtres 2/3/4 et
  réintroduira `combined_a06_s10` sur des prompts, graines et payloads inconnus. Une cascade
  générer-valider-escalader est préférable au mini-modèle tant que le jeu de données n'existe pas.
- **Rapport E014E :**
  [`docs/e014e-results-2026-07-28.md`](e014e-results-2026-07-28.md).
- **Protocole E014F :** vingt-quatre contextes inconnus sont formés par douze nouveaux prompts,
  deux graines et six payloads. Chaque source régénère Stage 1, blueprint adaptatif exact et
  Stage 2 FreeQR avant toute réparation.
- **Comparaison E014F :** `mask_s15`, `combined_a06_s10`, `combined_a10_s15` et
  `combined_a15_s15` sont comparées avec deux, trois et quatre pas. Les entrées sont appariées et
  chaque candidat possède une pipeline fraîche.
- **Validation E014F :** seize contextes servent à classer une recette fixe et huit contextes,
  issus de quatre prompts entiers, restent holdout. Une cascade préenregistrée ne livre que le
  premier candidat passant 39/39. Aucun sélecteur appris ni aucune projection de pixels ne sont
  autorisés.
- **Protocole détaillé E014F :**
  [`docs/e014f-protocol-2026-07-28.md`](e014f-protocol-2026-07-28.md).
- **Observation QArt :** les trois répétitions d'un seuil produisent le même SHA-256 dans les
  quatre contextes. Le CLI se comporte donc de façon déterministe dans cette image Docker ; cinq
  seuils donnent cinq candidats uniques par prompt.
- **Incident E016-02 — invalidation :** la clé d'image omettait la variante source. Quatre lignes
  pointaient donc vers le même JPEG écrasé ; 93/156 images avaient des labels contradictoires.
  Les QArt canoniques étaient en plus étiquetés contre le payload exact. Le TorchScript archivé
  est rejeté.
- **Incident E016-03 :** OpenCV n'avait que deux groupes positifs sur douze et aucun positif dans
  le test retenu. E016 contrôle désormais les classes par groupe et recherche un split où les
  trois partitions contiennent les deux classes pour chaque décodeur. Aucun modèle n'est exporté
  si les portes échouent.
- **Décision :** conserver l'adaptatif et la fusion canal 1 comme hypothèses, rejouer la fusion sur
  les cas difficiles, puis reconstruire E016 avec au moins 24 contextes et six groupes positifs et
  négatifs par moteur avant tout fine-tuning.
- **Rapport :**
  [`docs/e014-e016-results-audit-2026-07-27.md`](e014-e016-results-audit-2026-07-27.md).
- **Diagnostic visuel E014F :** sur 24 paires Stage 1/Stage 2, la netteté moyenne passe de
  942,10 à 535,33, la saturation moyenne de 0,3616 à 0,5219, les pixels fortement saturés de
  0,0061 à 0,2121 et les pixels écrêtés de 0,0016 à 0,1088. Le défaut visible n'est donc pas
  seulement une impression subjective.
- **Attribution E014F :** la fusion FreeQR ajoutée à `alpha=0,15`, canal 1, pendant les 40 pas
  n'appartient pas au Stage 2 publié par DiffQRCoder. Les résultats appariés E014B montrent
  toutefois qu'elle ne suffit pas à expliquer la saturation globale ; le redémarrage bruité et la
  reconstruction Stage 2 constituent le mécanisme principal, avec une interaction par prompt.
- **Décision d'outillage :** les nouvelles ablations courantes passent dans le laboratoire Web
  `/lab`. Chaque méthode, prompt et seed devient un essai persistant ; l'exécution GPU est
  séquentielle et les notes humaines et scans physiques rejoignent les métriques automatiques.
- **Incident Web Lab 01 — cache d'assets :** le navigateur conservait l'ancienne feuille de style
  et masquait mal le bouton d'arrêt d'une campagne terminée. Les assets possèdent désormais une
  version de cache, la règle HTML `[hidden]` est forcée, et les cartes d'essai sont activables au
  clavier.
- **Validation Web Lab :** smoke test complet avec le backend QR : création, progression,
  résultat 1/1 strict, affichage de l'artefact, export CSV et notation humaine. Les tests API,
  schéma laboratoire, repository et SRPG ciblés passent ; les deux tests GPU indisponibles
  localement restent explicitement ignorés.
- **Documentation Web Lab :** [`docs/web-lab.md`](web-lab.md).
- **Incident Web Lab 02 — faux gagnant réparé :** le laboratoire réutilisait le sélecteur de
  livraison. Dès que le Stage 1 ou une projection de centres passait la porte stricte, la sortie
  la moins modifiée pouvait remplacer le candidat SRPG dans `final.png`. Les métriques parfaites
  décrivaient alors le QR réparé et non la méthode artistique annoncée.
- **Correction Web Lab 02 :** chaque méthode déclare désormais une `output_variant`. Les profils
  de recherche forcent `raw` ou `srpg`, arrêtent l'itération avant les réparations déterministes,
  valident le candidat réel même lorsqu'il est rejeté et persistent la variante effectivement
  sélectionnée.
- **Appariement Web Lab 02 :** le Stage 1 est mis en cache par modèle, conditionnement,
  paramètres, prompt, seed et correction d'erreur. Une méthode compatible réutilise exactement
  le même raster CPU au lieu de relancer la diffusion. L'export indique la réutilisation et le
  run source.
- **Affichage Web Lab 02 :** la galerie nomme explicitement le résultat évalué, montre le Stage 1
  partagé lorsqu'il diffère et masque les doublons binaires identiques. Les anciennes campagnes
  restent consultables mais ne deviennent pas rétroactivement des comparaisons causales.
- **Diagnostic Web Lab 03 — Stage 2 destructif :** sur les exemples `courtyard` et `station`,
  l'ancien profil SRPG à `strength=1,0` rediffuse réellement 40 pas. Le résultat perd la netteté
  et parfois le contenu du prompt ; il reste rejeté malgré une MER localement plus faible.
  La porte `max_mean_absolute_change` intervient seulement après la génération et ne protège pas
  le Stage 1 pendant la boucle.
- **Confrontation au papier :** DiffQRCoder décrit bien un Stage 2 démarrant depuis le Stage 1
  rebruité, avec SRL `λ1=500`, LPIPS `λ2=3`, Cetus-Mix, QR Monster v2 à `1,35` et une condition
  QArt. Le transformateur QArt de l'article n'est pas livré par le dépôt public ; le profil local
  ne peut donc pas être qualifié de reproduction « papier ».
- **Correction Web Lab 03 :** le profil complet devient l'ablation désactivée
  `srpg_full_restart`. Les profils actifs `srpg_late_2` et `srpg_late_4` utilisent respectivement
  `strength=0,05` et `0,10` sur un calendrier de 40 pas, soit deux et quatre pas réellement
  exécutés. Ils enregistrent le nombre de pas demandé, le nombre effectif, la force de redémarrage
  et la différence moyenne avec le Stage 1. Il s'agit d'hypothèses à valider, pas d'une nouvelle
  revendication de 99 %.
- **Correction de modèle Web Lab 03 :** les profils génératifs ne dépendent plus des valeurs
  globales SD 1.5 + Dion de la ConfigMap. Ils figent Cetus-Mix Whalefall et QR Monster v2 ; le
  backend sait charger Cetus depuis son fichier Safetensors avec la configuration SD 1.5.
- **Déploiement Web Lab 03 :** `scripts/deploy-app-image.sh` construit un tag dérivé du commit,
  l'importe dans K3s, met à jour l'API et l'initContainer, puis vérifie l'image et le profil dans
  le pod. Le tag mutable `dev` n'est plus utilisé pour cette procédure.
- **Incident Web Lab 04 — pas SRPG ambigus :** le seul champ « Steps » visible modifiait les pas
  du Stage 1. Les pas SRPG et la force img2img restaient dans le JSON ; une configuration
  `srpg_steps=40`, `srpg_strength=0,10` affichait donc correctement quatre artefacts, mais
  l'interface donnait l'impression que 40 pas avaient été ignorés.
- **Correction Web Lab 04 :** les contrôles Stage 1 et Stage 2 sont séparés. Le formulaire affiche
  et persiste les pas SRPG planifiés, la force de redémarrage, le nombre effectif
  `floor(steps × strength)`, ControlNet, les poids QR/perceptuel/fonctionnel et la limite de
  gradient. Les mêmes valeurs résolues sont ajoutées aux métriques du résultat. Une combinaison
  qui produirait zéro pas est désormais rejetée avant l'inférence.
- **Audit Web Lab 05 — SR-MPGD absent :** l'export
  `prooftag-lab-fe011965-82af-483a-8350-2fcee96f82a9.csv` contient 48 essais répartis entre
  `controlnet_raw`, `qr_reference`, `srpg_late_2` et `srpg_late_4`, mais aucune variante
  SR-MPGD. Les deux profils SRPG génératifs ont zéro payload original exact et une MER moyenne
  proche de 25 %, tandis que le témoin binaire passe 12/12.
- **Cause Web Lab 05 :** `run_srmpgd` existait et reproduisait les équations 12-14, mais le
  backend Web appelait uniquement l'ancien `refine_candidate_latent`. Celui-ci réencodait le PNG
  et n'avait pas accès au latent propre du Stage 2. `SRPGResult` supprimait en outre ce latent à
  la fin de la boucle.
- **Correction Web Lab 05 :** `SRPGResult` conserve maintenant le latent propre exact. Le nouveau
  profil `srpg_late_4_srmpgd` transmet directement ce tenseur à `run_srmpgd`, teste et sauvegarde
  tous les états, puis sélectionne scannabilité/MER avant LPIPS. La validation interne utilise
  la même matrice de décodeurs que la porte finale du service.
- **Traçabilité Web Lab 05 :** l'interface sépare `SR-MPGD papier` de l'ancien raffinement latent,
  expose itérations, gamma et lambda LPIPS, exporte les paramètres et résultats dans les colonnes
  `quality_srmpgd_*`, et ajoute métriques Prometheus, logs détaillés et images
  `srmpgd_iteration_XX`.
- **Limite Web Lab 05 :** seul le post-traitement SR-MPGD suit les équations du papier. Le profil
  actif conserve un Stage 2 tardif Prooftag ; le QArt Reed-Solomon exact de la Figure 3 reste
  indisponible dans le dépôt public.
- **Incident Web Lab 06 — SR-MPGD détruit l'esthétique :** l'export
  `prooftag-lab-4decd849-9aa5-436e-bb27-2fd52ff0138f.csv` contient huit sorties SR-MPGD, toutes
  rejetées. Le MER moyen passe bien d'environ 31,6 % à 3,2 %, mais le CLIP-aesthetic moyen tombe
  d'environ 5,65 pour le Stage 2 à 4,05. Les images présentent des contours bleus et une texture
  QR haute fréquence absents des résultats du papier.
- **Causes Web Lab 06 :** (1) la SRL divisait par le nombre de modules encore actifs au lieu des
  `N` modules de l'équation 6, maintenant ainsi un gradient fort jusque sur les derniers modules ;
  (2) SRL et LPIPS incluaient la quiet zone alors que le code officiel applique `crop_padding` ;
  (3) `gamma=1000` était appliqué pendant 20 itérations à un Stage 2 à 0 % de SSR et 20–38 % de
  MER, alors que le tableau 7 présente SR-MPGD comme finition d'un Stage 2 déjà à 88–94 % de SSR ;
  (4) le profil actif ne lançait que quatre pas tardifs et n'était donc pas le Stage 2 de 40 pas.
- **Correction Web Lab 06 :** SRL est désormais normalisée par tous les modules, la quiet zone est
  retirée automatiquement des pertes SRPG/SR-MPGD, et les seuils papier redeviennent symétriques
  à 0,5. L'ablation tardive SR-MPGD est désactivée. Deux profils appariés exécutent maintenant le
  chemin public complet de 40 pas, avec puis sans SR-MPGD. Une précondition à 10 % de MER empêche
  SR-MPGD d'essayer de reconstruire une image trop éloignée ; l'état initial est alors conservé
  et la raison est tracée.
- **Limite maintenue :** le dépôt DiffQRCoder public ne fournit toujours pas le transformateur
  QArt Reed-Solomon de la Figure 3. Les nouveaux profils complets utilisent donc le QR binaire
  valide, comme le chemin exécutable public, et ne revendiquent pas encore les 99–100 % du papier.
- **Incident Web Lab 07 — quiet zone absente de la livraison :** l'export
  `prooftag-lab-996ce01f-3003-4e70-a3f1-a4b37486b7cf.csv` contient huit prompts. Pour le profil
  SR-MPGD, le cœur optimisé est en moyenne à 2,7 % de MER, mais l'image réellement livrée reste
  à 21,8 % de MER et aucune sortie n'est acceptée.
- **Cause Web Lab 07 :** SRL et LPIPS ignorent volontairement le padding, tandis que la diffusion
  peint cette zone. SR-MPGD voit donc un cœur presque correct, mais les décodeurs voient une
  image dépourvue de quiet zone uniforme.
- **Correction Web Lab 07 :** les quatre modules périphériques sont remis en blanc après chaque
  décodage final, avant validation et livraison. Aucun pixel du cœur artistique n'est modifié.
  Les métriques séparent désormais le MER du cœur, celui de la quiet zone et celui de l'image
  réellement livrée. Une nouvelle campagne est obligatoire ; les résultats historiques ne sont
  pas réinterprétés.
- **Diagnostic Web Lab 08 — marge corrigée, détection toujours absente :** l'export
  `prooftag-lab-67489f0d-0391-4a68-87f9-46dc7954cc08.csv` contient douze essais appariés sur
  `courtyard` et `station`. La restauration blanche ramène le MER de la quiet zone à zéro.
  Cependant, `srpg_late_2` et `srpg_late_4` plafonnent à 17,95 % de SSR moyen et tous les autres
  candidats artistiques restent à 0 %. Sur `station`, le MER tombe à environ 0,37 % sans aucune
  lecture : le MER global ne suffit donc pas à mesurer la capacité du détecteur à localiser les
  trois finders.
- **Cause Web Lab 08 :** le contour, le contraste local et la géométrie interne des finders
  restent trop décoratifs. Le cadre blanc obligatoire les isole mais produit en plus une rupture
  esthétique visible. Augmenter encore SR-MPGD ne résout pas cette détection et risque de
  dégrader l'image.
- **Correction Web Lab 08 :** la marge devient une couleur uniforme claire tirée de la palette,
  avec luminance minimale réglable. Deux métriques séparent maintenant motifs fonctionnels et
  données. Un profil apparié tonifie uniquement finders, séparateurs, timing, format et
  alignements ; aucun module de données n'est projeté. Un second profil applique ensuite le vrai
  SR-MPGD au latent propre pour mesurer son apport marginal.
- **Justification externe Web Lab 08 :** Face2QR renforce explicitement finder et alignment
  patterns avant son raffinement latent. Text2QR sépare également amélioration de scannabilité
  et raffinement esthétique. Ces travaux motivent le test ciblé, sans promettre leurs chiffres
  sur les prompts génériques Prooftag.
- **Décision Web Lab 08 :** désactiver par défaut les redémarrages complets et l'ancienne fenêtre
  quatre pas. La campagne suivante doit comparer quatre sorties partageant le même Stage 1 :
  brute, SRPG tardif, SRPG tardif avec motifs fonctionnels et la même sortie suivie de SR-MPGD.
  Le critère primaire reste le payload exact/SSR ; CLIP-aesthetic et CLIPScore ne départagent que
  les candidats ayant franchi la porte de lecture.
# Reprise Web Lab DiffQRCoder — 29 juillet 2026

- **Décision :** abandonner dans le laboratoire actif les mélanges FreeQR,
  Nacholmo, rediffusion locale, projection de modules et réparation
  déterministe. Ils restent dans l’historique Git et dans les anciens rapports,
  mais ne sont plus proposés dans `/lab`.
- **Référence exécutable :** dépôt `jwliao1209/DiffQRCoder`, commit
  `e24ea73ee2e13c7e6e87cb422e8b11784e70ae00`, Cetus-Mix Whalefall,
  QR Monster v2 et DDIM.
- **Géométrie :** QR v3, correction M par défaut, masque 4, 20 px par module,
  source 740 px, preprocessing 736 px et crop public de 78 px.
- **Sorties :** QR témoin, Stage 1 brut, Stage 2 SRPG, Stage 2 SRPG + SR-MPGD.
  Le candidat demandé est toujours évalué tel quel.
- **Corrections nécessaires :** `torch.stack` préserve le gradient de la perte
  VGG que le dépôt détachait avec `torch.tensor`; `srmpgd_lr` est transmis
  explicitement à `_run_stage2`.
- **Limite maintenue :** le transformateur QArt Reed–Solomon du papier n’est
  pas publié. Cette reprise reproduit le chemin du dépôt public et ne revendique
  pas une reproduction bit-à-bit de toute la Figure 3.
- **Évaluation :** validation automatique par décodeur et dégradation, SSR,
  payload original, MER, CLIP-aesthetic, CLIPScore et temps ; puis deux verdicts
  humains distincts, esthétique et scan téléphone.
- **Objectif :** constituer une base comparable multi-prompts/multi-seeds avant
  de chercher une recette générale. Le QR témoin est exclu des scores
  artistiques et aucune réparation cachée n’est autorisée.

## Audit du Stage 2 et reconstruction papier — 29 juillet 2026

- **Campagne diagnostiquée :**
  `prooftag-lab-20c302a1-dab0-4813-9f4f-103968dcc256.csv`, quatre prompts,
  une seed et quatre sorties appariées. Le Stage 1 atteint en moyenne 0,6 % de
  SSR et 6,27 de CLIP-aesthetic ; SRPG 9,0 % et 3,98 ; SR-MPGD 4,5 % et 3,32.
  Le Stage 2 change entre 93 % et 99,95 % des pixels du Stage 1. Les images
  saturées observées ne sont donc pas un simple problème de note humaine.
- **Cause principale vérifiée dans le commit amont épinglé :** la méthode
  publique `_run_stage2` appelle `prepare_latents` et repart d’un bruit
  aléatoire. L’image du Stage 1 n’intervient alors que dans la perte perceptuelle,
  alors que l’algorithme 1 et l’équation 9 demandent
  `sqrt(alpha_T) E(x̂) + sqrt(1-alpha_T) ε`.
- **Deuxième écart :** la Figure 3 demande `Qart(x̂, y)`, mais le dépôt ne
  fournit pas le constructeur QArt. Utiliser directement le QR binaire force
  ControlNet à reconstruire une grille éloignée de l’image et accentue la
  destruction du Stage 1.
- **Troisième écart :** le SR-MPGD public réemploie `self.srpg.compute_loss`,
  donc les poids SRG/PG du Stage 2, et son appel haut niveau ne transmet pas
  correctement `srmpgd_lr`. Ce n’est pas l’objectif séparé de l’équation 13.
- **Correction Stage 2 :** le backend encode maintenant le raster Stage 1 avec
  le VAE, ajoute le bruit DDIM au timestep de départ et injecte explicitement ce
  latent dans `_run_stage2`. Le mode historique `public_random` reste disponible
  comme ablation, jamais comme valeur papier implicite.
- **Correction QArt :** une cible déterministe est reconstruite sur le canvas
  736 px. Elle préserve l’œuvre et sa périphérie, copie exactement les modules
  fonctionnels et déplace seulement les centres des modules de données de part
  et d’autre des seuils 0,45/0,65. Elle est nommée « QArt reconstruite », car
  elle ne prétend pas être le code Reed–Solomon privé des auteurs.
- **Correction SR-MPGD :** le latent propre de sortie du Stage 2 est optimisé
  avec `LSR + 0,01 × LPIPS`, `gamma=1000`, VGG et crop de 78 px. Chaque état est
  décodé, validé par les vrais décodeurs et classé ; le meilleur est conservé.
  Un MER initial maximal, les gradients non finis et la validation stricte
  déclenchent des arrêts explicites.
- **Garde-fous et transparence :** aucun cadre blanc, QR binaire final ou
  projection déterministe n’est ajouté. Le Web Lab affiche l’initialisation
  réellement employée, les pas, l’erreur des centres QArt, la variation au
  Stage 1, la saturation, la divergence et l’itération SR-MPGD retenue. Une
  divergence reste visible et rejetable ; elle n’est pas maquillée.
- **Validation locale avant GPU :** tests unitaires de géométrie QArt, motifs
  fonctionnels, absence de cadre blanc, schéma API, paramètres du profil,
  dépendances épinglées, lint Python et syntaxe JavaScript. La validation
  numérique CUDA doit ensuite être faite sur la RTX Ada avec une nouvelle
  campagne appariée ; les anciens scores ne sont pas réinterprétés.
- **Incident GPU 09 — calendrier DDIM personnalisé refusé :** la première
  campagne déployée termine avec huit erreurs Stage 2 en environ une seconde.
  L’API conserve l’exception exacte :
  `DDIMScheduler.set_timesteps does not support custom timestep schedules`.
  Les Stage 1 sont valides ; aucune itération SRPG ni SR-MPGD n’a commencé.
- **Correction GPU 09 :** le backend ne transmet plus `timesteps=[...]` à
  Diffusers 0.32. Pour 40/40 pas, il laisse DDIM construire son calendrier
  normal. Pour un redémarrage partiel, il construit d’abord ce même calendrier,
  en installe le suffixe directement sur le scheduler et conserve les 40 pas
  comme dénominateur du calcul manuel amont. Un test de non-régression vérifie
  à la fois le suffixe et l’intervalle. Le Web Lab affiche désormais le message
  d’erreur des générations sans image au lieu d’une vignette cassée.
- **Incident scientifique 10 — fausse cible QArt :** l’artefact
  `stage2_control_target` a montré une image Stage 1 colorée sur laquelle étaient
  posés les motifs fonctionnels et des petits centres de modules. Cette
  construction n’est pas la transformation Reed–Solomon QArt décrite en
  annexe C.1. Le dépôt public transmet son argument `qrcode` à ControlNet et le
  rebinarise dans `ScanningRobustLoss`; notre hybride devenait donc une cible
  SRL dont la matrice et le payload n’étaient pas garantis.
- **Correction scientifique 10 :** le pseudo-QArt est supprimé du backend, de
  la configuration Kubernetes et de l’interface. Stage 2 reçoit désormais le
  QR binaire exact pour ControlNet et SRL. Ce repli ne reproduit pas la
  proximité visuelle apportée par le vrai QArt du papier, mais il garantit que
  la guidance optimise bien le payload Prooftag demandé. Un véritable QArt ne
  pourra être réintroduit qu’après validation exacte du payload avant diffusion.
  Les campagnes produites avec la cible hybride restent historiques et ne
  doivent pas être comparées aux nouvelles campagnes.

## E018 — appariement strict SRPG → SR-MPGD — 4 août 2026

- **Déclencheur :** l'audit de la campagne `831e74cb` montre des SHA de latent
  Stage 2 différents entre SRPG et SR-MPGD sur les dix prompts. Les bons scores
  SR-MPGD (9/10 scans téléphone) ne prouvaient donc pas l'effet du
  post-traitement sur une même image SRPG.
- **Correction :** la clé de partage est reconstruite avec les seuls paramètres
  qui modifient mathématiquement le Stage 2. Les options d'aperçu, les gardes de
  sélection et tous les paramètres SR-MPGD n'en font pas partie.
- **Contrat dur :** SR-MPGD ne peut plus recalculer silencieusement un Stage 2.
  Il doit trouver une source SRPG antérieure dans la campagne, importer son
  latent et réussir deux contrôles SHA-256. Sinon l'essai passe en erreur avec
  la cause exacte.
- **Traçabilité :** `run_id`, méthode source, SHA source, SHA utilisé et statut
  d'appariement sont enregistrés dans la provenance, exportés en CSV et affichés
  dans le Web Lab.
- **Protocole suivant :** QR témoin, Stage 1, SRPG 0,65 puis le SR-MPGD strict
  sont activés. La campagne suivante doit utiliser au moins trois seeds et trois
  essais téléphone documentés par image.
- **Validation locale :** suite pytest complète réussie, tests d'intégration de
  l'absence de source et de la propagation exacte `run_id`/SHA réussis, Ruff
  réussi et interface vérifiée dans le navigateur local.

## E019 — SR-MPGD borné — 4 août 2026

- **Déclencheur :** malgré l'appariement E018, les états SR-MPGD retenus peuvent
  encore contenir des taches. Le pas `gamma × gradient` n'était pas borné et le
  classement acceptait la dégradation visuelle dès que le SSR progressait.
- **Protection :** borne RMS par pas et cumulée, gardes LPIPS, changement moyen,
  saturation et écrêtage. L'état SRPG initial reste toujours candidat.
- **Sélection :** un état hors garde est inéligible. Parmi les états sûrs, la
  lecture reste prioritaire puis LPIPS et le changement visuel départagent les
  candidats avant MER/SRL.
- **Profil prudent :** 4 itérations, gamma 100 et LPIPS 0,10. Ce n'est pas une
  recette finale, mais le point de départ de la recherche E019.
- **Recherche :** cinq lots appariés couvrent 100 configurations : 5 nombres
  d'itérations × 5 gamma × 4 poids LPIPS. Un seul lot est lancé à la fois.

## E020 — trajectoire complète et loss robuste — 4 août 2026

- **Résultat E019B :** gamma 10/30/100/300/1000 a bien modifié les pas, mais
  les cinq sorties ont retenu l’état zéro. MER initial 0 %, SSR 2,56 % et
  original 0/3 : la métrique module-centre ne décrit pas l’échec des scanners.
- **Arrêt de la grille :** aucun nouveau balayage massif n’est lancé tant que
  la trajectoire de loss n’est pas observable.
- **Traçabilité :** chaque itération exporte SRL, MER, SSR, LPIPS, déplacement,
  garde, gain QR et composantes robustes dans `srmpgd_trace.json` et le Web Lab.
- **Ablation :** `diffqrcoder_srmpgd_robust` partage strictement le latent SRPG
  du témoin et moyenne la loss publique sur flou, réduction, luminosité et
  contraste. La sortie sûre conserve toujours l’état zéro comme candidat.
# E021 — prompts atypiques hors distribution

- **Date** : 4 août 2026.
- **But** : tester la dépendance au prompt sur douze sujets jamais employés dans les
  campagnes historiques.
- **Contrôle** : quatre sorties appariées par prompt, avec le même Stage 1 et le même
  latent Stage 2 pour SRPG, SR-MPGD officiel et SR-MPGD robuste.
- **Lots** : six prompts de diversité puis six prompts de stress visuel ; un seed avant
  toute réplication.
- **Décision attendue** : mesurer séparément généralisation esthétique, SSR et réponse
  des deux losses, sans présenter une baisse de loss comme une réussite de scan.
- **Protocole** : `docs/e021-atypical-prompt-generalization.md`.

## E023 — métriques logicielles honnêtes — 5 août 2026

- **Déclencheur :** sur E022, le SSR et le proxy logiciel corrèlent mal avec les
  vingt lectures téléphone. Ils ne peuvent plus être nommés ni optimisés comme
  une probabilité de scan physique.
- **Comparaison contrôlée :** E023 reprend les dix prompts, la seed 61001 et les
  deux recettes E022 sans modifier la génération. Seule la mesure change.
- **Lecture logicielle :** ajout du décodeur WeChatQRCode d'OpenCV contrib, de
  son détecteur CNN et de sa super-résolution. Les quatre poids officiels sont
  épinglés et vérifiés au build.
- **Esthétique :** ajout de HPS v2.1, plus proche des préférences humaines que
  l'ancien régresseur CLIP-AES. Son dépôt officiel est épinglé car le wheel
  PyPI omet un fichier BPE ; l'inférence est forcée sur CPU.
- **Contrat d'affichage :** SSR devient « indice synthétique » et le phone
  proxy devient « proxy de prétraitements logiciels ». Aucun des deux n'est
  présenté comme un taux téléphone.
- **Limite :** en l'absence de banc caméra, WeChat et le consensus des
  décodeurs servent de filtre conservateur, pas de vérité terrain physique.
- **Protocole :** `docs/e023-honest-software-metrics.md`.

## E024 — QR-Verify comme autorité unique — 5 août 2026

- **Décision :** `antfu/qr-verify@0.2.0` devient le seul validateur logiciel du
  Web Lab. Les anciens SSR, proxies et consensus de décodeurs sortent du verdict.
- **Adaptation :** les 37 presets amont sont déterministes et chacun est exécuté
  depuis une image Sharp fraîche. Un succès exige le payload Prooftag attendu,
  pas simplement un texte quelconque.
- **Affichage :** le score est la fraction de presets exacts ; l'acceptation exige
  au moins un preset exact. Lecture directe, nombre de presets et verdict humain
  sont affichés séparément.
- **Esthétique à l'étape E024 :** CLIP-AES, CLIPScore et HPS étaient retirés de
  la production. E025 annule ce point sans modifier l'autorité QR-Verify.
- **Limite :** QR-Verify reste un test logiciel WASM, pas une probabilité de scan
  téléphone. Le scan humain demeure la vérité terrain.
- **Sécurité :** Sharp est forcé en `0.35.3`, le lockfile est commité et l'audit
  npm ne signale aucune vulnérabilité connue.
- **Protocole :** `docs/e024-qr-verify.md`.

## E025 — scores d'image séparés de QR-Verify — 5 août 2026

- **Décision :** rétablir CLIP-Aesthetic, similarité CLIP brute, CLIPScore et
  HPS v2.1 sans réintroduire d'ancien décodeur ni modifier l'acceptation.
- **Comparabilité papier :** `clip_similarity` conserve l'échelle proche de
  0,30 utilisée dans le tableau DiffQRCoder ; `clip_score` expose séparément la
  formule rescalée `2,5 × max(cosinus, 0)`.
- **Ressources :** scoring sur CPU après la génération, modèles chargés une
  seule fois et cache persistant `/cache` ; la RTX reste réservée à DiffQRCoder.
- **Préférence moderne :** HPS v2.1 est retenu dans le pod actuel. HPSv3 est
  plus récent, mais son Qwen2-VL 7B nécessite un service de scoring GPU séparé.
- **Re-test :** `scripts/e025-quality-retest.py` reprend les dix prompts et les
  deux recettes appariées d'E024.
- **Protocole :** `docs/e025-quality-scoring.md`.

## E026I v1 — inférence du conseiller incomplète — 17 août 2026

- **Archive auditée :** `20260817T103009Z-e026-prompt-parameter-advisor-v1.tar.gz`.
- **Entraînement :** 7 945 observations utilisables, 170 groupes de prompt, 16 recettes et
  validation groupée sans fuite de texte ; 7 402 succès contre 543 échecs QR-Verify.
- **Plan comparatif :** 10 prompts inconnus, top-3 conseillé, baseline Stage 1 et trois seeds,
  soit 120 résultats attendus.
- **Incident :** les 75 sélections SR-MPGD ont échoué avant génération, car aucun Stage 2 SRPG
  strictement apparié n'avait été exécuté plus tôt dans leur campagne.
- **Images réelles :** 45 seulement : 30 baselines Stage 1 et 15 recommandations SRPG. Les
  recommandations réellement générées sont 15/15 QR-Verify, contre 21/30 pour la baseline, mais
  cet échantillon sélectionné ne permet aucune conclusion sur le top-3 complet.
- **Erreur de rapport :** les moyennes ignoraient les 75 valeurs QR absentes et affichaient 100 %.
  Sur le plan complet, la complétion technique du top-3 est 15/90 et la couverture est 15/30
  couples prompt/seed.
- **Correction :** le protocole `e026i-v2-paired-srmpgd` insère un prérequis SRPG dédupliqué avant
  toute recette SR-MPGD, exclut ce prérequis de la comparaison et compte les erreurs techniques
  dans le dénominateur principal. Le nouvel identifiant de plan empêche de reprendre l'état v1.

## E026J — recommandations réellement distinctes — 17 août 2026

- **Archive source :** `20260817T113847Z-e026-prompt-parameter-advisor-v1.tar.gz` : 150/150 essais
  terminés, 90/90 recommandations QR-Verify valides et 21/30 baselines valides.
- **Constat :** les 75 recettes SR-MPGD ont toutes sélectionné l'itération zéro. Les 90 sorties
  conseillées ne représentaient que 45 images uniques ; la robustesse venait de SRPG.
- **Sélection :** déduplication des recettes par signature Stage 2 effective avant génération,
  puis choix de trois profils `robust`, `balanced` et `aesthetic_scannable`.
- **Provenance :** un SR-MPGD à l'itération zéro est désormais étiqueté `srpg`. Le SHA-256 des PNG
  téléchargés fournit une seconde déduplication et corrige le compteur d'images uniques.
- **Déclenchement adaptatif :** E026J exige une tolérance QR-Verify de `0,80` ; un Stage 2 exact
  mais fragile reste éligible au raffinement, alors qu'un SRPG déjà robuste s'arrête à l'état zéro.
- **Reprise :** `RUN_COLLECTION=False` conserve la collecte E026 existante ; seul le nouveau plan
  `e026j-v1-diversified-adaptive-srmpgd` est généré dans `/data/e026j-inference`.
