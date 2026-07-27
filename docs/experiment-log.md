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
