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
