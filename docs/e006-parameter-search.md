# E006 — recherche et validation des paramètres SRPG

## Point de départ

Le 21 juillet 2026, le même exemple est devenu lisible sur un téléphone après passage de
`SRPG_STEPS=40` à `SRPG_STEPS=100`. C'est un signal expérimental utile, mais pas encore un taux de
réussite : il ne porte que sur une image, une graine, un payload et un téléphone.

E006 transforme cette observation en campagne reproductible. La priorité de classement est la
lecture exacte du payload, puis le pire cas, l'erreur module, la conservation artistique et enfin
le temps. Une image réparée de façon déterministe ne compte pas comme réussite du modèle SRPG.

## Audit de la référence DiffQRCoder

Sources primaires :

- [article DiffQRCoder v3](https://arxiv.org/abs/2409.06355) ;
- [dépôt officiel](https://github.com/jwliao1209/DiffQRCoder) ;
- [script officiel](https://github.com/jwliao1209/DiffQRCoder/blob/main/run_diffqrcoder.py).

Le protocole publié utilise deux diffusions de 40 pas, Stable Diffusion 1.5 via Cetus-Mix
Whalefall, QR Code Monster v2, une échelle ControlNet de 1,35 et `easynegative`. Les QR de
l'évaluation sont fixes : version 3, correction M, masque 4, modules de 20 pixels et marge de
80 pixels. Prooftag utilise au contraire une correction H, un payload et donc une version QR
variables, dans une image de 512 pixels. Les chiffres publiés ne sont donc pas directement
transférables.

L'ablation de l'article teste `lambda1` QR à 400, 500, 600 et 1000. Sans SR-MPGD, les SSR publiés
sont respectivement 86 %, 88 %, 94 % et 93 %. Avec SR-MPGD, ils deviennent 98 %, 100 %, 99 % et
99 %. Pour `lambda1=500`, `lambda2` perceptuel à 2, 3, 5 et 10 donne 90 %, 89 %, 89 % et 88 % sans
SR-MPGD, puis 98 %, 99 %, 100 % et 97 % avec SR-MPGD. Ces taux sont des résultats de dataset avec
leur outil de vérification, pas une garantie universelle de lecture physique.

Deux divergences de la référence sont consignées :

1. l'algorithme de l'article décrit un démarrage Stage-2 à partir de Stage-1 bruité et un guide
   QArt, alors que la version actuelle du pipeline officiel initialise des latents aléatoires et
   n'applique pas clairement QArt dans cette passe ;
2. le texte nomme LPIPS, mais le code officiel de guidage utilise aussi une distance de features
   VGG. Prooftag utilise LPIPS/AlexNet.

Ces divergences interdisent de présenter notre adaptation comme une reproduction bit à bit.

## Paramètres réellement testés

Le plan de criblage comporte 17 essais causaux. L'image brute et la graine Stage-2 restent fixes :

| Famille | Valeurs |
|---|---|
| Nombre de pas | 40, 60, 80, 100 |
| Profil Prooftag actuel | CFG 12, fonctionnels 4, seuils 0,50 / 0,50 |
| Profil officiel adapté | CFG 7,5, fonctionnels 1, seuils 0,45 / 0,65 |
| Poids QR | 400, 500, 600, 1000 |
| Poids perceptuel | 0, 2, 3, 5 |
| Échelle ControlNet | 1,05, 1,35, 1,50 |
| Force Stage-2 | 0,85, 1,00 |
| Cap du delta de bruit | 1, 2, 4 |

Il ne s'agit pas d'un produit cartésien. Une recherche exhaustive coûterait cher et mélangerait
les causes. Le notebook isole d'abord l'effet du nombre de pas observé, reproduit ensuite les
plages publiées, puis ne confirme que les trois meilleurs profils.

## Métriques et artefacts

Chaque essai écrit immédiatement dans `/data/parameter-search/<EXPERIMENT_NAME>` :

- l'image SRPG et les paramètres exacts ;
- le résultat individuel de chaque décodeur et scénario dans `*.validations.json` ;
- les métriques de chaque timestep dans `*.steps.csv` ;
- le taux exact global et sur l'original, le nombre de validations, l'erreur module ;
- MAE et proportion de pixels changés par rapport au brut ;
- durée, pic VRAM, taux de clipping, gradient et delta de bruit finaux ;
- acceptation ou rejet par la porte interne et traceback complet en cas d'erreur ;
- un journal reprenable `results.jsonl`, un tableau de classement et une planche des meilleurs.

Les trois premiers profils sont rejoués sur trois autres styles, payloads et graines. Le
classement de confirmation utilise d'abord le **pire taux de lecture**, pas la moyenne. Le fichier
`phone-validation.csv` impose ensuite dix essais par téléphone et condition : écran frontal,
angle de 30 degrés, faible lumière et impression de 5 cm.

## Portes de promotion

Un profil ne devient candidat de livraison que s'il satisfait toutes les conditions :

1. 100 % des validations automatiques exactes sur tous les cas de confirmation ;
2. aucune substitution de payload ;
3. résultat physique rempli et accepté sur plusieurs téléphones ;
4. qualité artistique examinée sur la sortie SRPG non réparée ;
5. temps et VRAM compatibles avec la RTX 4000 Ada 20 Go ;
6. seconde campagne indépendante avec de nouveaux payloads, prompts et graines.

La configuration Kubernetes reste donc à 40 pas et SRPG désactivé tant que la campagne E006 n'a
pas produit ces preuves. Le notebook pédagogique 02 adopte 100 pas pour reproduire l'observation,
mais cela ne constitue pas une modification de production.

## Exécution

Depuis PowerShell sur le PC :

```powershell
git pull
.\scripts\notebook-remote.ps1 -Notebook 03_srpg_parameter_search.ipynb
```

Dans Jupyter, exécuter `Run > Run All Cells`. Une première vérification peut utiliser
`SCREEN_LIMIT = 4` et `RUN_CONFIRMATION = False`. Pour la campagne probante, conserver
`SCREEN_LIMIT = None`, `TOP_COUNT = 3` et `RUN_CONFIRMATION = True`. Une relance reprend les essais
déjà écrits ; après toute modification de code ou de protocole, changer `EXPERIMENT_NAME`.

À la fin :

```powershell
.\scripts\notebook-remote.ps1 -Stop
```

## Erreurs et risques déjà identifiés

- conclure à partir d'un seul scan téléphone ;
- confondre la finale réparée avec la sortie du modèle ;
- modifier plusieurs paramètres et perdre la causalité ;
- comparer des essais avec des images brutes ou graines différentes ;
- reprendre un ancien `results.jsonl` après un changement de code ;
- optimiser seulement la lecture logicielle et oublier écran, angle, lumière et impression ;
- revendiquer les 99–100 % de l'article malgré un dataset, un QR et une implémentation différents.

Tous ces points sont désormais soit bloqués par le protocole, soit rendus visibles dans les
artefacts. Aucun taux E006 n'est inscrit avant l'exécution réelle sur la RTX.
