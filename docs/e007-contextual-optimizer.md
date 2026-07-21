# E007 - optimiseur contextuel scannability-first

## Décision issue d'E006

E006 montre qu'un réglage unique ne généralise pas : `current_steps_60` obtient 46/104 validations
sur quatre contextes, avec 1/26 sur geometric et 23/26 sur engraving. Le nombre de pas n'est pas
monotone : sur botanical, 40/60/80/100 pas donnent 2/11/9/2 validations réussies sur 26.
L'erreur module et la lecture réelle sont pratiquement décorrélées (`r=0,044`). Huit essais sont
en outre invalides à cause d'un second processus GPU.

E007 ne cherche donc plus une constante universelle. Il construit un système adaptatif où la
scannabilité est une contrainte dure et où la qualité intervient seulement après 26/26.

## Références de métriques

Le PDF [`traduction_expliquee_diffqrcoder.pdf`](../output/pdf/traduction_expliquee_diffqrcoder.pdf)
rappelle le protocole DiffQRCoder : SSR, CLIP-aesthetic et CLIP-score sur 100 prompts. Il rapporte
99 % de SSR, 6,8233 en esthétique et 0,2992 en similarité CLIP, tout en déclarant que les
hyperparamètres restent sensibles.

E007 utilise :

- la similarité cosinus CLIP brute pour rester lisible face au tableau DiffQRCoder ;
- le CLIPScore publié `2,5 * max(cosinus, 0)` en mesure secondaire ;
- le prédicteur linéaire CLIP ViT-B/32 de
  [LAION](https://github.com/LAION-AI/aesthetic-predictor) pour CLIP-aesthetic ;
- `openai/clip-vit-base-patch32`, exécuté sur CPU pour ne pas prendre la VRAM de SRPG ;
- TPE multiobjectif d'[Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html),
  adapté aux évaluations coûteuses et aux espaces continus.

CLIPScore et CLIP-aesthetic sont des approximations. Ils ne remplacent ni les tests humains ni la
lecture physique.

## Ordre de décision non négociable

```text
1. statut d'exécution valide
2. payload exact sur 26/26 validations
3. confirmation complète sur tous les contextes holdout
4. CLIP-aesthetic
5. CLIPScore
6. modification du brut et durée
```

Il n'existe aucune somme pondérée permettant à une belle image 25/26 de battre une image 26/26.
Si aucun candidat n'est strict, la sortie est `REJET` et aucune image n'est livrée.

## Espace de recherche

Toutes les dimensions actuellement actionnables sont incluses. Un produit cartésien serait infini
à cause des variables continues ; TPE commence par 24 points exploratoires, puis apprend quelles
zones méritent les essais suivants.

| Bloc | Paramètre | Domaine E007 |
|---|---|---|
| Stage-1 | pas | 8, 12, 16, 20, 24 |
| Stage-1 | strength | 0,65 à 1,00 |
| Stage-1 | CFG | 5 à 15 |
| Stage-1 | ControlNet | 0,8 à 2,0 |
| Stage-2 | pas | 32, 40, 60, 80, 100, 120 |
| Stage-2 | strength | 0,65 à 1,00 |
| Stage-2 | CFG | 5 à 13 |
| Stage-2 | ControlNet | 0,9 à 1,9 |
| SRPG | lambda QR | 300 à 1800, échelle logarithmique |
| SRPG | lambda LPIPS | 0 à 8 |
| SRPG | poids fonctionnels | 1 à 16, logarithmique |
| SRPG | zone centrale | 0,25 ; 1/3 ; 0,45 ; 0,60 |
| SRPG | seuil noir | 0,35 à 0,52 |
| SRPG | marge noir-blanc | 0 à 0,28, seuil blanc plafonné à 0,80 |
| Robustesse | flou différentiable | poids 0 à 2 ; noyau 3, 5 ou 7 |
| Robustesse | réduction puis remontée | poids 0 à 2 ; facteur 0,50 à 0,90 |
| Robustesse | luminosité faible/forte | poids 0 à 2 ; facteurs 0,60–0,90 et 1,10–1,40 |
| Robustesse | contraste réduit | poids 0 à 2 ; facteur 0,50 à 0,90 |
| SRPG | seuil d'arrêt de guidance | erreur module 0 à 0,08 |
| DDIM | cap du delta de bruit | 0,5 à 4, logarithmique |
| DDIM | eta | 0 à 1 |
| Aléatoire | variante seed Stage-2 | 0 à 3 |
| Texte | negative prompt | minimal, standard, structure-safe |

ECC reste fixé à H et la taille à 512 pendant cette campagne. Ce sont des contraintes produit, pas
des hyperparamètres artistiques. Les versions et matrices QR changent avec les payloads et sont
enregistrées comme contexte.

## Nouvelle SRL robuste

La loss QR peut maintenant être calculée sur cinq vues différentiables :

1. image native ;
2. flou 3x3 ;
3. réduction variable puis remontée à 512 ;
4. luminosité faible et forte, avec sévérités variables ;
5. contraste réduit, avec sévérité variable.

Chaque poids est optimisé indépendamment. Le centre des modules, les seuils asymétriques, la zone
calme, les finders, timings, formats et alignements continuent d'être protégés. Le but est de
réduire l'écart observé entre erreur centrale et comportement des vrais décodeurs.

## Plan factoriel

E006 changeait prompt, seed et payload ensemble. E007 les sépare :

- six prompts avec payload et seed identiques ;
- trois seeds supplémentaires avec prompt et payload identiques ;
- trois payloads avec prompt et seed identiques ;
- quatre prompts/payloads/seeds jamais vus pour la confirmation holdout.

Une configuration n'est jamais marquée complète si un holdout manque ou termine en erreur.

La recherche ne promeut pas directement un essai chanceux sur un seul prompt. Les huit meilleurs
essais du screening sont rejoués sur les douze contextes factoriels (96 exécutions de calibration).
Le classement utilise alors, dans l'ordre, la complétude, le strict sur tous les contextes, le pire
taux de lecture, le taux moyen, puis seulement les deux scores CLIP. Les cinq configurations ainsi
promues passent enfin sur les quatre holdouts jamais vus (20 exécutions). Cette double confirmation
évite de confondre une bonne seed avec une configuration réellement robuste.

## Mini-modèle adaptatif

`ContextualParameterAdvisor` est un ensemble ExtraTrees léger. Ses entrées sont :

- projection CLIP du prompt ;
- projection CLIP de l'image Stage-1 ;
- version, taille et densité de la matrice QR ;
- luminance, contraste, entropie et densité de contours du brut ;
- erreur fonctionnelle/data, marges de modules ;
- tous les paramètres candidats.

Il prédit trois sorties séparées : taux de validation, CLIP-aesthetic et CLIPScore. L'incertitude
est l'écart entre les arbres. La validation croisée est groupée par contexte, ce qui empêche de
mesurer uniquement la mémorisation d'un prompt.

Pour une nouvelle demande :

1. une configuration globale génère Stage-1 ;
2. 1024 configurations candidates sont échantillonnées sans diffusion ;
3. le mini-modèle en recommande six ;
4. elles sont réellement générées et validées ;
5. l'exécution s'arrête après trois résultats stricts ou six tentatives ;
6. parmi les stricts seulement, CLIP-aesthetic puis CLIPScore choisissent la livraison.

Le mini-modèle réduit le nombre d'essais ; il ne garantit rien par lui-même. La garantie de
livraison vient exclusivement du validator.

Pour cette phase adaptative, les paramètres Stage-1 et le profil de negative prompt sont verrouillés
sur le brut déjà mesuré. Seuls les paramètres Stage-2/SRPG peuvent varier. C'est indispensable : la
prédiction du conseiller resterait sinon associée à une image Stage-1 différente de celle réellement
générée.

## Garde-fous GPU et erreurs

Avant tout import CUDA, le notebook interroge `nvidia-smi`. La présence d'un processus GPU provoque
un arrêt immédiat. Une OOM durant E007 interrompt également la campagne au lieu de fabriquer un
classement partiel. Après chaque essai, les références temporaires sont libérées, le garbage
collector est exécuté et le cache CUDA est vidé.

Les erreurs, tracebacks, paramètres et durées restent écrits dans `results.jsonl`. Chaque image
possède son JSON de 26 validations et son CSV par timestep.

Le notebook produit aussi `run-summary.csv` (table aplatie de tous les contextes, paramètres et
métriques), `campaign-summary.json`, les agrégats calibration/holdout, les recommandations du
conseiller et quatre graphiques : objectifs de recherche, importance des paramètres, heatmap
configuration-contexte et prédiction contre observation.

## Exécution

Après reconstruction des deux images Docker sur le serveur, lancer depuis Windows :

```powershell
.\scripts\notebook-remote.ps1 -Notebook 04_e007_contextual_optimizer.ipynb
```

Puis `Run > Run All Cells`. La campagne est reprenable via Optuna SQLite et `results.jsonl`.
Les résultats persistent dans `/data/parameter-search/e007-contextual-v1`. Le notebook crée
automatiquement `results/e007-contextual-v1.tar.gz` pour téléchargement.

## Limites avant résultats

- 72 essais de screening ne couvrent pas un espace continu ; les 96 calibrations factorielles et
  20 holdouts renforcent ensuite les configurations les mieux classées.
- un ExtraTrees entraîné avec peu de stricts peut être mal calibré ; son MAE groupé doit être lu.
- CLIP peut favoriser certains styles et comprend mal des détails fins ou la négation.
- OpenCV/ZBar ne couvrent pas tous les téléphones.
- la sortie `DELIVERY.png` doit encore réussir les tests physiques du CSV.
- aucune promesse de taux E007 n'est faite avant la campagne RTX et les scans terrain.
