# E046 large — dataset pour le futur conseiller de paramètres

## Objet scientifique

`e046-large-advisor-dataset-v1` ne sert pas à entraîner un générateur d'images. Il produit un
jeu de données supervisé pour un futur conseiller qui devra estimer, à partir d'un prompt, d'un
payload et d'une recette complète, la robustesse QR et la qualité visuelle attendues. Un mauvais
QR n'est donc pas jeté : les échecs et les compromis difficiles sont des observations nécessaires
pour apprendre la frontière entre lecture et esthétique.

Cette expérience est nouvelle et indépendante. Elle écrit uniquement sous :

```text
/data/e046-large-advisor-dataset-v1
```

Les racines historiques `/data/e045-foundation-v1` et
`/data/e046-controlled-best-generator-v1`, les plans E045/E046 antérieurs, E031 et la quarantaine
E016 ne sont ni migrés, ni réinterprétés, ni supprimés. Le moteur et l'ordonnanceur refusent aussi
explicitement d'utiliser ces deux racines ou l'un de leurs descendants comme sortie.

## Dimensions exactes de la campagne

Le catalogue canonique contient exactement 256 prompts : 16 familles visuelles, 16 prompts par
famille. Chaque prompt possède un texte, des tags structurels et un payload court unique de la
forme `https://ptag.io/d2/NNNN`. Chaque famille suit la même grille de 16 lignes structurelles
(fréquence spatiale, symétrie, luminosité, contraste, colorimétrie, espace négatif, densité de
texture, composition) ; les textes sont assemblés de façon déterministe à partir de 256 sujets
distincts et des descripteurs de leur ligne. Le prompt brutaliste historique E044 est conservé mot
pour mot comme ordinal 1 (payload `d2/0001`, masque 4 avec la configuration ancre) et occupe la
ligne 5, la seule dont les tags correspondent à sa formulation ; il est présent dans les trois
profils.

Les douze configurations parentes comprennent l'ancre historique E044 et onze points maximin
d'un hypercube latin déterministe. Elles couvrent 13 facteurs :

1. ECC ;
2. masque QR ;
3. pas Stage 1 ;
4. guidance Stage 1 ;
5. force ControlNet Stage 1 ;
6. début du contrôle ;
7. fin du contrôle ;
8. initialisation Stage 2 ;
9. force de redémarrage Stage 2 ;
10. pas Stage 2 ;
11. force ControlNet Stage 2 ;
12. poids QR Stage 2 ;
13. poids perceptuel Stage 2.

Le masque réel d'un parent est `(mask_slot + mask_rotation) mod 8`, où la rotation par prompt
vaut `(design_row - 5 + 2 × (family_index - 1)) mod 8` : l'ancre brutaliste (rotation 0) garde
son masque 4 historique, chaque masque reçoit exactement 384 parents en full, 24 par slot en
pilot, et le masque n'est pas une fonction pure de la ligne structurale à configuration fixée.
Le seed est dérivé de manière reproductible du prompt et de l'index de configuration. Le JSON du
catalogue et celui du DOE sont versionnés. Leurs SHA-256, le commit source, le digest de l'image
runtime et le profil font partie du plan scientifique immuable.

| Profil | Prompts | Configurations par prompt | Parents indépendants | Parents SR-MPGD max. | Trajectoires SR-MPGD max. | Checkpoints estimés |
|---|---:|---:|---:|---:|---:|---:|
| `smoke` | 4 | 2 | 8 | 4 | 4 | 32 |
| `pilot` | 32 | 6 | 192 | 40 | 80 | 640 |
| `full` | 256 | 12 | 3 072 | 320 | 640 | 5 120 |

Le profil par défaut est toujours `smoke`. Le profil `full` exige un choix explicite et un jeton
de confirmation ; aucun enchaînement après le smoke ne peut le lancer implicitement.

## Observation principale et scoring

Pour chacun des 3 072 parents complets, la génération conserve Stage 1, Stage 2 brut, le latent
Stage 2, la recette, le seed et la provenance. Le scoring coûteux porte cependant sur une seule
observation principale : `stage2_raw`.

Les cibles restent séparées :

- `wechat_exact_presets`, sur `qr-verify@0.2.0` / WeChat WASM, 37 presets et trois répétitions ;
- `wechat_exact_rate`, `wechat_original_exact`, stabilité et résultats directs répétés ;
- CLIPScore, CLIP-Aesthetic et HPSv2.1 ;
- MER et erreurs structurelles par module ;
- clipping, saturation et garde visuelle ;
- score multiobjectif additionnel, jamais utilisé pour masquer les cibles élémentaires.

Le cache QR-Verify est adressé par contenu. Une reprise ne doit donc ni régénérer un raster déjà
promu, ni rescanner aveuglément une image déjà connue.

## Validité et sélection

Stage 1 est une provenance, jamais une livraison. Les variantes de quiet zone ne sont pas
produites en masse et ne sont jamais éligibles. Le gate final automatique exige simultanément :

```text
raw == true
visual_guard_pass == true
wechat_original_exact == true
wechat_exact_presets >= 34
```

Un QR invalide ne peut pas gagner grâce à son esthétique. Une fois le gate franchi, le
classement multiobjectif descriptif utilise 40 % WeChat, 25 % CLIPScore, 20 % HPSv2.1 et 15 %
CLIP-Aesthetic. Les cinq intervalles QR `0–5`, `6–15`, `16–25`, `26–33` et `34–37` restent tous
dans le dataset, y compris les beaux QR illisibles et les QR robustes visuellement faibles.

## Phase B : sous-échantillonnage SR-MPGD

SR-MPGD n'est pas exécuté sur tous les parents. Après le scoring de la Phase A, la sélection
automatique prend un parent principal par prompt et, selon le profil, des parents secondaires
informatifs. Les strates couvrent le meilleur valide, la frontière 34/37, les conflits
QR/esthétique, l'instabilité et les zones clairsemées du DOE.

Les recettes de départ sont `g250_r100_i04`, `g500_r200_i08`, `g1000_r150_i08` et
`g500_r150_i08`. Tous les checkpoints sont conservés, mais ils sont corrélés. Les champs
`trajectory_id`, `generation_group_id`, `prompt_group_id`, `parent_id`, `iteration` et
`observation_independence_class` interdisent de les compter comme des expériences parentes
indépendantes. Le checkpoint `i0` conserve obligatoirement le raster Stage 2 parent pixel pour
pixel et son latent exact ; le décodage VAE flottant sert à l'objectif différentiable, pas à
remplacer silencieusement ce témoin no-op.

Le message terminal exact `local upstream SRL port diverged from the pinned official class` est
un échec scientifique local à la branche :

```text
classification = scientific_fidelity_mismatch
retryable       = false
usable          = false
```

Il reste visible, n'assouplit jamais la tolérance et n'arrête pas les autres tâches. Une erreur
technique différente reste un échec du smoke et doit être diagnostiquée avant reprise.

Une branche arrêtée pour cette divergence terminale n'est jamais relancée par une reprise
normale. Après correction et vérification de la cause, l'opérateur peut autoriser **une reprise
manuelle explicite** des seules branches encore non promues :

```bash
PROOFTAG_E046_LARGE_OVERRIDE_SRL_TERMINAL=1 \
  bash scripts/run-e046-large-dataset.sh resume
```

La valeur par défaut est `0` et le runner refuse toute autre valeur que `0` ou `1`. Cet override
ne relâche ni la tolérance SRL, ni les portes scientifiques, ne supprime aucun artefact et ne
remplace jamais une sortie déjà promue. Il ne doit pas servir à réessayer une panne technique
ordinaire ; `PROOFTAG_E046_LARGE_RETRY_FAILED=1` reste réservé à un incident Kubernetes transitoire
diagnostiqué.

## Fichiers advisor-ready

Après agrégation, le dossier `<plan-id>/dataset` contient :

```text
parent-observations.jsonl / .csv
srmpgd-observations.jsonl / .csv
advisor-observations.jsonl / .csv
data-summary.json
parameter-coverage.json
prompt-coverage.json
score-distributions.json
split-contract.json
splits.json
advisor-training-contract.json
best-valid-by-prompt.json
```

`splits.json` fixe des plis déterministes au niveau du prompt : un GroupKFold à 5 plis stratifié
par famille (`(design_row - 1) mod 5`) et un leave-one-family-out à 16 plis. Chaque ligne, y
compris chaque checkpoint SR-MPGD, hérite du pli de son `prompt_id` ; une validation refuse tout
groupe (prompt, payload, groupe de génération, trajectoire) réparti sur plusieurs plis.
`advisor-training-contract.json` décrit les colonnes de features et de cibles, les têtes de
prédiction séparées attendues pour E047 et les crochets d'active learning ; il déclare
explicitement qu'aucun conseiller n'est entraîné.

Le contrat de split interdit un tirage aléatoire ligne par ligne. Les évaluations admises sont
un GroupKFold par `prompt_id` et un leave-one-family-out. Un `trajectory_id`, un payload et un
groupe de génération doivent rester entièrement du même côté du split.

Le notebook `notebooks/50_e046_large_advisor_dataset.ipynb` est une vue CPU et en lecture seule.
Il sait lire un plan partiel ou terminé, montre la progression, les couvertures, distributions,
corrélations, fronts de Pareto, taux par famille/masque/ECC, corrélations descriptives des
paramètres, hard negatives et planches des meilleurs/pire cas. Une galerie distincte montre aussi
les derniers Stage 2 bruts atomiquement promus qui attendent encore leur scoring ; ils sont
explicitement marqués non scorés. La force de redémarrage n'est analysée que lorsqu'elle est
effective (`paper_stage1_noise`), et l'initialisation Stage 2 reste un facteur catégoriel séparé.
Le notebook ne lance aucune génération et ne choisit aucun résultat manuellement.

## Budget initial et limites

Les hypothèses de plan donnent environ 278,8 heures séquentielles (217,6 h GPU, 61,2 h de scoring
CPU) et 26,0 Gio pour le profil complet : 180 s de génération et 30 s de scoring par parent, 360 s
par trajectoire SR-MPGD et 25 s par checkpoint scoré. Le pilot est estimé à 23,6 h et 2,3 Gio, le
smoke à 1,1 h. Ces hypothèses sont volontairement conservatrices : E013 a mesuré 160 à 216 s par
paire Stage 1 + Stage 2 à 768 px, et aucun temps SR-MPGD ou scoring par image n'est archivé dans
le dépôt. Ce sont des estimations initiales, pas une promesse. Dès que des tâches
sont terminées, `status` calcule une ETA observée qui doit remplacer ces chiffres.

Une seule RTX 4000 SFF Ada de 20 Gio est disponible. Les Jobs GPU restent séquentiels ; API,
notebook et vLLM restent à zéro pendant un Job, conformément au contrat opérateur. Lorsqu'il est
ouvert entre deux séquences, le notebook E046 est configuré en `offline-cpu` et ne réserve pas la
RTX. Toute remise en route de l'API ou restauration après déploiement est refusée si un autre pod
GPU est détecté dans le cluster.

## Procédure opérateur sûre

Le déploiement E046 construit et épingle à la fois l'image API/Jobs et l'image notebook contenant
`50_e046_large_advisor_dataset.ipynb`. Il vérifie le commit embarqué indépendamment des variables
Kubernetes :

```bash
bash scripts/deploy-e046-large-dataset.sh check
bash scripts/deploy-e046-large-dataset.sh deploy
```

Sur le cluster K3s local, `status.containerStatuses.imageID` peut être rendu soit sous la forme
OCI `dépôt@sha256:…`, soit comme le seul digest de configuration `sha256:…` après un import direct.
Le second format n'est accepté que s'il est strictement égal au digest enregistré au build. Les
Jobs emploient alors le tag propre au commit déjà importé avec `imagePullPolicy: Never`, et le
processus GPU compare en plus `/app/prooftag-build-commit.txt` au commit gelé dans le plan.

Après déploiement, créer seulement le plan smoke :

```bash
bash scripts/run-e046-large-dataset.sh plan
```

Vérifier les nombres, hashes, identité runtime et estimations affichés. Démarrer ensuite le smoke
explicitement :

```bash
bash scripts/run-e046-large-dataset.sh run
```

Suivi sans relance :

```bash
bash scripts/run-e046-large-dataset.sh status
bash scripts/run-e046-large-dataset.sh progress
bash scripts/run-e046-large-dataset.sh logs
```

Pour inspecter les images pendant une longue campagne, demander d'abord la pause après le Job
courant. Une fois le runner revenu au prompt, ouvrir la vue CPU depuis le PC :

```powershell
.\scripts\notebook-remote.ps1 -Reset -Notebook 50_e046_large_advisor_dataset.ipynb
```

Fermer ensuite le notebook avec `-Stop`, puis reprendre la campagne. Une commande `resume` remet
automatiquement le notebook à zéro avant le prochain Job GPU.

Arrêt après le Job courant et reprise :

```bash
bash scripts/run-e046-large-dataset.sh pause-after-current
bash scripts/run-e046-large-dataset.sh resume
```

Arrêt immédiat du Job courant, uniquement si nécessaire :

```bash
bash scripts/run-e046-large-dataset.sh cancel-current
```

Cette commande supprime le Job Kubernetes courant mais ne supprime aucune donnée sous `/data`.
Une interruption SSH ou Ctrl+C demande une pause après le Job courant ; les promotions atomiques
et marqueurs `GENERATION_COMPLETE.json` / `SCORING_COMPLETE.json` rendent la reprise idempotente.

## Critères automatiques du smoke

Le smoke est valide seulement si `verify` confirme tous les points suivants :

- manifeste complet, fichiers présents et SHA-256 conformes ;
- exactement 8 observations parentes Stage 2 raw ;
- CLIPScore, CLIP-Aesthetic et HPSv2.1 finis pour les 8 parents ;
- exemples négatifs non filtrés ;
- sélection Phase B écrite, au moins une trajectoire SR-MPGD réussie et complète, puis chaque
  autre trajectoire soit entièrement scorée, soit exposée comme échec scientifique terminal ;
- aucune erreur technique non résolue ;
- contrat de groupes présent et anciennes racines E045/E046 intactes.

Un PASS smoke valide l'infrastructure, pas la généralisation scientifique, pas un taux de scan
téléphone et pas le futur conseiller. Le profil pilot ne doit être planifié qu'après ce PASS ; le
profil complet ne doit être envisagé qu'après validation du pilot et de son budget observé.
