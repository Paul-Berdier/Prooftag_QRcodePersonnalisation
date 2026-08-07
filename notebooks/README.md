# Notebooks Prooftag QR

## Série actuelle E014–E016

- `11_e014a_qart_blueprint_bakeoff.ipynb` compare vrai QArt, QR binaire, recherche de masque
  exact-payload et blueprint adaptatif dans une Stage 2 appariée.
- `12_e014b_freeqr_latent_fusion.ipynb` teste la fusion latente canal/timestep/force, puis isole
  l'apport d'une loss différentiable.
- `13_e015_aesthetic_backbone_reference.ipynb` compare SD 1.5, SDXL et FLUX comme références
  esthétiques uniquement.
- `14_e016_differentiable_scan_surrogate.ipynb` construit le dataset des vrais décodeurs, entraîne
  le surrogate et vérifie son gradient contre les décodeurs externes.

Le protocole, les limites scientifiques, les artefacts et les portes de décision sont dans
[`../docs/e014-e016-experiment-protocol.md`](../docs/e014-e016-experiment-protocol.md).

Les neuf notebooks n'ont pas le même rôle :

- `01_srpg_step_by_step.ipynb` analyse une archive de benchmark déjà produite. Il ne génère rien.
- `02_generate_live_on_gpu.ipynb` exécute réellement le modèle sur la RTX du serveur et montre
  chaque étape au fur et à mesure.
- `03_srpg_parameter_search.ipynb` compare 17 profils SRPG, reprend une campagne interrompue,
  classe les sorties non réparées et confirme les trois meilleurs profils.
- `04_e007_contextual_optimizer.ipynb` optimise toutes les dimensions utiles sur un plan factoriel,
  mesure CLIP-aesthetic/CLIPScore et entraîne le mini-modèle de recommandation.
- `05_controlnet_model_bakeoff.ipynb` choisit le ControlNet SD1.5 sur une comparaison appariée avant
  de relancer l'optimisation E007.
- `06_nacholmo_generate_live.ipynb` génère en direct avec Nacholmo v2 en text2img, compare trois
  forces ControlNet, puis charge séparément la pipeline img2img nécessaire au SRPG.
- `07_diffqrcoder_official_live.ipynb` exécute le code public DiffQRCoder figé au commit
  `e24ea73`, montre les deux stages pas à pas et compare Stage 1, SRPG et SRPG + SR-MPGD.
- `08_diffqrcoder_vs_qrbtf_four_prompts.ipynb` est le comparatif contrôlé actuel : quatre prompts,
  un QR partagé, DiffQRCoder-paper et reproduction publique QRBTF, puis SR-MPGD sur les deux.
- `09_diffqrcoder_faithful_srmpgd.ipynb` est la référence corrective E012 : DiffQRCoder sur quatre
  prompts, Stage 2 à 40 puis 100 pas, latent final exact et SR-MPGD conforme aux équations 12-14.
- `10_exact_geometry_sd15_sd21_policy.ipynb` est l'expérience E013 : QR aligné sans
  redimensionnement, comparaison DiffQRCoder SD 1.5 / ControlNet QR SD 2.1, SR-MPGD papier et
  amont séparés, Optuna contraint, confirmation multi-prompts et dataset pour le sélecteur
  CatBoost.

## E013 recommandé : géométrie, modèle et recette adaptative

E013 corrige la cause géométrique découverte après E012. Un QR v3 a un cœur de 29 modules. Avec
20 pixels par module, ce cœur mesure exactement 580 pixels. Le notebook utilise donc :

- 744 pixels avec 82 pixels de quiet zone de chaque côté pour SD 1.5 ;
- 768 pixels avec 94 pixels de quiet zone de chaque côté pour SD 2.1.

Il ne redimensionne jamais un QR complet de 740 à 736 pixels. Les contrôles M, Q et H sont décodés
avant le chargement du modèle. Les images intermédiaires de la baseline sont enregistrées à chaque
pas. La recherche Optuna conserve seulement les points d'observation nécessaires afin de ne pas
doubler inutilement le temps de calcul et le volume disque.

Lancer depuis PowerShell après reconstruction de l'image distante :

```powershell
.\scripts\notebook-remote.ps1 -Notebook 10_exact_geometry_sd15_sd21_policy.ipynb
```

La procédure complète, les objectifs, les limites et la stratégie de livraison sont documentés
dans [`../docs/e013-exact-geometry-sd21-policy.md`](../docs/e013-exact-geometry-sd21-policy.md).

## E012 recommandé : SR-MPGD corrigé et auditable

```powershell
.\scripts\notebook-remote.ps1 -Notebook 09_diffqrcoder_faithful_srmpgd.ipynb
```

E012 ne réencode jamais un PNG pour lancer SR-MPGD. Il sauvegarde le latent propre produit par la
Stage 2, puis minimise `SRL + 0,01 × LPIPS` avec `gamma=1000` contre le QR binaire original. La SRL
est celle du dépôt DiffQRCoder figé ; chaque itération est enregistrée et validée. Le nombre
d'itérations n'étant pas publié dans l'article, le notebook teste les états 0 à 20 et s'arrête au
premier 26/26.

La condition Stage 2 est également le QR binaire original valide. Le transformateur QArt
Reed–Solomon décrit dans le papier n'est pas fourni dans le dépôt public : E012 n'invente plus de
proxy visuel susceptible de casser le payload. Il s'agit donc de la baseline du code public avec
SR-MPGD fidèle, et non d'une reproduction complète de l'étape QArt du papier.

Les 16 résultats sont quatre prompts × deux budgets Stage 2 (40/100) × base/SR-MPGD. Toutes les
frames, GIF, latents safetensors, durées, SSR logiciel, MER, CLIP-aesthetic, CLIPScore, manifest,
rapport et grille de validation physique sont exportés. Le protocole complet et ses limites sont
dans [`../docs/e012-faithful-srmpgd.md`](../docs/e012-faithful-srmpgd.md).

## Comparatif E011 historique

Depuis PowerShell :

```powershell
.\scripts\notebook-remote.ps1 -Notebook 08_diffqrcoder_vs_qrbtf_four_prompts.ipynb
```

Le notebook sauvegarde chacun des 40 pas de diffusion, produit un GIF par phase et mesure les
16 sorties attendues : quatre prompts × deux méthodes × sans/avec SR-MPGD. Il rapporte le SSR
logiciel exact, le SSR de l'image originale, la MER, CLIP-aesthetic, CLIPScore et les temps.

La branche QRBTF est explicitement une reproduction publique locale, avec QR Code Monster v2 et
Brightness ControlNet : le backend privé de QRBTF n'est pas publié. De même, le dépôt public
DiffQRCoder ne fournit pas le générateur Reed–Solomon QArt du papier ; le notebook conserve la
matrice exacte dans une cible visuelle documentée. Ces limites et le protocole complet sont dans
[`../docs/e011-diffqrcoder-vs-qrbtf.md`](../docs/e011-diffqrcoder-vs-qrbtf.md).

Attention : les variantes appelées SR-MPGD dans E011 réencodaient l'image et réutilisaient la loss
SRPG pondérée avec un LR 0,1. Elles ne testent pas les équations 12-14 du papier. Leurs résultats
restent utiles comme constat d'échec de cette ancienne correction, pas comme mesure de SR-MPGD.

## Baseline DiffQRCoder officielle (notebook recommandé)

Depuis PowerShell :

```powershell
.\scripts\notebook-remote.ps1 -Notebook 07_diffqrcoder_official_live.ipynb
```

Le notebook 07 est maintenant la référence scientifique. L'image notebook possède sa propre pile
PyTorch 2.6 / Diffusers 0.32.2 et embarque le dépôt officiel au commit complet
`e24ea73ee2e13c7e6e87cb422e8b11784e70ae00`. Il ne passe pas par
`run_srpg_controlnet_img2img` : il appelle directement `_run_stage1` puis `_run_stage2` de
`DiffQRCoderPipeline`.

La configuration initiale reproduit le cadre publié : Cetus-Mix Whalefall, QR Code Monster v2,
QR version 3/M/masque 4, modules 20 px, 40 pas par stage, ControlNet 1,35, SRG 500 et PG 3. Les
variantes SRPG et SRPG + SR-MPGD réutilisent exactement le même état aléatoire du Stage 2. Les
aperçus tous les cinq pas, les validations, CLIP-aesthetic, CLIPScore, le CSV comparatif et le
manifest complet sont conservés dans `/data/notebook-runs`.

Le payload par défaut est le témoin court du projet de recherche. Pour Prooftag, le remplacer par
une URL courte. Si elle ne tient pas dans un QR version 3/M, le notebook s'arrête explicitement :
changer silencieusement de version rendrait les taux incomparables. Détails et limites :
[`../docs/e010-diffqrcoder-official-reference.md`](../docs/e010-diffqrcoder-official-reference.md).

Le notebook 06 reste une expérience Nacholmo documentée, mais n'est plus la trajectoire
recommandée après les sorties visuellement quadrillées observées.

## Génération live Nacholmo corrigée après E008

Depuis PowerShell :

```powershell
.\scripts\notebook-remote.ps1 -Notebook 06_nacholmo_generate_live.ipynb
```

Le Stage-1 suit l'architecture publiée par Nacholmo : `StableDiffusionControlNetPipeline` text2img,
et non un img2img initialisé par le QR. La base par défaut est maintenant
`Nacholmo/Counterfeit-V2.5-vae-swapped`, recommandée par l'auteur dans la discussion Diffusers du
modèle. Le QR binaire complet n'est plus injecté au Stage-1 : le profil
`nacholmo_extremes_25` conserve des centres noirs et blancs arrondis sur un fond gris neutre. C'est
une approximation documentée du conditionnement 25 % noir / 25 % blanc, car le code exact de
prétraitement n'est pas publié.

Le notebook produit avec la même seed trois compromis : `art` (0,40 jusqu'à 55 % des pas),
`balanced` (0,55 jusqu'à 70 %) et `structured` (0,75 jusqu'à 85 %). Il retient explicitement
`balanced`. Ensuite seulement il libère la VRAM et charge une pipeline img2img DDIM avec condition
binaire pour les 100 pas SRPG. Modifier `PAYLOAD`, `PROMPT`, `SEED` ou `RAW_SELECTED_PROFILE` dans
la première cellule permet de tester une demande réelle.

Le notebook n'écrit `06_DELIVERY.png` que si une candidate passe les 26 validations ; sinon il
produit explicitement `06_BEST_OBSERVED_NOT_DELIVERABLE.png`. Le 26/26 isolé d'E008 reste une
preuve de scannabilité sur un contexte, pas une validation des anciens paramètres artistiques.

## Génération réelle depuis le PC Windows

Le navigateur s'ouvre sur le PC, mais le kernel Python, Stable Diffusion, ControlNet et CUDA
s'exécutent dans Kubernetes sur `pcIA`. Il ne faut donc pas lancer le notebook 02 avec le Python
Windows.

Une fois la version notebook construite et déployée sur le serveur, lancer depuis PowerShell dans
le dépôt local :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
.\scripts\notebook-remote.ps1
```

Le tunnel utilise `http://127.0.0.1:18888` par défaut afin de ne pas entrer en conflit avec un
Jupyter Windows déjà lancé sur le port 8888. Si 18888 est occupé, choisir explicitement un autre
port : `./scripts/notebook-remote.ps1 -LocalPort 18889`.

Le tunnel SSH joint directement l'adresse ClusterIP du service Jupyter. Il ne lance plus un
second `kubectl port-forward` sur le serveur : une tentative interrompue ne peut donc plus laisser
le port distant 18888 occupé. En cas d'échec SSH, le diagnostic est conservé dans
`$env:TEMP\prooftag-qr-notebook-ssh.log` sur le PC.

Pour les notebooks de génération directe (02 à 20), cette commande :

1. mémorise l'état de l'API QR et de vLLM ;
2. les arrête pour libérer l'unique GPU ;
3. démarre le pod Jupyter avec la RTX ;
4. crée un tunnel SSH privé ;
5. ouvre directement `02_generate_live_on_gpu.ipynb` sur le PC.

Exception : le notebook 21 E026 est lancé en CPU, conserve l'API à une réplique et lui laisse la
RTX. C'est indispensable pour que le notebook puisse soumettre et suivre la génération des
données sans concurrence GPU avec le kernel Jupyter.

Sans clé SSH, une seconde fenêtre s'ouvre pour le tunnel : saisir le mot de passe `paul@pcIA`
dans cette fenêtre et la laisser ouverte pendant la session Jupyter. Elle sera fermée par la
commande `-Stop`.

Dans Jupyter, utiliser **Run > Run All Cells**. Le notebook fabrique alors, sans archive :

1. le QR de contrôle ;
2. la diffusion artistique brute ;
3. sa validation ;
4. la seconde diffusion SRPG avec aperçu `x0` et carte d'erreur à chacun des 100 pas par défaut ;
5. les courbes de loss et d'erreur de modules ;
6. chaque réparation candidate ;
7. la validation multi-décodeur et multi-dégradation de chaque candidate ;
8. la sélection finale et les exports dans `results/notebook-runs/<date>-<seed>`.

## Recherche des paramètres vers 100 % de lecture

Le notebook 03 exécute le criblage réel sur la RTX. Depuis PowerShell :

```powershell
.\scripts\notebook-remote.ps1 -Notebook 03_srpg_parameter_search.ipynb
```

`Run > Run All Cells` lance 17 essais sur un brut fixe, puis 9 confirmations (trois profils sur
trois autres cas). Chaque essai écrit immédiatement son image, ses validations individuelles,
son CSV par timestep et sa ligne dans `/data/parameter-search/e006-srpg-search-v1/results.jsonl`.
Une interruption ne détruit donc pas la campagne : la relance ignore les clés déjà terminées.

Pour vérifier le pipeline avant la campagne complète, mettre temporairement `SCREEN_LIMIT = 4`
et `RUN_CONFIRMATION = False`. Remettre ensuite `SCREEN_LIMIT = None`, changer
`EXPERIMENT_NAME`, puis lancer la vraie campagne. Le protocole et les portes de décision sont dans
[`../docs/e006-parameter-search.md`](../docs/e006-parameter-search.md).

## Optimisation contextuelle E007

Après E006, le notebook 04 devient la campagne principale :

```powershell
.\scripts\notebook-remote.ps1 -Notebook 04_e007_contextual_optimizer.ipynb
```

Il refuse de démarrer si un autre processus utilise la RTX. Par défaut, il exécute 72 essais TPE,
96 recalculs de calibration factorielle, 20 confirmations holdout, entraîne l'advisor puis simule
jusqu'à six tentatives adaptatives. Cette campagne est nettement plus longue qu'E006 mais chaque
essai est persisté. Voir
[`../docs/e007-contextual-optimizer.md`](../docs/e007-contextual-optimizer.md).

## Conseiller prompt → paramètres E026

E026 remplace le prototype E007 par un contrat adapté au laboratoire actuel : QR-Verify est la
cible de scan, CLIP-Aesthetic, CLIPScore et HPS v2.1 restent des objectifs secondaires, et la
validation sépare entièrement les textes de prompts. Le notebook lance lui-même les campagnes
persistantes via l'API GPU, affiche leur progression, reprend après incident, charge les CSV,
refuse les datasets non identifiables, entraîne le modèle et exporte un top-K avec incertitude :

```powershell
.\scripts\notebook-remote.ps1 -Notebook 21_e026_prompt_parameter_advisor.ipynb
```

Le notebook E026 tourne en CPU pour laisser la RTX à l'API. Les exports sont écrits après chaque
lot sous `/data/e026-week`; `/workspace/imports` reste seulement une source facultative pour les
anciens CSV. Le notebook ne remplace jamais la validation finale QR-Verify. Le protocole est dans
[`../docs/e026-prompt-parameter-advisor.md`](../docs/e026-prompt-parameter-advisor.md).

Pour remplir ce dataset pendant une absence sans laisser Jupyter ouvert, E026W lance un Job
Kubernetes CPU reprenable et réserve la RTX à l'API. Le plan borné contient 300 prompts, 16
recettes, trois seeds et jusqu'à 14 400 essais. Les commandes de démarrage, suivi, reprise et
restauration sont dans
[`../docs/e026-week-unattended.md`](../docs/e026-week-unattended.md).

## Comparaison des ControlNet E008

```powershell
.\scripts\notebook-remote.ps1 -Notebook 05_controlnet_model_bakeoff.ipynb
```

Le notebook compare Dion, QR Code Monster v1/v2 et Nacholmo v2 sur quatre échelles et douze
contextes, soit 192 exécutions. Il mesure séparément le brut ControlNet et la sortie SRPG et crée
le gabarit `physical-validation-template.csv` pour les trois meilleurs modèles. Voir
[`../docs/e008-controlnet-bakeoff.md`](../docs/e008-controlnet-bakeoff.md).

À la fin, arrêter Jupyter et restaurer exactement les nombres de réplicas précédents :

```powershell
.\scripts\notebook-remote.ps1 -Stop
```

Entre deux notebooks GPU, ne pas conserver l'ancien kernel : il garde presque toute la VRAM.
La commande suivante supprime le pod Jupyter et tous ses kernels, attend la libération du GPU,
recrée le pod puis ouvre directement le notebook suivant, sans restaurer temporairement vLLM/API :

```powershell
.\scripts\notebook-remote.ps1 -Reset -Notebook 16_e014b_statistical_freeqr_confirmation.ipynb
```

## Confirmation statistique E014B v2

Le notebook 16 est la suite utile après l'audit E014C. Il ne relance pas une grande recherche de
paramètres : il compare quatre recettes figées sur le cas difficile `p3_detailed`. Chaque recette
est répétée quatre fois, dans un carré latin équilibré de Williams qui neutralise la position et
la recette précédente, avec une pipeline DiffQRCoder fraîche par répétition. Il mesure la
scannabilité sur 39 tests, CLIPScore, CLIP-aesthetic, le temps et la variabilité du contrôle.

```powershell
.\scripts\notebook-remote.ps1 -Reset -Notebook 16_e014b_statistical_freeqr_confirmation.ipynb
```

Lancer ensuite **Run > Run All Cells** une seule fois. Les 16 diffusions prennent typiquement
30 à 45 minutes sur la RTX 4000 Ada. Une répétition interrompue n'est volontairement pas reprise :
il faut créer un nouveau run afin de ne pas modifier l'historique d'exécution de la pipeline. Le
fichier final `DECISION.json` indique si une fusion mérite une confirmation sur les trois autres
prompts. L'ancien notebook 12 reste exploratoire et ne doit pas servir à promouvoir une recette.

## Généralisation multi-contexte E014B v3

Après la promotion de `fusion_all` par E014B v2, le notebook 17 fige cette recette et la compare
à la baseline sur `p1_simple`, `p2_medium` et `p4_complex`. Il exécute quatre blocs appariés par
contexte, avec une pipeline fraîche par bloc et les deux ordres répétés deux fois. Il ne contient
aucune recherche de paramètres.

```powershell
.\scripts\notebook-remote.ps1 -Reset -Notebook 17_e014b_multicontext_generalization.ipynb
```

La campagne effectue 24 diffusions et prend typiquement 50 à 60 minutes sur la RTX 4000 Ada. La
porte corrigée exige la lecture originale 3/3 dans les quatre répétitions de chaque contexte.
Seul le statut `production_candidate` signifie que les douze sorties fusionnées ont franchi
39/39 ; `generalized_not_strict` signifie seulement que le gain se généralise. Les identifiants,
révisions des modèles et le hash du pipeline DiffQRCoder sont conservés dans le manifeste.

## Réparation fonctionnelle tardive E014D

Le notebook 18 repart des meilleurs latents `fusion_all` d'E014B v2/v3. Il ne régénère pas les
quarante pas déjà calculés. Pour `p1`, `p2`, `p3` et `p4`, il compare le contrôle fusionné à trois
forces structurelles prédéfinies : 0,15, 0,30 et 0,45.

Chaque candidat utilise une pipeline fraîche, le même bruit par contexte et les huit derniers
timesteps du planning DDIM à quarante pas. À chaque pas, le canal latent 1 conserve la fusion
alpha 0,15, tandis qu'un masque protège seulement la quiet zone et les motifs fonctionnels. Les
modules de données restent libres et aucune projection de pixels n'est appliquée à la fin.

```powershell
.\scripts\notebook-remote.ps1 -Reset -Notebook 18_e014d_functional_late_rediffusion.ipynb
```

Le classement donne la priorité à l'original 3/3, puis au SSR 39 tests, au pire décodeur, au pire
scénario, à CLIP-aesthetic et à CLIPScore. Une force fixe doit réussir dans les quatre contextes ;
le meilleur réglage différent pour chaque prompt est exporté comme oracle diagnostique, jamais
comme recette générale.

## Ablation mécanistique et temporelle E014E

E014D a confirmé le gain fonctionnel mais a trop dégradé l'image. Le notebook 19 cherche la
contrainte minimale utile. Sa phase A compare, sur `p2_medium` et `p3_detailed`, la rediffusion
DiffQRCoder seule, la fusion globale seule, le masque fonctionnel seul et trois combinaisons
faibles. La combinaison E014D 0,15/0,15 est conservée comme référence forcée.

La phase B promeut la référence et les deux meilleures recettes faibles, puis compare des fenêtres
de 2, 4, 6 et 8 pas sur les quatre contextes. `p1_simple` et `p4_complex` n'interviennent pas dans
la sélection initiale et servent de holdout. Le notebook génère 74 lignes complètes, les GIF et
frames de chaque rediffusion, les audits d'appariement, les erreurs, les métriques par décodeur et
un rapport automatique.

```powershell
.\scripts\notebook-remote.ps1 -Reset -Notebook 19_e014e_mechanism_window_ablation.ipynb
```

Lancer ensuite **Run > Run All Cells**. Une reprise après interruption doit renseigner
`RESUME_RUN_NAME` avec le nom du dossier E014E existant avant de relancer les cellules. E014E
n'entraîne aucun sélecteur : quatre contextes restent insuffisants pour cela.

## Généralisation inconnue et cascade E014F

E014F ne réutilise pas les quatre images d'E014E. Il régénère vingt-quatre sources à partir de
douze nouveaux prompts, deux graines et six payloads. Chaque source utilise une Stage 1, un
blueprint adaptatif exact sélectionné avant la Stage 2, puis une Stage 2 FreeQR complète.

Les quatre recettes préenregistrées sont comparées à 2, 3 et 4 pas, soit 288 réparations. Le
notebook sépare seize contextes de calibration et huit contextes holdout, classe une recette fixe,
mesure l'oracle exhaustif et simule une cascade qui ne livre que les sorties à 39/39.

```powershell
.\scripts\notebook-remote.ps1 -Reset -Notebook 20_e014f_unseen_generalization_cascade.ipynb
```

Lancer **Run > Run All Cells** une seule fois. La campagne peut durer plusieurs heures. Après une
interruption, renseigner `RESUME_RUN_NAME` avec le dossier E014F existant. `CONTEXT_LIMIT` sert
uniquement à vérifier l'installation dans un run distinct, jamais à produire un résultat
revendiqué.

Protocole : [`../docs/e014f-protocol-2026-07-28.md`](../docs/e014f-protocol-2026-07-28.md).

## Première installation ou mise à jour sur le serveur

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git pull
docker build -t prooftag-qr:dev .
docker build -f Dockerfile.notebook -t prooftag-qr-notebook:dev .
docker save prooftag-qr:dev prooftag-qr-notebook:dev | sudo k3s ctr images import -
bash scripts/create-database-secret.sh
kubectl apply -k deploy/k8s
kubectl get deployment/prooftag-qr-notebook -n qr-core
```

Le Deployment notebook reste à zéro réplique tant que la commande PowerShell ne le démarre pas.
Les modèles réutilisent le PVC `prooftag-qr-model-cache` et les résultats persistent dans le PVC
`prooftag-qr-data`, sous `/data/notebook-runs` et `/data/parameter-search`.

Pour éviter qu'un pod réutilise un ancien tag `dev`, préférer le déploiement immuable suivant après
chaque `git pull`. Le script construit une image portant les douze premiers caractères du commit,
vérifie le notebook dans l'image, l'importe dans k3s et met explicitement à jour le Deployment :

```bash
bash scripts/deploy-notebook-image.sh \
  notebooks/20_e014f_unseen_generalization_cascade.ipynb
```

## Analyse d'une ancienne archive sur Windows

Le notebook 01 reste utile pour comparer une campagne déjà rapatriée :

```powershell
python -m pip install -e ".[notebook]"
$env:PROOFTAG_QR_BENCHMARK_ARCHIVE = "$HOME\Downloads\prooftag-benchmarks\20260721T090541Z-0b3c040b.tar.gz"
jupyter lab notebooks\01_srpg_step_by_step.ipynb
```

Il réutilise le dossier déjà extrait ou écrit dans `.prooftag-notebook-cache` à côté de l'archive.
Ce notebook d'analyse n'utilise pas le GPU du serveur.
