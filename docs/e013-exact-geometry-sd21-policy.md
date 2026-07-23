# E013 - géométrie exacte, SD 2.1 et politique de paramètres

## Objectif

E013 cherche une amélioration mesurable de la scannabilité sans transformer l'image en grille QR
visible. L'expérience ne cherche plus un unique réglage « magique ». Elle construit une politique
qui choisit plusieurs recettes selon le prompt, génère des candidats diversifiés, valide le
payload exact et rejette ce qui échoue.

Le notebook exécutable est
[`../notebooks/10_exact_geometry_sd15_sd21_policy.ipynb`](../notebooks/10_exact_geometry_sd15_sd21_policy.ipynb).
Il est généré par
[`../scripts/build_e013_notebook.py`](../scripts/build_e013_notebook.py) pour rendre les
modifications auditables.

## Cause corrigée avant la recherche

E012 construisait un QR de 740 pixels, puis Diffusers le ramenait à 736 pixels. Un QR v3 avec sa
quiet zone contient 37 modules. Après ce redimensionnement, chaque module mesurait environ
19,89 pixels, tandis que la Scanning Robust Loss avançait par pas fixes de 20 pixels. Les centres
observés par la loss dérivaient donc progressivement par rapport à la matrice réelle.

E013 sépare le cœur et la quiet zone :

| Fondation | Canvas | Cœur v3 | Module | Padding par côté | Quiet zone |
|---|---:|---:|---:|---:|---:|
| SD 1.5 | 744 | 29 × 20 = 580 px | 20 px | 82 px | 4,10 modules |
| SD 2.1 | 768 | 29 × 20 = 580 px | 20 px | 94 px | 4,70 modules |

Une ablation à 16 pixels par module est aussi incluse, car QR Monster v2 recommande cette valeur.
Elle conserve elle aussi une grille entière ; seul le padding blanc augmente.

Le cœur est collé sur un canvas blanc, sans aucune interpolation. Le module
[`prooftag_qr.geometry`](../prooftag_qr/geometry.py) mesure ensuite les modules sur ces coordonnées
exactes. Les modifications extérieures au cœur ne contaminent plus la MER.

## Méthodes comparées

### DiffQRCoder / SD 1.5

- fondation : Cetus-Mix Whalefall fp16 ;
- ControlNet : `monster-labs/control_v1p_sd15_qrcode_monster`, sous-dossier `v2` ;
- Stage 1 ControlNet ;
- Stage 2 SRPG initialisée par l'encodage bruité de la Stage 1 ;
- 40 puis 100 pas, et comparaison 744/768 ;
- SR-MPGD papier : latent final exact, gamma 1000, LPIPS VGG 0,01 ;
- SR-MPGD du dépôt : Stage 2 rejouée avec 20 itérations et SGD à 0,1, référence Stage 1.

Les deux SR-MPGD portent des noms distincts parce que le papier et le dépôt public ne décrivent pas
exactement la même correction.

### Stable Diffusion 2.1

- fondation essayée dans l'ordre :
  `stabilityai/stable-diffusion-2-1`, puis
  `sd2-community/stable-diffusion-2-1` ;
- ControlNet natif :
  `DionTimmer/controlnet_qrcode-control_v11p_sd21` ;
- résolution 768 recommandée par sa fiche modèle ;
- text-to-image à 50 et 100 pas ;
- variante de sauvetage img2img à faible strength ;
- SR-MPGD papier sur le latent final.

Cette branche ne s'appelle pas DiffQRCoder SD 2.1 : le portage de SRPG vers la fondation 2.1 n'est
pas validé. E013 compare deux chaînes réellement exécutables et conserve cette différence dans le
manifeste.

## Matrice expérimentale

La baseline contient quatre prompts de complexité croissante et huit recettes. Pour chaque recette :

- même payload ;
- même prompt et seed dans la paire ;
- QR H et masque 4 ;
- image de chaque pas de diffusion ;
- latent initial et latent final lorsque disponibles ;
- temps de chargement, de diffusion et de correction ;
- pic VRAM ;
- 13 scénarios multipliés par OpenCV et ZBar, soit 26 tests ;
- exactitude du payload, pire décodeur et pire scénario ;
- MER globale, fonctionnelle et data ;
- sécurité par rapport aux seuils 0,45/0,65 ;
- CLIP-aesthetic, CLIPScore, netteté, contraste et changement d'image.

La baseline produit 80 lignes : quatre prompts multipliés par douze variantes SD 1.5 et huit
variantes SD 2.1.

## Recherche des paramètres

Deux études Optuna sont conservées dans des bases SQLite séparées. Cela évite de recharger une autre
fondation à chaque essai et permet de reprendre après une interruption.

Les espaces incluent notamment :

- M, Q et H ;
- les huit masques QR ;
- seed ;
- module de 16 ou 20 pixels ;
- nombre de pas ;
- CFG ;
- poids ControlNet ;
- début et fin de la fenêtre ControlNet ;
- profil de prompt négatif ;
- ETA pour DiffQRCoder ;
- SRG et PG pour DiffQRCoder ;
- 744/768 pour SD 1.5 ;
- strength img2img pour SD 2.1 ;
- absence ou présence du SR-MPGD papier ;
- itérations, gamma, poids LPIPS, seuils clair/sombre et fraction centrale de SR-MPGD.

L'optimisation est multiobjectif, mais la lecture est aussi une contrainte. La condition
`1 - pass_rate <= 0` empêche une image non stricte de gagner grâce à CLIP-aesthetic. Les trois
meilleures configurations uniques de chaque famille sont ensuite rejouées sur les quatre prompts.
Cette confirmation est nécessaire : un bon essai isolé ne constitue pas une recette générale.

## Sélecteur contextuel

Toutes les lignes sont exportées dans `policy-dataset.csv` et `policy-dataset.jsonl`. Elles
contiennent prompt, projection CLIP textuelle en 16 dimensions, modèle, QR, paramètres, scores,
durée et résultat de lecture. La projection donne au sélecteur un contexte sémantique au lieu de
mémoriser uniquement l'identifiant d'un prompt.

Un CatBoost de probabilité stricte n'est entraîné que si deux portes sont franchies :

- au moins 100 observations ;
- au moins 12 succès stricts.

La validation est groupée par `prompt_id`, afin qu'un prompt présent dans l'entraînement ne soit
pas aussi utilisé pour mesurer le même modèle. Le notebook enregistre average precision, Brier
score et ROC AUC. Si les données sont insuffisantes, il exporte le dataset mais refuse d'entraîner
un mini-modèle trompeur.

## Ce qui peut réellement approcher 100 %

Une configuration fixe ne peut pas garantir tous les prompts. La garantie opérationnelle vient
de la porte de validation :

```text
contexte utilisateur
  -> ranking de recettes
  -> seeds/configurations diversifiés
  -> génération
  -> validation exacte
  -> succès : livraison
  -> échec et budget restant : candidat suivant
  -> budget épuisé : rejet
```

Sous une hypothèse approximative d'indépendance, la probabilité d'au moins un succès est
`1 - (1 - p)^N`. E013 calcule le nombre de tentatives associé à 99,9 %, puis le plafonne à 12.
Cette formule ne remplace pas la confirmation : des échecs corrélés entre prompts rendent
l'hypothèse optimiste.

## Reprise et artefacts

Les lignes JSONL sont écrites immédiatement. Les études Optuna utilisent `load_if_exists=True`.
Après un arrêt, renseigner `RESUME_RUN_NAME` dans la section 1 et relancer les cellules. Les
artefacts principaux sont :

- `geometry-audit.json` ;
- `results.jsonl` et `search-results.jsonl` ;
- `baseline-aggregates.csv` ;
- `baseline-comparison.png` et `baseline-metrics.png` ;
- `search-objectives.png` et `parameter-importance.json` ;
- bases `optuna-*.sqlite3` ;
- `policy-dataset.*` et `policy-status.json` ;
- modèle `prooftag-parameter-selector.cbm` si les portes sont franchies ;
- `delivery-budget.json` ;
- `physical-validation.csv` ;
- `manifest.json`, `run-report.md` et archive `.tar.gz`.

## Exécution sur le serveur

Le changement de dépendances impose une reconstruction de l'image notebook :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git pull
docker build -f Dockerfile.notebook -t prooftag-qr-notebook:dev .
docker save prooftag-qr:dev prooftag-qr-notebook:dev |
  sudo k3s ctr images import -
kubectl rollout restart -n qr-core deployment/prooftag-qr-notebook
```

Depuis PowerShell :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
git pull
.\scripts\notebook-remote.ps1 -Notebook 10_exact_geometry_sd15_sd21_policy.ipynb
```

Le lancement du notebook met en pause la charge GPU existante selon le script serveur. Utiliser
`.\scripts\notebook-remote.ps1 -Stop` après l'expérience pour restaurer la charge précédente.

## Limites connues

- QRBTF propriétaire reste hors périmètre : son backend et ses latents ne sont pas publics.
- La transformation QArt Reed-Solomon du papier DiffQRCoder n'est pas publiée ; aucun proxy
  susceptible de casser le payload n'est réintroduit.
- La validation logicielle ne remplace pas les téléphones, distances, angles, éclairages et
  impressions.
- Les scores CLIP évaluent la cohérence et une esthétique apprise ; ils ne représentent pas à eux
  seuls la qualité Prooftag.
- Un sélecteur appris réduit le coût moyen. Il ne remplace jamais la validation de la sortie.

## Incident E013-01 - gradient SR-MPGD non fini

Le premier lancement a rencontré un gradient non fini à l'itération zéro de l'essai Optuna 8 :
SD 1.5, canvas 744, modules 16, 70 pas, H, masque 0 et SR-MPGD papier. La diffusion de 70 pas
était terminée ; seule la correction latente était numériquement invalide.

La première implémentation levait `FloatingPointError`, ce qui arrêtait toute l'étude après presque
trois minutes de diffusion. Le comportement corrigé est :

1. conserver et valider l'état zéro, qui reste une image finie ;
2. arrêter uniquement cette correction avec
   `stop_reason=non_finite_gradient_at_iteration_0` ;
3. ne jamais remplacer silencieusement les NaN par un gradient arbitraire ;
4. continuer l'étude ;
5. viser 32 essais `COMPLETE`, sans compter les anciens essais `FAIL` dans ce quota ;
6. libérer le modèle dans un bloc `finally` si une autre exception interrompt l'étude.

La base SQLite et les huit essais complets antérieurs sont réutilisables. Il ne faut ni supprimer
le dossier d'expérience ni recommencer la baseline.
