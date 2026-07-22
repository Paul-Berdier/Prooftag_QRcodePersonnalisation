# E012 - DiffQRCoder et SR-MPGD conforme au papier

## Décision et objectif

E012 remplace seulement l'expérience SR-MPGD d'E011. Il ne remplace pas les résultats visuels de
base et n'ajoute pas une troisième méthode. La question mesurée est : avec le même QR, le même
prompt et la même seed, quel est l'effet causal de 40 contre 100 pas de Stage 2, puis du véritable
SR-MPGD sur le latent final ?

Le notebook est
[`../notebooks/09_diffqrcoder_faithful_srmpgd.ipynb`](../notebooks/09_diffqrcoder_faithful_srmpgd.ipynb).
Il est généré par [`../scripts/build_e012_notebook.py`](../scripts/build_e012_notebook.py) afin que
le JSON Jupyter ne soit jamais modifié manuellement.

## Correction scientifique

Le papier définit la correction post-diffusion par :

```text
L(x, y, x0) = LSR(x, y) + 0,01 * LLPIPS(x, x0)
z(i) = z(i-1) - 1000 * gradient_z L(D(z(i-1)), y, x0)
```

- `z(0)` est le latent propre exact produit par la Stage 2 ; aucun PNG n'est réencodé ;
- `y` est le QR binaire original, pas la condition QArt ;
- `x0` est l'image Stage 2 détachée, utilisée comme référence perceptuelle ;
- `LSR` est l'implémentation publique `ScanningRobustLoss` de DiffQRCoder, inchangée ;
- `LLPIPS` est le réseau LPIPS appris avec backbone VGG, et non une MSE de features ;
- les paramètres de diffusion, ControlNet, VAE et LPIPS sont gelés ; seul le latent est modifié ;
- aucune normalisation du gradient, aucun clipping et aucun poids SRPG 500/3 ne sont ajoutés.

L'article fixe `gamma=1000` et `lambda=0,01`, mais ne publie pas le nombre d'itérations. E012
enregistre les états 0 à 20, les valide tous et s'arrête dès la première réussite stricte 26/26.
Sans réussite stricte, il conserve le meilleur état suivant l'ordre : réussite totale, taux de
lecture, pire décodeur, pire scénario, LPIPS, puis MER. Il ne suppose jamais que la dernière
itération est la meilleure.

## Pipeline observé

```text
QR v3/M/masque 4 + prompt + seed
              |
              v
Stage 1 DiffQRCoder - 40 pas DDIM - image de référence
              |
              +--> QR binaire original valide (condition Stage 2 publique)
              |
              v
Stage 2 SRPG - 40 pas OU 100 pas - latent propre exact
              |                         |
              |                         +--> base évaluée
              v
SR-MPGD : SRL(QR original) + 0,01 LPIPS(image Stage 2)
              |
              v
états 0..20 + validations + sélection scannabilité d'abord
```

Chaque état affiché pendant DDIM est explicitement nommé `decoded latent state`. Ce n'est pas une
estimation `x0|t`. Cette distinction corrige aussi une ambiguïté des notebooks antérieurs.

## Modèles et paramètres figés

| Élément | Valeur |
|---|---|
| Dépôt DiffQRCoder | commit `e24ea73ee2e13c7e6e87cb422e8b11784e70ae00` |
| Fondation | Cetus-Mix Whalefall fp16, architecture Stable Diffusion 1.5 |
| ControlNet | `monster-labs/control_v1p_sd15_qrcode_monster`, sous-dossier `v2` |
| Scheduler | DDIM |
| QR | version 3, correction M, masque 4, module 20 px, quiet zone 4 |
| Condition Stage 2 | QR binaire original valide ; aucune imitation QArt |
| Stage 1 | 40 pas, CFG 7,5, ControlNet 1,35 |
| Stage 2 | profils appariés 40 et 100 pas, SRG 500, PG 3 |
| SR-MPGD | 0 à 20 mises à jour, gamma 1000, LPIPS 0,01, VGG |

Les quatre prompts couvrent une composition simple, moyenne, détaillée et complexe. Une seed fixe
est associée à chaque prompt. La campagne crée exactement 16 résultats : quatre prompts, deux
budgets Stage 2 et deux variantes (base/SR-MPGD).

## Mesures et preuves persistées

Pour chaque candidat : payload exact, total validé, SSR original, pire taux par décodeur et par
scénario, MER, erreurs fonctionnelles/données, métriques d'image, CLIP-aesthetic, CLIPScore, temps
par phase, itération sélectionnée et motif d'arrêt. Les artefacts incluent :

- les 40 images Stage 1 et chaque image Stage 2 (40 ou 100), avec GIF et trace temporelle ;
- le latent Stage 2 exact et le latent SR-MPGD sélectionné en `safetensors` ;
- l'image et la validation de chaque état SR-MPGD ;
- `results.jsonl`, `comparison.csv`, `comparison-4x4.png` et `metrics-overview.png` ;
- les hash SHA-256 des sources amont et des latents ;
- `manifest.json`, `upstream-patches.json`, `run-report.md` et le notebook exécuté ;
- `physical-validation.csv`, dix essais par téléphone et support, laissé vide tant que non testé.

Le QR binaire témoin doit être décodé exactement dans le scénario `original` par OpenCV et ZBar.
Ses autres scénarios servent de calibration et sont tous enregistrés, mais un échec simulé ne
bloque pas le lancement : exiger 26/26 à ce point confondrait intégrité du QR et robustesse. La
porte 26/26 reste inchangée pour déclarer une image artistique `DELIVERABLE`.

Une sortie `DELIVERABLE` signifie seulement qu'elle a passé la porte logicielle complète. Elle ne
devient pas une preuve de SSR physique avant les essais téléphone/écran/impression.

## Limite QArt non masquée

Le papier utilise QArt pour changer les bits sélectionnables tout en conservant le message et les
codes Reed-Solomon. Le dépôt DiffQRCoder ne publie pas ce générateur. Le premier brouillon d'E012
avait tenté un proxy visuel matriciel ; l'exécution sur `p1_simple` a démontré qu'il ne préservait
pas le payload. Ce proxy n'était donc pas un QArt valide et a été supprimé.

E012 utilise maintenant le QR binaire original comme condition Stage 2, ce qui correspond au
chemin reproductible du code public. Le fichier `stage2-binary-qr-condition.png` est exporté pour
lever toute ambiguïté. Cette décision permet une baseline honnête du code public et de SR-MPGD,
mais pas une reproduction complète de la pipeline du papier. Une future expérience QArt devra
réencoder réellement les degrés de liberté Reed-Solomon et franchir la porte payload avant toute
diffusion ; aucun proxy graphique ne sera accepté.

## Pourquoi ne pas changer immédiatement pour SDXL ou FLUX

QR Monster v2 et la classe publique DiffQRCoder sont construits pour Stable Diffusion 1.5. Un
checkpoint SDXL ou FLUX n'est pas interchangeable : UNet/transformer, encodeurs de texte,
conditionnement et pipeline ControlNet diffèrent. Le changement demanderait de porter le calcul
`x0|t`, SRPG et la gestion mémoire, puis de recommencer toute la baseline.

SDXL reste une expérience légitime après E012, avec un ControlNet QR SDXL tel que QR Monster SDXL
ou Nacholmo QR Pattern SDXL et la pipeline Diffusers dédiée. Ces cartes indiquent elles-mêmes que
toutes les sorties ne sont pas lisibles. Sur 20 Gio, l'inférence avec offload est envisageable,
mais SRPG demande en plus les gradients du décodeur à chaque pas ; la faisabilité et la vitesse
doivent être mesurées dans une campagne séparée. FLUX demande un port encore plus important et
n'est pas la prochaine variable à mélanger à la correction de SR-MPGD.

Sources : [papier WACV 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Liao_DiffQRCoder_Diffusion-Based_Aesthetic_QR_Code_Generation_with_Scanning_Robustness_Guided_WACV_2025_paper.pdf),
[dépôt officiel DiffQRCoder](https://github.com/jwliao1209/DiffQRCoder),
[QR Monster v2 SD1.5](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster),
[pipeline ControlNet SDXL](https://huggingface.co/docs/diffusers/main/api/pipelines/controlnet_sdxl),
[QR Monster SDXL](https://huggingface.co/monster-labs/control_v1p_sdxl_qrcode_monster) et
[Nacholmo QR Pattern SDXL](https://huggingface.co/Nacholmo/controlnet-qr-pattern-sdxl).

## Exécution

Après `git pull`, reconstruire et importer l'image notebook sur Linux :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
docker build -f Dockerfile.notebook -t prooftag-qr-notebook:dev .
docker save prooftag-qr-notebook:dev | sudo k3s ctr images import -
kubectl rollout restart deployment/prooftag-qr-notebook -n qr-core
```

Puis depuis PowerShell Windows :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
.\scripts\notebook-remote.ps1 -Notebook 09_diffqrcoder_faithful_srmpgd.ipynb
```

Dans Jupyter, utiliser `Run > Run All Cells`. En cas d'interruption, renseigner `RESUME_RUN_NAME`
avec le dossier existant avant de relancer. Les durées originales et les tenseurs sont conservés.
La dernière cellule donne les commandes exactes `kubectl cp` puis `scp`. Enfin :

```powershell
.\scripts\notebook-remote.ps1 -Stop
```

## Porte de décision suivante

E012 n'annonce aucun nouveau taux avant exécution RTX. Après la campagne :

1. si SR-MPGD améliore la lecture sans effondrement CLIP, conserver sa meilleure itération ;
2. si 100 pas dominent 40, utiliser 100 comme baseline de robustesse, puis rechercher une
   accélération sans changer de modèle ;
3. si aucun original n'est lisible malgré une faible MER, traiter séparément détection des finders
   et quiet zone avant toute recherche esthétique ;
4. ouvrir E013a pour les finders/quiet zone si nécessaire ; seulement ensuite, E013b comparera
   SDXL à la meilleure configuration SD1.5.
