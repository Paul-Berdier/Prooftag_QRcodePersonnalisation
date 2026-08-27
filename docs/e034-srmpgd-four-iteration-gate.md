# E034 — porte SR-MPGD appariée à quatre itérations

## Pourquoi E034 existe

E033 a validé localement une seule mise à jour des équations 13–14 : sur le cas
`e033_simple_greenhouse`, le gradient image et le gradient latent sont finis et non nuls, le
latent est effectivement déplacé, et la Scanning Robust Loss (SRL) diminue sans OOM ni saturation
destructive. Ce résultat est un acquis mécanistique, pas une preuve de scannabilité : le parent
Stage 2 et les deux sorties E033 restent à `0/37` QR-Verify.

E034 prolonge exactement cette paire pendant quatre mises à jour. Il doit déterminer si la
descente observée au premier pas reste active, si LPIPS contribue réellement dès que le candidat
s'éloigne de sa référence, et si le gain QR ne détruit pas l'image. Il ne change ni le prompt, ni
le seed, ni le parent Stage 2 et n'autorise aucune campagne multi-prompt.

## Contrat expérimental gelé

Le plan `e034-srmpgd-four-iteration-gate-v1` fixe :

- le prompt `e033_simple_greenhouse` : une serre ensoleillée avec plants de tomates et pots en
  terre cuite ;
- le seed `51001` ;
- le payload court E033 **figé** `https://ptag.io/t/e033`, avec correction d'erreur `M` ;
- la même recette Stage 2 publique : initialisation `public_random`, cible QR binaire exacte,
  ControlNet `1,05`, SRG `50` et PG `20` ;
- les équations papier `SRL + 0,01 × LPIPS`, `gamma=1000`, quatre mises à jour, loss scaling
  `32768`, LPIPS/VGG sur CPU et crop exact de `78 px` ;
- deux calculs appariés : VAE dans la précision du modèle et VAE temporairement en FP32 ;
- les jalons directs `i0`, `i1`, `i2` et `i4` ;
- aucun oracle de sélection : la sortie SR-MPGD est obligatoirement l'état final `i4` ;
- aucune livraison et aucun élargissement automatiques.

Le `plan_id` lie le payload haché, les méthodes complètes, le contrat de prédiction, les jalons,
les métriques, le commit et les digests OCI de l'API et du notebook. Une modification de code, de
configuration ou de provenance doit donc créer un nouveau plan et un nouveau dossier de reprise.
Avant toute campagne GPU, le notebook exécute aussi QR-Verify localement deux fois sur le QR
binaire exact et exige `37/37`. L'identité du moteur, son empreinte d'implémentation et ce résultat
sont liés au plan.

## Quatre sorties finales, un seul parent

Une seule campagne atomique génère quatre sorties :

1. `diffqrcoder_stage1`, référence esthétique et source commune ;
2. `e033_public_demo_srpg`, parent Stage 2 public inchangé ;
3. `e034_equation_srmpgd_fp16`, quatre mises à jour avec le VAE dans sa précision modèle ;
4. `e034_equation_srmpgd_fp32`, branche primaire avec le VAE temporairement en FP32.

Les deux branches SR-MPGD doivent réutiliser exactement le même Stage 1 et le même latent Stage 2.
Le notebook vérifie leurs identifiants de provenance, les SHA-256 du Stage 1, du parent et du
latent, puis exige que leurs rasters `i0` soient pixel-identiques au parent E033 reproduit. Une
divergence de parent interdit toute comparaison scientifique.

## Trajectoire observée

La trace primaire doit contenir les cinq états `i0`, `i1`, `i2`, `i3` et `i4`. Les rasters
affichés et archivés sont :

- le parent Stage 2 ;
- le même latent redécodé par le VAE sans mise à jour ;
- `i0`, état initial exact ;
- `i1`, après une mise à jour ;
- `i2`, après deux mises à jour ;
- `i4`, état final après quatre mises à jour.

L'état `i3` reste présent dans la trace pour prouver la quatrième mise à jour, même s'il n'est pas
une colonne de la planche principale. Le SHA-256 du raster final doit être celui du jalon `i4`.
Une itération absente à la suite d'un arrêt numérique est enregistrée explicitement ; elle ne doit
jamais être remplacée par le parent ou par un état antérieur.

## Télémétrie SRL, LPIPS et objectif

Pour chaque mise à jour `i0 → i1` jusqu'à `i3 → i4`, E034 conserve au minimum :

- la SRL et son gradient RMS en espace image ;
- la loss LPIPS, son gradient image RMS et sa contribution pondérée par `0,01` ;
- le gradient image RMS de l'objectif complet ;
- le gradient latent RMS obtenu par le VJP du VAE ;
- le facteur de loss scaling effectivement retenu ;
- le pas demandé, le pas appliqué et le déplacement latent observé ;
- l'objectif total, le MER réel, le pic CUDA agrégé, les modules offloadés et la phase courante
  lorsqu'une OOM survient.

Conformément aux Sec. 3.2–3.3 et à l'Eq. 13, la référence LPIPS est le tenseur flottant figé
`x0 = D(z0)` du Stage 2. Elle n'est ni l'image Stage 1, ni un aller-retour PIL/uint8 qui ajouterait
clamp et quantification absents du papier. Le notebook vérifie son mode et le SHA-256 de son
raster témoin. À `i0`, le candidat est exactement sa propre référence : la loss LPIPS et son
gradient doivent donc être nuls à la tolérance numérique près. De `i1` à `i3`, la loss LPIPS, son
gradient image et sa contribution pondérée doivent être finis et strictement positifs.
L'objectif publié doit être numériquement cohérent avec `SRL + 0,01 × LPIPS`. Le déplacement latent
cumulé depuis `i0` est
contrôlé séparément et doit rester non nul aux états `i1` à `i4` ; il n'est jamais assimilé à un pas
élémentaire. Pour chacune des quatre mises à jour papier, le facteur de pas doit valoir `1` et le pas effectivement appliqué doit être
égal au pas demandé : cela prouve qu'aucun limiteur caché ne modifie Eq. 14. E034 compte aussi le
nombre de transitions où l'objectif est
monotone, mais sa porte minimale exige que l'objectif final et la SRL finale soient inférieurs à
leurs valeurs initiales.

Cette référence est volontairement fidèle aux équations du papier. Le dépôt public DiffQRCoder
passe pour sa part l'image Stage 1 comme `ref_image` à la loss combinée de son post-traitement.
E034 ne confond donc pas « reconstruction des Eq. 13-14 du PDF » et « exécution littérale de ce
chemin du dépôt public » ; cette différence est enregistrée dans le protocole et dans la trace.

## Scores QR et contrôles visuels

Le parent, les deux redécodages VAE et les jalons SR-MPGD sont rescorrés localement par contenu.
Le journal persistant évite de recalculer un raster déjà vérifié et associe chaque mesure à sa
provenance. Les mesures comprennent :

- QR-Verify sur ses 37 presets et le nombre de payloads exacts ;
- MER ;
- MAE et changement par rapport au parent ;
- pixels écrêtés, saturation moyenne et ratio de forte saturation ;
- pour les quatre sorties finales seulement, CLIP-Aesthetic, CLIPScore et HPS v2.1 avec
  provenance des modèles épinglée ;
- pour la cohérence numérique finale FP16/FP32, PSNR et écart maximal par canal.

Le témoin VAE sans mise à jour est scoré séparément : une erreur introduite par la seule
redécompression VAE ne doit pas être attribuée à Eq. 14. La planche compare ensuite le parent, ce
témoin et `i0/i1/i2/i4` en FP16 et FP32 avant d'afficher le verdict.

Les gardes automatiques limitent le changement moyen, l'écrêtage, la saturation et les pertes de
proxies esthétiques. Elles restent des garde-fous, pas un jugement esthétique définitif. Une revue
humaine des rasters natifs demeure obligatoire.

## Verdicts séparés

E034 ne fusionne pas des objectifs différents dans une note unique. Il publie séparément :

- `mechanism_pass` : provenance exacte, trace complète, gradients et déplacements valides,
  cohérence de l'objectif, sortie finale `i4` et cohérence numérique FP16/FP32 ;
- `scan_progress_pass` : amélioration de QR-Verify ou diminution du MER par rapport au parent ;
- `single_case_qr_verify_pass` : au moins un des 37 presets restitue exactement le payload final ;
- `visual_proxy_pass` : gardes de changement, clipping, saturation, CLIP et HPS respectées ;
- `manual_visual_review_required` : toujours vrai ;
- `production_ready` et `automatic_expansion_authorized` : toujours faux dans E034.

Un PASS mécanistique sans payload QR-Verify exact reste un échec de scannabilité. Inversement, un
gain QR accompagné d'une dégradation visuelle au-delà des gardes reste un STOP. Le prochain petit
holdout ne peut être conçu qu'après les PASS mécanistique, QR-Verify et visuel, puis une revue
humaine favorable.

## Reprise et erreurs techniques

Le runner écrit atomiquement le plan, les prédictions, l'état et les exports sous
`/data/e034-srmpgd-four-iteration-gate/<plan_id>`. Après une coupure, une campagne encore active est
reprise. Un export local absent ou corrompu peut être retéléchargé depuis la campagne terminée sans
nouvelle génération GPU.

E034 n'autorise qu'une tentative. Une campagne terminale contenant une erreur n'est jamais
régénérée automatiquement. Elle produit une archive distincte
`*-technical-failure.tar.gz` contenant le plan, l'état, les exports partiels, les messages des
trials, les diagnostics runtime et les artefacts disponibles. Cette archive décrit un incident et
ne porte aucun verdict scientifique. Son manifeste lie une identité canonique au SHA-256 de chaque
fichier régulier ; le lecteur vérifie l'inventaire complet, le document de checksums et chaque
contenu avant de présenter l'archive.

Un arrêt attendu du mécanisme — gradient nul, valeur non finie, jalon absent cohérent avec la trace
— est un STOP scientifique. Il conserve les images et les traces puis crée quand même l'archive
scientifique. Relancer le notebook ne doit ni écraser un plan valide, ni recommencer ses
générations.

Une exception imprévue après les générations est elle aussi interceptée : les résultats déjà
produits et l'erreur sont placés dans une archive `*-post-gpu-*.tar.gz`, les cellules suivantes sont
ignorées proprement et aucune nouvelle campagne n'est envoyée. Une archive scientifique existante
n'est acceptée comme résultat courant que si l'identité de son analyse correspond au manifeste
recalculé ; elle n'est jamais écrasée silencieusement.

Les deux archives d'incident suivent la même règle sans écrasement. Si l'archive primaire est
ancienne, incomplète ou corrompue, elle reste intacte et une récupération dont le nom contient
l'identité canonique courante est créée. Les contrôles portent sur tous les membres, et pas seulement
sur la présence du fichier d'erreur. Pour une panne post-GPU, le premier état disponible du plan,
des exports, des images et de l'analyse partielle est figé dans un sous-dossier `evidence` afin
qu'un nouveau `Run All` ne change pas rétroactivement l'incident.

## Archive scientifique

Après une exécution techniquement complète, l'archive scientifique contient :

- le plan, les prédictions, l'état, le schéma API, l'identité runtime et l'export brut ;
- les quatre sorties finales, les jalons directs et les témoins VAE ;
- les traces, la télémétrie de gradient et les diagnostics mémoire agrégés de l'export brut ;
- les scores QR et visuels des onze rasters, ainsi que les scores perceptuels des quatre sorties
  finales ;
- les audits de provenance, les portes, les verdicts et les planches-contact ;
- un manifeste SHA-256 de tous les artefacts.

L'archive est créée atomiquement sous `/data`, puis rouverte et vérifiée fichier par fichier avant
d'être copiée avec son checksum dans `/workspace/downloads`. Une porte scientifique échouée ne
supprime donc jamais les preuves nécessaires au diagnostic.

## Limites d'interprétation

E034 porte sur un prompt, un seed, un payload et le parent de démonstration publique. Même un PASS
complet ne prouve pas :

- la généralisation à d'autres prompts ou QR ;
- une probabilité de lecture sur téléphone ou après impression ;
- la reproduction du QArt privé ou du Stage 2 exact des auteurs ;
- une supériorité esthétique ;
- la capacité du conseiller de paramètres à choisir une recette de production.

QR-Verify reste un banc logiciel et CLIP/HPS restent des proxies. E034 autorise au mieux la
conception d'un petit holdout séparé ; il n'autorise ni entraînement, ni campagne massive, ni
livraison automatique.

## Exécution du notebook

Après commit et push, déployer sur le serveur Linux :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git fetch origin
git switch main
git pull --ff-only origin main
bash scripts/deploy-e034-notebook.sh
```

Le déploiement mémorise les nombres réels de réplicas API et vLLM avant la première
bascule GPU. Cette sauvegarde n'est pas remplacée par l'état temporaire `API=1 / vLLM=0` :
la commande d'arrêt ci-dessous restaure donc bien la charge qui existait avant E034, y compris
après un échec partiel du déploiement.

Puis, depuis PowerShell sur le PC :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
git pull --ff-only
.\scripts\notebook-remote.ps1 -Notebook 29_e034_srmpgd_four_iteration_gate.ipynb
```

Dans Jupyter, utiliser **Run > Run All Cells**. Inspecter d'abord les deux planches, puis les portes
et enfin l'archive indiquée dans `/workspace/downloads`. Pour fermer proprement le notebook et
restaurer les charges GPU précédentes :

```powershell
.\scripts\notebook-remote.ps1 -Stop
```
