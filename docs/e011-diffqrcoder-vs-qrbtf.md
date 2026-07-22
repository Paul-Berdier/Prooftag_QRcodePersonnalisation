# E011 — comparaison contrôlée DiffQRCoder / QRBTF public

## Objectif

E011 répond à une seule question : à QR, prompt et seed identiques, quel workflow offre le
meilleur compromis entre lecture et esthétique ? Aucune troisième famille de modèle n'est ajoutée.
La matrice contient quatre prompts de complexité croissante et quatre sorties par prompt :

1. DiffQRCoder-paper, sortie SRPG ;
2. DiffQRCoder-paper, puis SR-MPGD ;
3. reproduction publique QRBTF, sortie double ControlNet ;
4. reproduction publique QRBTF, puis le même SR-MPGD.

Le notebook associé est
[`../notebooks/08_diffqrcoder_vs_qrbtf_four_prompts.ipynb`](../notebooks/08_diffqrcoder_vs_qrbtf_four_prompts.ipynb).

## Ce qui peut réellement être reproduit

| Branche | Fondation | Conditionnement | Scheduler | Statut |
|---|---|---|---|---|
| DiffQRCoder-paper | Cetus-Mix Whalefall fp16 | QR Code Monster v2 | DDIM | dépôt public figé au commit `e24ea73`, initialisation Stage 2 corrigée selon le papier |
| QRBTF-public-reproduction | même Cetus-Mix | QR Code Monster v2 + `latentcat/control_v1p_sd15_brightness` | DPM++ SDE Karras | reproduction locale des éléments publiquement décrits, pas le service propriétaire |

Deux limites sont bloquantes pour une reproduction bit-à-bit :

- QRBTF documente Stable Diffusion + ControlNet et ses paramètres, mais ne publie ni le checkpoint
  de son backend IA, ni son code d'inférence, ni ses latents intermédiaires. L'étiquette
  `QRBTF officiel` serait donc fausse.
- le papier DiffQRCoder utilise une cible QArt, mais le dépôt public ne contient pas le générateur
  Reed–Solomon QArt. E011 emploie une cible visuelle transparente, dérivée du Stage 1, qui conserve
  la matrice et renforce les centres des modules. Le fichier est exporté pour audit et n'est jamais
  collé sur l'image finale.

Sources techniques : [papier DiffQRCoder WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Liao_DiffQRCoder_Diffusion-Based_Aesthetic_QR_Code_Generation_with_Scanning_Robustness_Guided_WACV_2025_paper.html),
[dépôt DiffQRCoder](https://github.com/jwliao1209/DiffQRCoder),
[documentation QRBTF](https://docs.qrbtf.com/introduction),
[paramètres QRBTF](https://docs.qrbtf.com/essentials/param-list),
[QR Code Monster v2](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster) et
[Brightness ControlNet](https://huggingface.co/latentcat/control_v1p_sd15_brightness).

## Pourquoi E010 est rejeté comme baseline de résultat

L'archive E010 `20260722T114948Z-e010-diffqrcoder-official-v1-seed1-Copy1.tar.gz` est complète,
mais aucune sortie n'est livrable :

| Sortie | Lecture exacte | MER | CLIP-aesthetic | CLIPScore | Temps |
|---|---:|---:|---:|---:|---:|
| Stage 1 | 0/26 | 29,36 % | 7,19 | 0,844 | 19,1 s |
| SRPG | 0/26 | 18,19 % | 3,88 | 0,567 | 95,0 s |
| SRPG + SR-MPGD | 0/26 | 16,51 % | 3,98 | 0,599 | 107,0 s |

Le guidage réduit donc la MER, mais détruit l'esthétique et ne franchit aucun décodeur. La cause
méthodologique principale est l'écart entre le dépôt et le papier : le Stage 2 public repart d'un
bruit aléatoire, alors que l'algorithme publié part d'un encodage bruité du Stage 1. E011 corrige
ce point explicitement et ne reprend pas les chiffres E010 comme preuve d'efficacité.

## Protocole E011

- QR unique : version 3, correction M, masque 4, quiet zone 4, module 16 px, image 592 px.
- Quatre prompts fixes : simple, moyen, détaillé et complexe.
- Une seed fixe par prompt, réutilisée par les deux branches.
- Quarante pas par diffusion. Chaque estimation latente est décodée et enregistrée ; l'affichage
  live ne montre qu'un pas sur cinq pour ne pas saturer Jupyter.
- DiffQRCoder effectue 40 pas Stage 1 puis 40 pas Stage 2 avec SRG 500 et PG 3.
- La reproduction QRBTF utilise Monster à 1,0 sur toute la diffusion et Brightness à 0,25 de 40 %
  à 80 % des pas.
- Chaque sortie de base reçoit ensuite 20 itérations du même SR-MPGD, taux 0,1.
- Les modèles sont chargés séquentiellement pour tenir sur la RTX 4000 Ada 20 Gio.

La campagne produit 16 lignes et s'arrête si ce total n'est pas atteint. Les images finales, les
40 images de chaque diffusion, les 20 images de chaque SR-MPGD, les GIF, paramètres, temps,
validations et scores sont persistés sous `/data/notebook-runs`.

Chaque résultat est ajouté immédiatement à `results.jsonl`. Après une interruption, recopier le
nom du dossier existant dans `RESUME_RUN_NAME` : les couples prompt/méthode déjà entièrement
mesurés sont ignorés. Si un couple n'avait produit qu'une de ses deux variantes, cette ligne
partielle est retirée puis le couple est recalculé proprement, sans doublon dans l'agrégat final.

## Métriques et ordre de décision

1. `software_ssr` : payload exact sur tous les couples décodeur × dégradation disponibles ;
2. `original_ssr` : payload exact sur l'image originale uniquement ;
3. pire SSR par décodeur et MER ;
4. uniquement après la lecture : CLIP-aesthetic puis CLIPScore ;
5. temps total instrumenté et coût incrémental SR-MPGD.

Le SSR logiciel n'est pas le SSR physique du papier. Le notebook crée donc `physical-ssr.csv`
avec dix tentatives par image sur Pixel 7 et iPhone 13. Ces cellules restent vides jusqu'aux vrais
tests ; aucun taux physique n'est inventé.

## Exécution et récupération

Après reconstruction de l'image notebook sur le serveur, lancer depuis PowerShell :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
.\scripts\notebook-remote.ps1 -Notebook 08_diffqrcoder_vs_qrbtf_four_prompts.ipynb
```

Dans Jupyter, exécuter `Run > Run All Cells`. La dernière cellule affiche deux commandes : un
`kubectl cp` à lancer sur Linux pour sortir l'archive du pod, puis un `scp` à lancer sur Windows.
Enfin, restaurer les workloads GPU avec :

```powershell
.\scripts\notebook-remote.ps1 -Stop
```

## Porte pour l'étape suivante

Il n'y aura aucune optimisation de paramètres avant cette matrice. Si une variante atteint la
lecture stricte sur les quatre prompts, elle devient baseline esthétique. Sinon, les traces
permettront d'identifier si l'échec vient du prompt, du dernier débruitage ou de SR-MPGD. Le
balayage suivant portera alors uniquement sur les paramètres influents de la branche gagnante,
pas sur un mélange de nouvelles méthodes.
