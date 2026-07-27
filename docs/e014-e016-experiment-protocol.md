# Protocole E014–E016 : blueprint, fusion latente, backbone et surrogate

## Pourquoi cette série remplace une nouvelle recherche globale

E013 a mélangé trop de causes possibles : géométrie, backbone, ControlNet, guidage, nombre de pas
et paramètres de correction. E014–E016 impose une progression où une étape ne commence qu'avec un
artefact explicite de l'étape précédente :

```text
E014A : construire une condition QR fiable
   │
   ├── artefact : blueprint exact-payload sélectionné + matrice + géométrie + Stage 1
   ▼
E014B : mesurer la fusion latente, canal par canal et timestep par timestep
   │
   ├── artefact : configuration de fusion observée + traces complètes
   ▼
E015 : choisir, séparément, une meilleure référence esthétique
   │
   ├── artefact : comparaison SD1.5 / SDXL / FLUX, sans prétendre à la compatibilité QR
   ▼
E016 : apprendre une loss différentiable sur les réponses des vrais décodeurs
   │
   └── artefact : dataset, modèle expérimental, calibration et audit adversarial
```

Un résultat logiciel strict signifie que tous les décodeurs disponibles lisent le payload attendu
sur les treize scénarios. Il ne signifie pas encore 99 % de scans physiques.

## E014A — vrai QArt et variantes exact-payload

Notebook :
[`11_e014a_qart_blueprint_bakeoff.ipynb`](../notebooks/11_e014a_qart_blueprint_bakeoff.ipynb)

Quatre conditions sont appariées dans la même Stage 2 DiffQRCoder :

| Variante | Payload | ECC | Nature |
|---|---:|---:|---|
| `binary_mask4_m` | exact | M | QR standard, baseline |
| `qart_fragment_l` | URL canonique | L | vrai QArt public, fragment ajouté |
| `exact_payload_mask_search_m` | exact | M | meilleur des huit masques standards |
| `adaptive_exact_payload_m` | exact | M | centres adaptés à la luminance, fonctions binaires |

Le binaire `qart` est construit depuis `andrewyur/qart` au commit
`6e0e00804a1994db7098432c19fadfc552071e30`. Cette implémentation ajoute un fragment `#…` pour
obtenir des degrés de liberté Reed–Solomon et suppose une correction L. Elle est donc testée en
comparant l'URL après suppression du fragment. Elle n'est jamais présentée comme exact-payload.
Son CLI n'expose pas de seed déterministe : E014A exécute trois répétitions pour chacun des cinq
seuils, conserve les quinze PNG, leurs SHA-256 et leur tableau `qart-screening.json`, et ne
régénère jamais ces blueprints pendant une reprise. Chaque frame de diffusion possède également
son propre numéro de pas ; ces deux invariants sont couverts par des tests de non-régression.

La variante exacte recherche les huit masques permis par la norme. L'adaptatif part du meilleur
masque et calcule, module par module, la plus petite zone centrale noire ou blanche nécessaire
d'après la luminance locale. Les finder, timing, alignment, format/version patterns restent
binaires. Ce procédé est une méthode Prooftag documentée, pas une reproduction QArt.

Artefacts essentiels, par prompt :

- `stage1-reference.png`, `stage1.safetensors`, toutes les frames et le GIF ;
- les quatre dossiers `blueprints/*`, matrices, métadonnées et validations ;
- les quatre Stage 2, leurs latents finaux, frames, GIF, validations et scores ;
- `selected-blueprint.png`, `selected-matrix.npy`, `selected-meta.json`.

La sélection E014B n'accepte que les variantes exact-payload. L'ordre est : porte stricte, SSR,
pire décodeur, pire scénario, CLIP-aesthetic, CLIPScore, visibilité de grille.

### Protocole déterministe E014A v2

La répétition du 27 juillet a montré que deux branches alimentées par le même PNG pouvaient
diverger. E014A v2 ajoute donc un cinquième passage, `binary_mask4_m_duplicate`, strictement
identique au binaire mais inéligible à la sélection. Avant Stage 1 et chaque Stage 2, le notebook
réinitialise Python, NumPy, PyTorch CPU et tous les générateurs CUDA. cuDNN déterministe et les
algorithmes déterministes PyTorch sont activés, avec la configuration cuBLAS enregistrée.

Chaque ligne conserve les SHA-256 du tensor Stage 1, de son image, de la condition, du latent
initial, du latent final et de l'image finale. `determinism-audit.json` compare le binaire et son
duplicata. Si les hashes finaux divergent, la porte échoue et l'expérience doit être interprétée
comme stochastique, jamais comme une comparaison appariée exacte.

`SEED_OFFSET = 30000` transforme automatiquement les seeds de base en
`31101/32202/33303/34404`. Il ne faut plus modifier manuellement les quatre dictionnaires dans
Jupyter. L'offset et les seeds effectives sont inscrits dans le manifeste.

## E014B — fusion latente inspirée de FreeQR

Notebook :
[`12_e014b_freeqr_latent_fusion.ipynb`](../notebooks/12_e014b_freeqr_latent_fusion.ipynb)

Le notebook retrouve le dernier E014A ou utilise le dossier renseigné dans `E014A_RUN_DIR`.
Il charge le blueprint sélectionné, sa matrice, sa géométrie et le tensor Stage 1. Il reprend le
même latent initial pour chaque variante.

La publication FreeQR ([DOI 10.1109/TMM.2026.3668595](https://doi.org/10.1109/TMM.2026.3668595))
décrit la fusion d'un canal précis avec le latent bruité du blueprint au timestep correspondant,
puis un guidage par les erreurs de scan dans le latent. Les détails exécutables complets et une
implémentation officielle n'étant pas disponibles dans le projet, E014B teste explicitement les
choix manquants au lieu de prétendre reproduire des paramètres non publiés.

L'ablation se déroule en quatre phases :

1. baseline puis canaux latents 0 à 3 ;
2. fenêtres `early`, `middle`, `late`, `all` sur le meilleur canal ;
3. coefficients `0.05`, `0.10`, `0.15`, `0.22` ;
4. meilleur réglage avec une loss centrale différentiable à trois pas d'apprentissage.

Le callback de Diffusers intervient après un pas DDIM. Le QR latent est donc bruité au timestep
suivant, inscrit dans `target_timestep_after_step`. La dernière étape utilise le latent QR propre.
La variante avec gradient est séparée : sa loss de marge n'est ni le SRL officiel DiffQRCoder, ni
SR-MPGD. Le manifeste utilise volontairement l'étiquette
`FreeQR-inspired channel/timestep reconstruction`.

La première campagne utilise `p1_simple`. Le réglage gagnant doit ensuite être confirmé en
modifiant `PROMPT_ID` pour les trois autres prompts ; une réussite unique n'est pas une recette.

## E015 — références esthétiques

Notebook :
[`13_e015_aesthetic_backbone_reference.ipynb`](../notebooks/13_e015_aesthetic_backbone_reference.ipynb)

E015 charge un seul modèle à la fois :

- Cetus-Mix SD 1.5, 30 pas ;
- SDXL Base 1.0, 30 pas ;
- FLUX.1-schnell, 4 pas.

Le tableau compare temps chaud, pic de VRAM, CLIP-aesthetic et CLIPScore sur quatre prompts.
Le constructeur adaptatif E014A est appliqué à chaque référence et validé, mais aucune Stage 2 QR
n'est lancée. E015 répond donc à « quelle référence est la plus utile/esthétique ? », pas à
« quel modèle remplace DiffQRCoder ? ». Une migration SDXL ou FLUX nécessitera un mécanisme de
contrôle compatible et une expérience dédiée.

FLUX utilise l'offload CPU **séquentiel** sur la carte 20 Go, avec slicing et tiling du VAE.
L'offload par modèle ne convient pas ici : il tente de déplacer le Transformer 12B BF16 entier sur
le GPU et occupe les 19,67 Gio avant même le premier pas. L'offload séquentiel ne charge que le
sous-module actif ; il est nettement plus lent mais conserve les poids BF16 et permet à E015 de
mesurer la référence sans quantification. Le mode exact est enregistré dans chaque ligne.
Le premier lancement peut surtout être limité par le téléchargement et l'espace du cache Hugging
Face. Les modèles sont libérés entre familles.
Son encodeur T5 exige `sentencepiece==0.2.0` et `protobuf==5.29.3`. Ces deux dépendances sont
installées et contrôlées pendant la construction de l'image notebook ; E015 les importe avant tout
téléchargement de modèle afin qu'une image obsolète échoue immédiatement.

`black-forest-labs/FLUX.1-schnell` reste publiquement visible mais ses fichiers sont restreints.
Avant E015, le compte Hugging Face doit accepter les conditions sur la page du modèle. Créer
ensuite le secret sans placer le jeton dans Git ni dans l'historique de commande :

```bash
read -rsp "Jeton Hugging Face : " HF_ACCESS_TOKEN
echo
kubectl create secret generic prooftag-huggingface -n qr-core \
  --from-literal=token="$HF_ACCESS_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -
unset HF_ACCESS_TOKEN
```

Le Deployment injecte ce secret de façon optionnelle sous `HF_TOKEN`. E015 teste l'accès aux trois
modèles avant leur chargement, enregistre seulement le résultat dans `model-access.json` et
n'écrit jamais le jeton. Après création ou remplacement du secret, recréer le pod avec
`notebook-remote.ps1 -Reset` ; un simple redémarrage du kernel ne met pas à jour l'environnement du
pod.

## E016 — surrogate différentiable

Notebook :
[`14_e016_differentiable_scan_surrogate.ipynb`](../notebooks/14_e016_differentiable_scan_surrogate.ipynb)

E016 prend uniquement les PNG finaux dont le contrat est `exact`, applique chaque scénario réel,
puis demande à OpenCV, ZBar et ZXing-cpp si le payload exact est lu. Les sources QArt à URL
canonique sont exclues et consignées. Une ligne contient l'image réellement transformée et un
label par décodeur. Son nom de fichier dépend du run, du chemin relatif, de la variante et du
SHA-256 source ; toute collision arrête le notebook. Le split train/validation/test est groupé par
contexte `(payload, prompt, seed)` : deux variantes dégradées d'un même contexte ne peuvent pas
être réparties entre entraînement et test.

Le CNN multi-sorties s'entraîne seulement si les seuils de volume, de groupes et de classes sont
atteints à la fois par ligne et par groupe. Une recherche déterministe sélectionne seulement un
split où chaque décodeur possède les deux classes dans train, validation et test ; sinon le
notebook demande de nouveaux prompts/seeds. Les métriques holdout sont AUCPR, ROC-AUC, Brier et
calibration. L'audit final optimise une image holdout avec le gradient du CNN sous une contrainte
de `±8/255`, puis mesure les vrais décodeurs avant/après. Si seul le CNN progresse, le surrogate a
été trompé, la porte échoue et aucun TorchScript n'est exporté.

Le `DataLoader` adapte ses workers à la mémoire partagée réellement disponible. Avec le `/dev/shm`
Docker historique d'environ 64 Mio, il utilise zéro worker et charge dans le processus principal.
Le Deployment notebook monte désormais un `emptyDir` mémoire de 2 Gio sur `/dev/shm` ; dans ce
profil, E016 utilise deux workers, un seul lot préchargé par worker et inscrit cette configuration
dans `surrogate-card.json`. Cela évite qu'un worker meure par `SIGBUS` au milieu d'une époque.

Un groupe correspond à un contexte `(payload, prompt, seed)`, et non à un simple fichier. Les
quatre contextes initiaux d'E014A ne suffisent donc volontairement pas au seuil par défaut de douze
groupes : il faut ajouter des campagnes avec d'autres prompts/seeds dans `SOURCE_RUNS`. Abaisser ce
seuil ferait fonctionner le code mais donnerait une validation trompeuse par fuite de contexte.

Les captures physiques suivent le fichier `physical-captures-template.csv`. Tant qu'aucune capture
n'est fournie, `production_usable` reste faux même si les métriques numériques sont bonnes.

## Installation et lancement

Le Docker notebook a de nouvelles dépendances : binaire QArt compilé et figé, ZXing-cpp et Kornia.
Il faut donc reconstruire l'image notebook avant le premier lancement :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git pull
docker build -f Dockerfile.notebook -t prooftag-qr-notebook:dev .
docker save prooftag-qr-notebook:dev | sudo k3s ctr images import -
kubectl apply -k deploy/k8s
```

Depuis PowerShell sur le PC :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
git pull
.\scripts\notebook-remote.ps1 -Notebook 11_e014a_qart_blueprint_bakeoff.ipynb
```

Après l'échec du contrôle dupliqué E014A v2, la prochaine étape n'est plus E014B. Lancer le
diagnostic court E014C :

```powershell
.\scripts\notebook-remote.ps1 -Reset -Notebook 15_e014c_stage2_determinism_diagnostic.ipynb
```

Il réutilise les artefacts E014A persistés. E014C v2 exécute huit diagnostics de cinq pas avec
checkpointing, puis quatre contrôles de quarante pas : guidance complète avec callback minimal,
puis avec le callback de décodage E014A. Le mode strict reste `warn_only=False`. Envoyer l'archive
`e014c-stage2-determinism-isolation-v2` avant de poursuivre.

E014B ne sera lancé qu'après correction ou caractérisation de la divergence :

```powershell
.\scripts\notebook-remote.ps1 -Reset -Notebook 12_e014b_freeqr_latent_fusion.ipynb
```

E015 et E016 se lancent avec les noms `13_e015_aesthetic_backbone_reference.ipynb` et
`14_e016_differentiable_scan_surrogate.ipynb`. `-Reset` supprime le pod Jupyter et tous ses
kernels, attend la libération réelle de la VRAM, recrée le pod et ouvre le notebook suivant, sans
restaurer temporairement l'API/vLLM. À la fin de toute la session, utiliser `-Stop` pour leur
restituer le GPU.

## Portes de décision

| Étape | Continuer si | Revenir en arrière si |
|---|---|---|
| E014A | une condition exacte réduit la grille sans baisser le pire SSR | aucun blueprint exact ne passe son propre contrôle |
| E014B | le gain tient sur les quatre prompts et ne détruit pas CLIP | seul un prompt ou un seul décodeur progresse |
| E015 | une référence améliore l'esthétique et reste intégrable | le coût/VRAM augmente sans gain apparié |
| E016 | holdout calibré + vrais décodeurs améliorés | classes pauvres, fuite, proxy adversarial |

Seulement après ces portes, le dataset contiendra une cible suffisamment stable pour entraîner un
ControlLoRA/ControlNet Prooftag et un sélecteur de paramètres sans apprendre les erreurs du pipeline.

Les résultats et l'audit forensique des six premières archives sont consignés dans
[`e014-e016-results-audit-2026-07-27.md`](e014-e016-results-audit-2026-07-27.md).
