# État de l'art 2024-2026 et nouvelle direction Prooftag QR

Date de la recherche : 23 juillet 2026

## Conclusion exécutive

Il n'existe pas, à cette date, de modèle public prêt à intégrer qui garantisse des QR artistiques
génériques à 99 % selon notre porte Prooftag. Les publications récentes confirment cependant que
notre pipeline actuelle essaie de résoudre le problème trop tard et avec une cible trop rigide.

Les meilleurs travaux ne se contentent plus d'imposer un QR binaire à un ControlNet. Ils combinent :

1. une réorganisation légale du QR avant la diffusion, en exploitant les degrés de liberté de
   l'encodage et de Reed–Solomon ;
2. un blueprint dont la luminance ressemble déjà à l'image souhaitée ;
3. une correction différentiable dans le latent, avec des contraintes spatiales et des simulations
   de lecture ;
4. un contrôle de livraison indépendant du modèle.

La prochaine étape ne doit donc pas être « remplacer SD 1.5 par SD 2.1, SDXL ou FLUX dans
DiffQRCoder ». La voie la plus défendable est :

> référence esthétique moderne → QArt/QAB exact → DiffQRCoder SD 1.5 → fusion latente de type
> FreeQR → correction adaptative → validation multi-décodeur et ISO → livraison ou rejet.

Cette architecture conserve la seule baseline qui a obtenu un 26/26 dans E013, tout en remplaçant
le QR binaire qui crée la grille visible.

## Ce qui a été trouvé

| Travail ou composant | Apport utile | Résultat annoncé | Code réellement disponible | Décision Prooftag |
|---|---|---|---|---|
| [DiffQRCoder, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Liao_DiffQRCoder_Diffusion-Based_Aesthetic_QR_Code_Generation_with_Scanning_Robustness_Guided_WACV_2025_paper.html) | SRPG pendant la diffusion et SR-MPGD dans le latent | 99 % SSR dans son protocole, plus de 95 % dans son test sévère | [Pipeline publique](https://github.com/jwliao1209/DiffQRCode), mais le chemin QArt complet du papier n'est pas fourni | Conserver comme baseline, pas comme preuve de notre 26/26 |
| [Text2QR, CVPR 2024](https://arxiv.org/html/2403.06452) | QR Aesthetic Blueprint : histogramme polarisé, réorganisation de modules, blocs centraux de taille adaptative ; SELR dans le latent | plus de 96 % dans un protocole physique de 50 essais, décodage en moins de 3 s | Le [dépôt officiel](https://github.com/mulns/Text2QR) ne contient que le site et les ressources, pas la pipeline | Réimplémenter QAB et comparer au QR binaire |
| [Face2QR, NeurIPS 2024](https://arxiv.org/html/2411.19246) | ReShuffle compatible avec l'image, renforcement des marqueurs et loss spatiale adaptative | plus de 94 % dans son protocole physique spécialisé visage | Le [dépôt annoncé](https://github.com/cavosamir/Face2QR) ne contient qu'un README | Réutiliser ReShuffle et loss adaptative, pas le modèle visage |
| [FreeQR, IEEE TMM 2026](https://doi.org/10.1109/TMM.2026.3668595) | Fusion d'un canal latent avec le latent bruité du blueprint et gradient d'erreur de scan directement dans le latent | l'article annonce une génération scannable en quelques secondes et une amélioration conjointe de l'esthétique et de la lecture | Aucun dépôt public ou checkpoint reproductible trouvé pendant cette recherche | Piste algorithmique prioritaire après QArt ; résultat à reproduire, pas à supposer |
| [BeautyMark, Applied Intelligence 2026](https://link.springer.com/article/10.1007/s10489-026-07204-2) | ControlNet spécialisé et couche de simulation de sampling différentiable ; watermark d'authenticité en plus | l'article annonce un meilleur compromis décodage/image/watermark | Aucun dépôt public trouvé | Reprendre la simulation différentiable ; watermark hors chemin critique initial |
| [QArt Rust](https://github.com/andrewyur/qart) | Implémentation MIT des QArt Codes, documentée et rapide | benchmark du dépôt : environ 285 ms sur un QR v40 | Code et crate disponibles | Premier candidat pour remplacer notre faux proxy QArt |
| [ISO/IEC 15415:2024](https://www.iso.org/standard/76876.html) | Méthode actuelle de mesure et de notation de qualité des symboles 2D | norme, pas un taux de génération | Spécification payante ; les attributs de contrôle peuvent être instrumentés | Ajouter une note de qualité, distincte du simple décodage |
| [ZXing-C++](https://github.com/zxing-cpp/zxing-cpp) | Décodeur QR actif, bindings Python et choix du binariseur | bibliothèque, pas un benchmark Prooftag | Code Apache-2.0, paquet Python disponible | Ajouter au minimum aux décodeurs OpenCV et ZBar |

### Attention aux pourcentages publiés

Les taux ci-dessus ne sont pas directement comparables :

- DiffQRCoder mesure 100 images dans un protocole `qr-verify` ;
- Text2QR et Face2QR mesurent des tentatives physiques sur un petit ensemble, avec un délai de
  décodage de trois secondes ;
- Face2QR est spécialisé dans les visages ;
- notre porte E013 exige 26 validations simultanées par image ;
- aucun de ces travaux ne démontre 99 % sur notre distribution de prompts, nos URLs Prooftag, nos
  dégradations et nos téléphones.

Ils indiquent une direction technique. Ils ne permettent pas d'annoncer un taux Prooftag avant une
campagne appariée.

## Les avancées réellement utiles

### 1. Optimiser les bits avant de corriger les pixels

Text2QR et Face2QR ont le même enseignement central : il faut d'abord choisir une représentation QR
qui ressemble à l'image. Face2QR part d'une sortie contenant plus de 43 % d'erreurs et indique que
son ReShuffle réduit ce taux de plus de moitié avant le raffinement latent.

DiffQRCoder décrit lui aussi QArt entre Stage 1 et Stage 2. Notre E013 remplace cette cible par le QR
binaire original. Le modèle doit donc choisir entre :

- respecter des carrés noirs et blancs qui détruisent l'image ;
- respecter le prompt et perdre la lecture.

Un QArt/QAB correct déplace ce compromis en amont. Le QR fourni à Stage 2 ressemble déjà à la
luminance de Stage 1 tout en conservant un code valide.

L'implémentation Rust trouvée utilise notamment un fragment ajouté aux URLs pour créer des degrés de
liberté. Le navigateur n'envoie pas le fragment au serveur, mais le texte décodé n'est plus
strictement identique à l'URL sans fragment. Deux modes devront donc être séparés :

- `canonical_url` : le fragment est autorisé, la destination et la signature Prooftag avant `#`
  doivent être identiques ;
- `exact_payload` : aucun caractère supplémentaire n'est autorisé ; il faudra alors implémenter un
  solveur matriciel limité aux degrés de liberté qui préservent exactement le payload.

La politique de sécurité Prooftag doit décider lequel de ces contrats est acceptable. Aucun
assouplissement silencieux de la validation n'est autorisé.

### 2. Utiliser un blueprint adaptatif, pas un QR uniforme

Le QAB de Text2QR :

1. polarise l'histogramme de la référence esthétique ;
2. réorganise le QR sans changer le message ;
3. adapte, module par module, la taille du carré central nécessaire ;
4. remet explicitement les motifs finder et alignment.

Ce procédé est directement pertinent pour notre défaut visuel. Les modules dont la référence est
déjà suffisamment sombre ou claire ont besoin de très peu de signal QR. Seuls les modules ambigus
reçoivent un centre plus fort.

### 3. Fusionner le signal dans le latent

FreeQR propose de fusionner un canal précis du latent courant avec le canal correspondant du
blueprint bruité au même timestep. Le QR n'est donc pas repeint après génération : sa distribution
de luminance participe au débruitage.

L'ablation Prooftag devra mesurer :

- chacun des quatre canaux latents ;
- l'intervalle de timesteps où la fusion est active ;
- le coefficient de fusion ;
- fusion fixe contre fusion commandée par la marge de lecture du module ;
- fusion avec QR binaire contre QArt/QAB ;
- coût, CLIP-aesthetic, CLIPScore, LPIPS/DINO et lecture.

Cette méthode est plus prometteuse que la projection finale testée en E004, car l'image peut
réinterpréter le signal QR pendant la diffusion.

### 4. Remplacer la MER centrale par une loss plus proche d'un scanner

E013 a produit 39 images avec une MER centrale nulle, mais une seule à 26/26. Les nouveaux travaux
renforcent l'idée qu'une loss uniforme ne suffit pas :

- marqueurs fonctionnels explicitement renforcés ;
- noyau de sampling variable selon les régions ;
- simulation de lecture dans BeautyMark ;
- erreur évaluée après perturbations de capture.

La nouvelle loss doit combiner :

```text
L = L_codeword
  + λ_marker * L_finder_alignment_timing
  + λ_quiet * L_quiet_zone
  + λ_iso * L_quality_proxy
  + λ_camera * E[L_decode_proxy(augmentation(image))]
  + λ_content * L_DINO_or_LPIPS
  + λ_aesthetic * L_aesthetic
```

`L_codeword` doit connaître les blocs Reed–Solomon et leur budget d'erreurs. Deux images ayant le
même nombre de modules faux ne présentent pas nécessairement le même risque si les erreurs sont
concentrées dans un même bloc.

La loss reste un professeur de gradient, jamais la preuve de livraison. Les vrais décodeurs restent
la porte finale.

### 5. Moderniser l'image de référence sans porter immédiatement DiffQRCoder

Il est possible d'utiliser un meilleur modèle moderne sans disposer d'un ControlNet QR compatible :
le modèle moderne génère uniquement la référence esthétique en Stage 0.

Deux candidats réalistes :

- [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), 1 à 4 étapes, avec
  CPU offload ; utile pour obtenir vite une composition fidèle au prompt ;
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), plus simple à intégrer
  et plus léger que FLUX pour un usage local.

La référence est ensuite transformée en QArt/QAB, puis la Stage 2 reste initialement sur SD 1.5.
Cela évite de réécrire SRPG pour une autre architecture avant d'avoir prouvé l'apport du blueprint.

En revanche, remplacer directement le générateur QR principal est prématuré :

- le [Control-LoRA SDXL brightness](https://huggingface.co/Oysiyl/controlnet-lora-brightness-sdxl/blob/main/README.md)
  indique explicitement que ses QR actuels ne sont pas scannables ;
- le [Control-LoRA FLUX brightness](https://huggingface.co/Oysiyl/controlnet-lora-brightness-flux/blob/main/README.md)
  annonce environ 45 Go de VRAM pour son entraînement et ne fournit pas de métrique de lecture ;
- le [ControlNet QR FLUX de lucataco](https://huggingface.co/lucataco/flux-dev-controlnet-qr-code)
  n'a ni exemple d'usage complet, ni description du dataset, ni évaluation ;
- le [LoRA QR-Verse](https://huggingface.co/Qrverse/qrart-lora) est intéressant pour une ablation
  de style, mais ses jeux annoncés ne contiennent que 182 images SD 1.5 et 67 images SDXL, sans
  protocole de lecture publié.

## Architecture recommandée

```text
prompt + payload Prooftag court
          |
          +--> Stage 0 esthétique rapide
          |    FLUX.1-schnell ou SDXL
          |              |
          |              v
          +--> encodeur exact QR + sélection masque/version/ECC
                         |
                         v
               QArt / QAB Reed–Solomon
               + preuve de décodage
                         |
                         v
               DiffQRCoder SD 1.5 Stage 2
               + fusion latente FreeQR
               + loss spatiale adaptative
                         |
                         v
               candidats et arrêt anticipé
                         |
                         v
          ZXing-C++ + OpenCV + ZBar + WeChat
          transformations caméra + note ISO 15415
                         |
               accepté / nouvel essai / rejet
```

L'image brute, le blueprint, chaque étape latente, les erreurs par codeword, les résultats de tous
les décodeurs, les métriques d'image, la durée et la VRAM doivent être archivés.

## Plan expérimental recommandé

### E014A — QArt réel avant toute nouvelle recherche

But : mesurer le gain de la cible, sans changer le modèle.

Comparer, à seeds et prompts identiques :

1. QR binaire actuel ;
2. QArt Rust en mode URL canonique ;
3. QArt exact-payload si le solveur est disponible ;
4. QArt + adaptive-halftone de Text2QR.

Conditions préalables :

- chaque blueprint doit être décodable avant la diffusion ;
- la destination et la signature Prooftag doivent être vérifiées ;
- seules les géométries dont le QR témoin passe la porte sont admises ;
- commencer par 4 prompts × 4 seeds × 3 payloads, puis promouvoir sur 100 cas.

Porte : ne conserver une cible que si elle améliore la lecture originale et réduit la visibilité de
la grille sans dégrader CLIP-aesthetic de plus de 5 % face à la meilleure baseline valide.

### E014B — Fusion latente FreeQR

But : vérifier si le QR peut rester implicite sans projection finale visible.

Recherche courte et factorielle :

- canal latent 0, 1, 2 ou 3 ;
- trois fenêtres de timesteps ;
- trois coefficients de fusion ;
- QR binaire contre meilleur blueprint E014A ;
- sans puis avec gradient adaptatif.

La recherche doit utiliser un seul prompt de mise au point, puis confirmer les paramètres figés sur
des prompts, seeds et payloads jamais vus. Aucun Optuna à 28 dimensions à ce stade.

### E015 — Meilleure référence esthétique

Comparer SD 1.5, SDXL et FLUX.1-schnell uniquement en Stage 0. Stage 2, blueprint et validations
restent identiques. Le modèle moderne est promu seulement si l'amélioration esthétique ne réduit
pas le rendement de candidats validés.

### E016 — Simulateur différentiable et modèle Prooftag

Quand E014 a créé suffisamment de couples image/échec :

1. entraîner un petit surrogate de détection et de sampling sur les résultats de ZXing, OpenCV,
   ZBar, WeChat et captures physiques ;
2. appliquer blur, perspective, downscale, bruit, exposition, contraste, compression, moiré,
   gain/perte de point et balance des blancs pendant l'entraînement ;
3. distiller cette fonction dans un ControlLoRA ou un correcteur latent Prooftag ;
4. conserver les vrais décodeurs comme contrôle indépendant.

Le mini-modèle de paramètres demandé précédemment ne doit être entraîné qu'après plusieurs
centaines de succès, répartis sur des prompts et payloads différents. Sa bonne fonction est de
classer les prochaines recettes par probabilité de franchir la porte, pas de remplacer la porte.

## Validation et définition du « presque 100 % »

Trois nombres doivent être publiés séparément :

1. `generation_yield` : pourcentage des générations brutes qui passent ;
2. `budget_success` : pourcentage des demandes pour lesquelles au moins un candidat passe dans un
   budget fixé ;
3. `delivered_validity` : pourcentage des images livrées qui ont passé la porte.

Le troisième peut être 100 % dès maintenant grâce au rejet. Le premier et le deuxième doivent être
mesurés, pas extrapolés à partir d'un succès isolé.

La porte actualisée doit inclure :

- payload exact ou contrat URL canonique explicitement choisi ;
- ZXing-C++, OpenCV, ZBar et WeChat ;
- détection et décodage séparés dans les métriques ;
- scénarios écran et impression ;
- attributs de qualité inspirés d'ISO/IEC 15415:2024 ;
- test physique multi-téléphone avant promotion production.

Une revendication de 99 % nécessitera au minimum plusieurs centaines de cas jamais vus, un intervalle
de confiance, les échecs conservés, et une définition précise du délai de scan et des appareils.

## Compatibilité RTX 4000 Ada 20 Go

| Élément | Faisabilité |
|---|---|
| QArt/QAB | CPU, coût négligeable face à la diffusion |
| DiffQRCoder SD 1.5 + gradients | déjà démontré sur la carte |
| Fusion latente FreeQR sur SD 1.5 | compatible ; surcoût attendu faible hors guidance |
| SDXL comme Stage 0 | compatible en FP16 |
| FLUX.1-schnell comme Stage 0 | compatible avec offload/quantification, mais plus lent en chargement |
| SDXL avec correction latente complète | possible avec checkpointing/offload, à tester après E014 |
| FLUX 12B avec ControlNet et gradients | non prioritaire sur 20 Go ; la carte du modèle comparable annonce environ 45 Go à l'entraînement |

Le meilleur usage de la carte est donc d'améliorer d'abord l'encodage et la guidance SD 1.5, puis
d'utiliser un modèle moderne uniquement pour la composition esthétique.

## Décision

À faire maintenant :

1. intégrer et auditer un vrai QArt ;
2. implémenter le QAB adaptatif ;
3. ajouter ZXing-C++ et séparer détection/décodage ;
4. réaliser E014A ;
5. seulement si E014A gagne, implémenter la fusion latente FreeQR en E014B.

À ne pas faire maintenant :

- relancer une recherche Optuna large ;
- entraîner CatBoost avec un seul positif ;
- porter toute la pipeline vers SD 2.1, SDXL ou FLUX ;
- fine-tuner un ControlNet sur des cibles réparées et quadrillées ;
- annoncer le taux d'un article comme taux Prooftag.

L'évolution décisive n'est pas un checkpoint plus récent. C'est de rendre l'encodage QR compatible
avec l'image avant la diffusion, puis de guider la diffusion avec un signal qui se rapproche du
comportement des scanners réels.
