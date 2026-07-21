# Programme de recherche Prooftag QR

## Objectif mesurable

Le projet doit produire des QR codes visuellement intégrés à une illustration, rapidement,
sans jamais livrer un payload erroné. Deux performances distinctes sont suivies :

1. **robustesse intrinsèque** : proportion des images brutes ou raffinées qui passent sans
   réparation binaire ;
2. **fiabilité de livraison** : proportion des images publiées ayant passé toutes les
   validations exactes.

La fiabilité de livraison visée est 100 %. Cela signifie qu'une image refusée n'est pas
publiée ; cela ne signifie pas que tout essai du modèle sera lisible dans toute condition
physique imaginable.

## Décision d'architecture

La voie retenue est un modèle Stable Diffusion 1.5 spécialisé, et non un entraînement complet
depuis zéro :

```text
QR + prompt
  -> ControlNet QR Prooftag
  -> SRL pendant ou après la diffusion
  -> réparations ciblées sur la sortie SRL
  -> validation exacte multi-décodeurs et multi-dégradations
  -> secours issu du brut uniquement si nécessaire
  -> publication ou rejet
```

Le ControlNet est responsable de la structure QR. Un LoRA ultérieur sera responsable du style
Prooftag. Cette séparation évite qu'une amélioration esthétique soit interprétée à tort comme
une amélioration de lecture.

## Étapes et portes de décision

### R0 — Baseline figée

- conserver le benchmark historique et le commit associé ;
- exécuter le protocole 2.0 sur le commit courant ;
- mesurer séparément `raw`, `latent_srl`, profils arrondis et secours binaires ;
- ne comparer que des campagnes ayant le même hash de cas et les mêmes paramètres.

**Sortie attendue :** au moins 100 générations avant une décision de modèle. Les six cas fixes
restent un test de non-régression rapide, pas une estimation de fiabilité.

### R1 — Professeur sans entraînement

La loss SRL suit le principe de DiffQRCoder : erreur de luminance par module, pondération
gaussienne, priorité au sous-module central et arrêt local quand le centre est correct. Le
raffinement `latent_srl` optimise le latent VAE, conserve le meilleur intermédiaire et rend
l'image originale si le taux d'erreur ne baisse pas.

Le mode est volontairement désactivé par défaut :

```bash
PROOFTAG_QR_LATENT_REFINEMENT_ENABLED=true
PROOFTAG_QR_LATENT_REFINEMENT_ITERATIONS=8
PROOFTAG_QR_LATENT_REFINEMENT_LEARNING_RATE=0.02
PROOFTAG_QR_LATENT_REFINEMENT_QR_WEIGHT=1.0
PROOFTAG_QR_LATENT_REFINEMENT_PRESERVATION_WEIGHT=1.0
PROOFTAG_QR_LATENT_REFINEMENT_FUNCTIONAL_WEIGHT=4.0
PROOFTAG_QR_LATENT_REFINEMENT_MAX_LATENT_DELTA=0.10
PROOFTAG_QR_LATENT_REFINEMENT_MAX_MEAN_ABSOLUTE_CHANGE=0.08
```

L'expérience E002 a invalidé la normalisation globale du gradient et les anciens paramètres
`learning_rate=0.20`, `preservation_weight=0.15`. La v2 masque les mises à jour sur les modules
incorrects, borne le déplacement du latent autour de son encodage et refuse toute variante
dont le changement absolu moyen dépasse 0,08. Ces paramètres restent expérimentaux et le mode
reste désactivé par défaut.

La préservation actuelle est une distance L1 multi-échelle. Elle est légère et reproductible,
mais ce n'est pas LPIPS. LPIPS ne sera ajouté qu'avec une comparaison montrant un bénéfice
esthétique qui compense son coût GPU.

E003 confirme que ces garde-fous préservent l'image (MAE moyenne 0,0279 et netteté moyenne
-1,19 %), mais le réglage v2 ne gagne aucun des 156 scénarios de lecture et ne réduit l'erreur
module que de 4,63 % en moyenne. Une ablation courte E004 doit trouver un point intermédiaire
avant toute génération massive de dataset.

E003 a aussi montré que les réparations ciblées repartaient du brut après le latent. La chaîne
R1 impose désormais l'ordre `brut -> seconde diffusion guidée -> SR-MPGD -> réparation ciblée
guided_latent_* -> secours brut`. Une
expérience n'est pas valide si elle compare seulement `final.png` sans vérifier le nom de la
variante sélectionnée et l'image intermédiaire correspondante.

La première implémentation E004 utilise l'image brute comme source img2img, une carte locale de
modules incorrects comme condition ControlNet, huit étapes à `strength=0.30`, puis projette le
résultat dans un masque dilaté et adouci. SR-MPGD travaille ensuite sur cette sortie. Cette
version reproduit la structure en deux étapes de DiffQRCoder, mais n'implémente pas encore le
gradient SRPG exact à travers l'UNet à chaque timestep ; cette différence doit rester explicite
dans les comparaisons avec l'article.

**Porte R1 :** conserver les paramètres uniquement si `latent_srl` réduit l'erreur des modules,
améliore le taux de lecture et ne dégrade pas de plus de 5 % les mesures perceptuelles par
rapport à l'image brute.

### R2 — Générateur de dataset

Le pilote comportera 2 000 exemples d'entraînement, 500 de validation et 1 000 de test. Les
splits sont faits par payload, famille de prompt et version QR afin d'empêcher une fuite entre
entraînement et test.

Chaque exemple doit contenir :

- identifiant, version de schéma et licence/provenance ;
- hash du payload, QR source, niveau ECC, version et masque ;
- prompt, prompt négatif, seed et paramètres de diffusion ;
- image brute et image raffinée dans le latent ;
- carte des modules incorrects et motifs fonctionnels ;
- résultats complets des décodeurs et dégradations ;
- métriques d'image, durées et consommation GPU ;
- décision automatique et, pour l'échantillon contrôlé, note humaine.

Les images issues d'une réparation binaire globale sont exclues des cibles d'entraînement.
Elles peuvent être conservées comme exemples négatifs.

### R3 — ControlNet Prooftag

Initialisation depuis le ControlNet QR actuel, SD 1.5 gelé, batch 1, accumulation de gradients,
précision mixte, gradient checkpointing et Adam 8 bits. La loss prévue combine :

```text
L = L_diffusion
  + lambda_qr * L_SRL
  + lambda_functional * L_functional
  + lambda_preservation * L_perceptual
  + lambda_camera * L_after_simulated_capture
```

La RTX 4000 Ada 20 Go est compatible avec cette configuration. vLLM doit être arrêté et la
VRAM libre vérifiée avant tout entraînement.

**Porte R3 :** sur le test jamais vu, au moins 95 % des images doivent passer tous les tests
logiciels sans réparation binaire, puis au moins 99,5 % après raffinement latent.

### R4 — Style Prooftag

Entraîner un ou plusieurs LoRA de style seulement après R3. Le taux de lecture ne doit pas
baisser de plus d'un point absolu. Toute famille de style qui dépasse cette limite est rejetée
ou réentraînée.

### R5 — Terrain et accélération

Campagne physique : plusieurs iPhone, Pixel et Android intermédiaires, caméra native et
lecteurs industriels, écran et impressions, différentes tailles, matières, lumières,
distances et inclinaisons. Objectif initial : au moins 97 % de succès dans le protocole
physique documenté et 100 % des images livrées prévalidées.

LCM/LCM-LoRA ou une distillation 8–12 étapes ne sera testée qu'après validation du modèle
robuste. Une accélération perdant plus de deux points de lecture est refusée.

## Protocole de comparaison

- même commit déployé que celui inscrit dans le rapport ;
- dépôt Git propre ou modifications explicitement archivées ;
- mêmes seeds, payloads, prompts et paramètres ;
- modèle, image Docker, versions CUDA/PyTorch/Diffusers et GPU enregistrés ;
- démarrage à froid distingué des générations chaudes ;
- moyenne, médiane, P95 et intervalles de confiance sur les grandes campagnes ;
- résultats par QR version, ECC, style, décodeur et scénario ;
- aucune suppression des échecs du rapport.

## Références méthodologiques

- [DiffQRCoder, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Liao_DiffQRCoder_Diffusion-Based_Aesthetic_QR_Code_Generation_with_Scanning_Robustness_Guided_WACV_2025_paper.html)
- [Implémentation officielle DiffQRCoder](https://github.com/jwliao1209/DiffQRCoder)
- [Text2QR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Text2QR_Harmonizing_Aesthetic_Customization_and_Scanning_Robustness_for_Text-Guided_QR_CVPR_2024_paper.html)
- [ControlNet, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)
- [Exemple officiel d'entraînement ControlNet](https://github.com/huggingface/diffusers/tree/main/examples/controlnet)
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378)
