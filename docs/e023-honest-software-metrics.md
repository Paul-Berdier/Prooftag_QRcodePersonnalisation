# E023 — métriques logicielles honnêtes sans banc caméra

## Décision

E023 répète exactement la matrice E022 : dix prompts, une seed et les deux
recettes `diffqrcoder_srpg` et `diffqrcoder_paper_srpg`. La génération n'est pas
modifiée. Seule la mesure change, afin de comparer les nouvelles colonnes avec
les anciens résultats sans confondre un changement de modèle et un changement
de métrique.

Le téléphone n'étant pas automatisable pour l'instant, aucune colonne E023
n'est appelée « probabilité téléphone ».

## Nouvelles mesures

| Champ | Sens exact | Usage |
|---|---|---|
| `synthetic_robustness_normalized_pass_rate` | matrice décodeur × transformation normalisée par le QR témoin | diagnostic logiciel |
| `synthetic_original_decoder_pass_rate` | consensus des décodeurs sur le PNG original | porte logicielle conservatrice |
| `wechat_qrcode_original_exact` | payload exact du détecteur CNN + super-résolution WeChat | signal logiciel principal |
| `software_preprocessing_proxy_normalized_pass_rate` | anciens prétraitements « phone proxy » | diagnostic seulement |
| `hpsv2_1` | préférence image-prompt HPS v2.1 | comparaison esthétique à prompt identique |
| `clip_aesthetic` | ancien prédicteur LAION ViT-B/32 | historique seulement |
| `clip_similarity` | cosinus image-prompt CLIP | diagnostic d'alignement historique |

Le champ API historique `scan_pass_rate` reste un alias de l'indice synthétique
pour ne pas invalider les campagnes, CSV et notebooks existants.

## Bibliothèques et versions

- OpenCV contrib fournit WeChatQRCode, composé d'un détecteur CNN et d'un modèle
  de super-résolution : <https://docs.opencv.org/master/d5/d04/classcv_1_1wechat__qrcode_1_1WeChatQRCode.html>.
- Les quatre poids officiels WeChat sont épinglés à la révision
  `3487ef7cde71d93c6a01bb0b84aa0f22c6128f6b` et leur dépôt publie leurs MD5 :
  <https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode>.
- HPS v2.1 est appelé depuis la révision officielle
  `866735ecaae999fa714bd9edfa05aa2672669ee3`; son score n'est comparable qu'entre
  images issues du même prompt : <https://github.com/tgxs002/HPSv2>. Le wheel
  PyPI 1.2.0 n'est volontairement pas utilisé car il omet le vocabulaire BPE.
  HPS est forcé sur CPU pour laisser la RTX exclusivement à DiffQRCoder.

## Limites

WeChatQRCode reste un décodeur serveur. HPS ne mesure pas la scannabilité. E023
améliore donc les diagnostics et évite les libellés faux, mais ne crée pas une
vérité terrain artificielle. Les anciennes notes téléphone E022 restent le seul
jeu physique disponible pour contrôler a posteriori les corrélations.

## Campagne identique à E022

Après déploiement, ouvrir le tunnel puis lancer :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
.\scripts\lab-remote.ps1
```

Dans un second PowerShell :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
python .\scripts\e023-honest-software-metrics.py `
  --payload "https://ptag.io/t/e022" `
  --family all `
  --seeds 61001 `
  --launch
```

Le payload E022 est volontairement conservé : changer l'URL pourrait changer
la matrice QR et invalider l'appariement avec les vingt images précédentes.

Le laboratoire affiche HPS v2.1, le résultat WeChat sur l'original et les
indices logiciels avec leur nature explicite. Les notes humaines peuvent être
remplies comme précédemment puis exportées en CSV.

## Critère de lecture provisoire

Sans caméra automatisée, il n'existe pas de taux fiable à 99 %. Pour sélectionner
les images à soumettre aux rares vérifications manuelles, l'ordre provisoire est :

1. payload exact WeChat sur l'original ;
2. consensus original des décodeurs ;
3. absence de divergence, saturation et écrêtage ;
4. indice synthétique, uniquement comme départage ;
5. HPS v2.1 puis notes esthétiques humaines.

Cette règle privilégie les faux rejets aux faux succès, conformément à la priorité
de scannabilité du projet.
