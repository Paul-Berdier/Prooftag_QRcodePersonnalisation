# E004 - seconde diffusion guidée et localisée

## Objectif

Réinjecter les corrections QR dans un processus génératif afin que le modèle les transforme en
formes compatibles avec le style, au lieu de terminer par des points noirs ou blancs ajoutés sur
l'illustration.

## Chaîne évaluée

```text
raw
  -> construction d'un guide sur les modules incorrects
  -> nouvelle injection de bruit img2img
  -> 8 étapes SD 1.5 + ControlNet QR
  -> projection dans un masque local dilaté et adouci
  -> guided
  -> SR-MPGD avec porte MAE
  -> guided_latent_srl
  -> réparations ciblées guided_latent_*
  -> chaîne brute complète en secours
```

Le guide verrouille les motifs fonctionnels et ne corrige les modules de données que lorsqu'ils
sont incorrects ou insuffisamment contrastés. Le masque est dérivé de la différence entre le
guide et l'image brute, dilaté de quatre pixels puis adouci sur quatre pixels.

## Paramètres initiaux

| Paramètre | Valeur E004a |
|---|---:|
| Étapes effectives seconde diffusion | 8 |
| Timesteps planifiés à strength 0,30 | 27 |
| Strength img2img | 0,30 |
| Poids ControlNet | 1,75 |
| Centre du guide | 0,45 |
| Marge d'incertitude | 16 |
| Dilatation du masque | 4 px |
| Adoucissement du masque | 4 px |
| Porte MAE seconde diffusion | 0,12 |
| Itérations SR-MPGD | 8 |
| Porte MAE SR-MPGD | 0,08 |

Ces paramètres sont des hypothèses de départ. Ils ne doivent pas être activés en production avant
la campagne GPU.

Dans une pipeline img2img, `strength` réduit le nombre réel de débruitages. Le code planifie donc
`ceil(8 / 0,30) = 27` timesteps afin que Diffusers en exécute environ huit, au lieu des deux étapes
qu'aurait produites naïvement `num_inference_steps=8` avec `strength=0,30`.

## Artefacts obligatoires

- `raw.png` : sortie de la première diffusion ;
- `attempt_1_guided_control.png` : image de contrôle contenant les corrections ;
- `attempt_1_guided_mask.png` : zone dans laquelle la seconde diffusion peut remplacer le brut ;
- `attempt_1_guided_unprojected.png` : sortie globale brute de la seconde diffusion ;
- `attempt_1_guided_projected.png` : sortie reprojetée rejetée par la porte QR/MAE ;
- `attempt_1_guided.png` : résultat localisé de la seconde diffusion ;
- `attempt_1_guided_latent_srl.png` : résultat après SR-MPGD, lorsqu'il franchit sa porte ;
- `final.png` : variante valide ayant le plus faible écart visuel mesuré ;
- `refinements.csv` : paramètres et diagnostics internes des deux étages.

## Protocole serveur

Après construction, import et déploiement de l'image du commit courant :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
make pause-vllm
make benchmark-e004
```

Le script exécute d'abord une baseline avec les deux raffinements désactivés, puis E004 avec la
seconde diffusion et SR-MPGD activés. Un trap retire les surcharges Kubernetes à la sortie. Vérifier
ensuite que `/v1/runtime` rapporte de nouveau les deux options à `false`.

Depuis PowerShell, lorsque le service QR possède déjà le GPU, la même campagne et le rapatriement
des deux rapports s'exécutent en une commande :

```powershell
.\scripts\benchmark-remote.ps1 -E004
```

## Portes de décision

E004a est retenue pour ablation seulement si :

1. les six images finales passent encore 26/26 validations ;
2. au moins une variante sélectionnée porte le préfixe `guided_` ;
3. la réparation guidée sélectionnée modifie moins l'image que son équivalent issu du brut ;
4. aucun artefact ne dépasse sa porte MAE ;
5. aucun payload erroné n'est observé ;
6. l'inspection des six images, en particulier `dense-payload`, ne montre ni halo ni raccord de
   masque visible.

Si aucune variante guidée n'est sélectionnée mais que `guided` améliore l'erreur module, E004b
fera varier une seule dimension à la fois : strength `0,20/0,30/0,40`, poids ControlNet
`1,50/1,75/2,00`, puis taille du centre `0,35/0,45/0,55`. La campagne de 100 images reste bloquée
jusqu'à ce qu'une configuration franchisse les portes sur les six cas sentinelles.

## Différence avec DiffQRCoder

L'article calcule un gradient SRPG à travers la prédiction de bruit de l'UNet à chaque timestep et
utilise Qart pour adapter le motif cible. E004a emploie le guide local comme condition ControlNet
pendant la seconde diffusion, puis SR-MPGD après diffusion. C'est une adaptation exécutable avec
la pile actuelle et la RTX 4000 Ada, mais pas encore une reproduction mathématique complète de
SRPG. Cette limite est volontairement documentée afin de ne pas attribuer au projet les résultats
de l'article avant validation locale.

## Résultat mesuré — 21 juillet 2026

Campagnes `20260721T074704Z-308953d2` (contrôle) et
`20260721T074752Z-308953d2` (E004), mêmes six images brutes bit pour bit :

| Mesure moyenne | Contrôle | E004 | Décision |
|---|---:|---:|---|
| Lecture brute | 0/26 | 0/26 | aucun progrès |
| Lecture guidée / latent | — | 0/26 | aucun progrès |
| Erreur module brute | 13,152 % | 13,152 % | identique |
| Erreur du guide | — | 5,825 % | le guide porte bien le signal |
| Erreur après rediffusion | — | 15,072 % | régression |
| Couverture moyenne du masque | — | 68,16 % | pas réellement local |
| Pixels modifiés du final | 58,75 % | 74,00 % | +25,95 % relatif |
| Temps total | 5,18 s | 9,62 s | +85,7 % |
| Pic VRAM observé | 3 860 MiB | 6 192 MiB | +2 332 MiB |

Les six livraisons atteignent encore 26/26 uniquement grâce aux réparations déterministes. Le
garde-fou E004 acceptait la rediffusion sur la seule MAE, sans exiger une amélioration QR réelle.
De plus, `guided_candidate` et `guided` étaient dupliqués bit pour bit dans les six cas, ce qui
donnait artificiellement l'impression de deux étapes différentes dans le rapport.

**Décision : E004 est rejetée.** L'ancien mode reste désactivé. Le code postérieur renomme
l'artefact rejeté `guided_projected`, supprime la duplication et exige une amélioration module
réelle. La suite est E005, documentée dans `docs/e005-srpg.md`.
