# E008 — comparaison des ControlNet QR

## Conclusion de la recherche documentaire

Il n'existe pas de benchmark public, apparié et reproductible permettant d'affirmer que
`Nacholmo/controlnet-qr-pattern-v2` est meilleur que QR Code Monster v2 ou que le modèle Dion.
Les cartes de modèles montrent des exemples et donnent des conseils, mais ne publient ni taux de
scan multi-décodeurs, ni distribution de prompts, ni comparaison sur les mêmes seeds.

La décision correcte est donc un bake-off local avant E007, avec les mêmes entrées, la même base
Stable Diffusion, le même SRPG et les mêmes 26 validations.

Sources primaires vérifiées le 21 juillet 2026 :

- [QR Code Monster SD1.5](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster)
  et son [sous-dossier v2](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster/tree/main/v2) ;
- [Nacholmo QR Pattern v2](https://huggingface.co/Nacholmo/controlnet-qr-pattern-v2) ;
- [DionTimmer QR ControlNet](https://huggingface.co/DionTimmer/controlnet_qrcode-control_v1p_sd15) ;
- [code officiel DiffQRCoder](https://github.com/jwliao1209/DiffQRCoder) et
  [article WACV 2025](https://arxiv.org/abs/2409.06355).

## Candidats retenus

| Profil | Poids réellement chargés | Condition | Justification |
|---|---|---|---|
| `dion_sd15` | `DionTimmer/controlnet_qrcode-control_v1p_sd15` | binaire | baseline Prooftag actuelle |
| `monster_sd15_v1` | `monster-labs/control_v1p_sd15_qrcode_monster` racine | binaire | référence historique et nom par défaut dans DiffQRCoder |
| `monster_sd15_v2` | même dépôt, sous-dossier `v2` | quiet zone grise | la carte annonce une amélioration v2 en scan et créativité |
| `nacholmo_sd15_v2` | `Nacholmo/controlnet-qr-pattern-v2` | binaire | entraînement qui conditionne les 25 % plus noirs et 25 % plus blancs |

Le sous-dossier Monster `v2` est essentiel. Appeler seulement
`ControlNetModel.from_pretrained("monster-labs/control_v1p_sd15_qrcode_monster")` charge la racine,
pas automatiquement `v2/`. Le chargeur Prooftag accepte maintenant explicitement
`controlnet_model_subfolder`.

## Ce que les sources permettent réellement d'affirmer

- DiffQRCoder utilise par défaut un checkpoint local nommé `control_v1p_sd15_qrcode_monster` et
  rapporte le passage de 60 % pour ControlNet seul à 99 % pour son pipeline complet. Ce résultat
  prouve surtout l'apport du raffinement, pas la supériorité universelle du ControlNet.
- QR Code Monster v2 est présenté par son auteur comme une forte amélioration de v1, mais la carte
  précise que toutes les générations ne sont pas lisibles et recommande plusieurs essais.
- Nacholmo v2 est plus récent et son conditionnement extrême est intéressant pour dissimuler la
  grille, mais sa carte ne donne aucun SSR quantitatif.
- La popularité ou le nombre de téléchargements n'est pas une métrique de scannabilité.

En l'absence de résultats locaux, l'a priori est donc : Monster v2 comme candidat principal,
Nacholmo v2 comme challenger sérieux, Dion comme témoin connu. Ce n'est pas encore un classement
mesuré et aucun taux de réussite n'est attribué à ces trois poids.

## Avancées qui ne sont pas des ControlNet interchangeables

La recherche a aussi couvert les méthodes plus récentes. Elles valident la direction du projet,
mais ne fournissent pas un checkpoint SD1.5 que l'on pourrait simplement substituer :

- [Text2QR (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Text2QR_Harmonizing_Aesthetic_Customization_and_Scanning_Robustness_for_Text-Guided_QR_CVPR_2024_paper.html)
  apprend une génération et une correction propres à son architecture ;
- [GladCoder (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/861) optimise conjointement
  esthétique et robustesse ;
- [DiffQRCoder (WACV 2025)](https://arxiv.org/abs/2409.06355) reste l'avancée directement
  applicable ici, car son raffinement est sans entraînement et accepte un ControlNet préentraîné ;
- [Face2QR](https://arxiv.org/abs/2411.19246) cible spécifiquement la préservation d'identité des
  visages ;
- [Claycode](https://arxiv.org/abs/2505.08666) remplace le QR standard par un autre code 2D : il
  n'est donc pas compatible avec l'exigence Prooftag de lecture par les scanners QR existants.

La suite réaliste reste : meilleur ControlNet public mesuré, SRPG, optimiseur contextuel E007,
puis fine-tuning Prooftag si les données locales montrent encore un plafond.

## Pourquoi SDXL est différé

Deux candidats SDXL sont enregistrés pour une expérience séparée :

- `monster-labs/control_v1p_sdxl_qrcode_monster` contient environ un milliard de paramètres et sa
  carte avertit encore que toutes les sorties ne sont pas lisibles ;
- `Nacholmo/controlnet-qr-pattern-sdxl` est décrit comme work in progress, avec lecture iPhone mais
  compatibilité Android/Google Lens encore visée par l'auteur.

Notre SRPG appelle directement les composants et embeddings d'une pipeline SD1.5. Passer un
ControlNet SDXL dans cette pipeline est invalide. Une vraie branche SDXL demanderait une pipeline,
un encodage de prompt et des tests VRAM spécifiques. Sur 20 Gio, l'inférence SDXL peut être testée
avec offload, mais les gradients SRPG à chaque pas sont beaucoup plus risqués. Mélanger SD1.5 et
SDXL dans E008 rendrait le résultat inexploitable.

## Protocole E008

Le notebook `05_controlnet_model_bakeoff.ipynb` exécute :

- quatre ControlNet SD1.5 ;
- quatre échelles communes : 0,90, 1,10, 1,35 et 1,60 ;
- douze contextes appariés issus du plan E007 ;
- Stage-1 fixe à 16 pas ;
- SRPG fixe à 100 pas ;
- 26 validations sur le brut et 26 sur la sortie SRPG ;
- CLIP-aesthetic, CLIPScore, erreur module, durée et VRAM.

Cela représente 192 exécutions complètes. Un seul pipeline réside en VRAM et chaque modèle possède
son dossier reprenable. Un modèle manquant ou en erreur rend son groupe incomplet.

## Ordre de décision

```text
1. campagne complète sur 12 contextes
2. 26/26 sur tous les contextes après SRPG
3. pire taux de lecture
4. taux de lecture moyen
5. CLIP-aesthetic
6. CLIPScore
7. durée
```

Le brut ControlNet est mesuré séparément. Cela permet de distinguer un meilleur modèle d'un résultat
qui aurait été sauvé uniquement par SRPG. Sans profil strict complet, `decision.json` contient
`NO_PROMOTION`; le meilleur résultat reste seulement `best_observed`.

Avec un profil strict complet, le statut est `AUTOMATIC_CANDIDATE`, pas « production ». Le notebook
recrée `physical-validation-template.csv` pour les trois meilleurs profils et initialise
`physical-validation.csv` sans jamais écraser les saisies existantes. Le candidat sert ensuite de
base à E007 ; seuls les holdouts et scans physiques multi-téléphones/impressions autoriseront une
livraison.

## Exécution

Après reconstruction de l'image notebook :

```powershell
.\scripts\notebook-remote.ps1 -Notebook 05_controlnet_model_bakeoff.ipynb
```

Puis `Run > Run All Cells`. L'archive est créée dans
`/workspace/results/e008-controlnet-bakeoff-v1.tar.gz`.

Après E008, le ControlNet promu devient une constante de la campagne E007. E007 optimise ensuite
les 28 paramètres pour ce modèle précis. Le modèle ControlNet ne doit pas être ajouté comme simple
variable au surrogate : changer de poids implique de décharger et recharger plusieurs gigaoctets et
constitue une population expérimentale distincte.
