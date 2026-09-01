# E040 — checkpoint-aware SR-MPGD + vue de pipeline finale

## But

E039 a montré que le meilleur SSR peut apparaître avant la dernière itération. E040 garde donc le parent figé E035, la loss `scanaware_v2`, LPIPS 0,01 et **gamma=1000**, puis explore les rayons 0,150 / 0,175 / 0,200 / 0,225 / 0,250 avec huit updates maximum.

Chaque état i0…i8 est un artefact complet : PNG, latent safetensors, QR-Verify, MER, LPIPS, CLIPScore, CLIP-Aesthetic, HPS, clipping, saturation et garde visuelle. Le gagnant est le meilleur checkpoint `visual_guard_pass=True` selon SSR réel, original decode, MER, score E016 secondaire, puis LPIPS.

## Modèles entraînés

- **Advisor E026/E031** : chargé depuis `/data/e031-prospective-stage2-models` lorsqu'un `.joblib` existe. Il produit une recommandation prospective de paramètres. Il ne certifie jamais la sortie.
- **Surrogate E016** : chargé seulement si `scan-surrogate.research-only.torchscript.pt` existe et si sa `surrogate-card.json` marque `promotion.research_usable=true`. Il score les checkpoints après génération. Il ne remplace ni QR-Verify ni les vrais décodeurs.

## Pipeline visuelle

Le dossier `pipeline/` contient :

1. `01-qr-reference.png` — QR exact payload ;
2. `02-control-condition.png` — condition binaire ControlNet ;
3. `03-stage1.png` — Stage 1 exact archivé ;
4. `04-stage2.png` — Stage 2 parent figé ;
5. la trajectoire SR-MPGD du gagnant i0…i8 ;
6. `99-FINAL-QR.png` et `99-FINAL-latent.safetensors` ;
7. `full-pipeline-contact-sheet.png`.

Le notebook `36_final_qr_pipeline_visualizer.ipynb` montre ces étapes une par une en grand.

## Limite scientifique

E040 reste une recherche mono-parent. Même un meilleur checkpoint sûr n'autorise pas encore une conclusion de généralisation ou de scan téléphone. Le prochain gate n'est autorisé qu'après revue visuelle du gagnant et comparaison avec E039.
