# E025 — rétablissement des scores d'image

## But

E025 conserve `antfu/qr-verify@0.2.0` comme unique autorité de scannabilité et
réactive quatre mesures d'image indépendantes. Aucune ne peut accepter ou
rejeter un QR : elles servent à comparer les sorties déjà générées.

| Champ CSV/API | Signification | Usage |
|---|---|---|
| `clip_similarity` | cosinus image–prompt OpenAI CLIP ViT-B/32 | échelle comparable au tableau DiffQRCoder, typiquement autour de 0,2–0,4 |
| `clip_score` | `2,5 × max(clip_similarity, 0)` | définition CLIPScore rescalée |
| `clip_aesthetic` | régression linéaire LAION sur l'embedding CLIP, environ 0–10 | continuité avec les campagnes et le CLIP-aes. du papier |
| `hpsv2_1` | préférence image–prompt HPS v2.1 | départager des images produites pour le même prompt |

Le score HPS ne doit pas être comparé naïvement entre deux prompts différents.
La moyenne de campagne affichée dans le Web Lab reste donc indicative ; la
comparaison scientifique se fait prompt par prompt, avec les mêmes seeds.

## Choix technique

CLIP et le prédicteur LAION sont déjà compatibles avec la version de
`transformers` épinglée par DiffQRCoder. HPS v2.1 est installé depuis la révision
officielle déjà verrouillée dans `pyproject.toml`. Les modèles tournent sur CPU
et sont chargés une seule fois, afin de réserver la RTX 4000 Ada à la diffusion.
Un échec de téléchargement ou de scoring est journalisé et laisse le champ
absent ; il ne détruit jamais le résultat QR-Verify.

HPSv3 (ICCV 2025) est plus récent et plus performant sur les benchmarks publiés,
mais repose sur Qwen2-VL 7B. Le charger dans le pod de génération entrerait en
concurrence avec DiffQRCoder pour les 20 Go de VRAM et impose une pile
`transformers` différente. Il sera pertinent dans un service de scoring GPU
séparé, pas dans ce déploiement mono-GPU.

## Refaire la comparaison E024

Après déploiement, depuis le PC Windows :

```powershell
.\scripts\lab-remote.ps1
.\.venv\Scripts\python.exe .\scripts\e025-quality-retest.py `
  --api http://127.0.0.1:18080 `
  --payload "https://ptag.io/t/e025" `
  --family all `
  --seeds 61001 `
  --launch
```

La campagne contient les dix prompts et les deux recettes appariées d'E022/E024.
Le premier score peut être plus lent, car les poids sont téléchargés sur le PVC
`/cache`; les générations suivantes réutilisent les modèles en mémoire.
