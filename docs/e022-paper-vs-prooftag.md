# E022 — Prooftag sécurisé contre protocole du PDF

## Question et matrice

Sur dix prompts inédits, quelle recette de Stage 2 offre le meilleur compromis
entre lecture QR et qualité visuelle ? Les cinq prompts `simple` et les cinq
prompts `atypical` sont fixés avant de voir les résultats.

| Paramètre | Prooftag sécurisé | Protocole PDF public |
|---|---:|---:|
| Stage 1 | partagé | le même latent exact |
| Cible Stage 2 | QR binaire exact | QArt public à fragment d'URL |
| Force de bruit | 0,65 | 1,00 |
| Pas effectifs attendus | 26/40 | 40/40 |
| ControlNet | 1,35 | 1,35 |
| SRG `λ1` | 500 | 500 |
| PG `λ2` | 2 | 3 |
| SR-MPGD | non | non |

La seconde branche est une **approximation publique**, pas une reproduction
parfaite : le constructeur QArt des auteurs n'est pas publié. Notre binaire
`andrewyur/qart` est épinglé et ajoute un fragment à l'URL ; la validation
compare donc l'URL canonique après retrait du fragment.

SR-MPGD est volontairement exclu. L'ajouter introduirait simultanément les
itérations, `gamma`, LPIPS et les gardes esthétiques. Il sera appliqué ensuite à
la branche gagnante.

## Invariants

Pour chaque prompt, le payload, la correction M, la seed, le prompt négatif et
le Stage 1 sont identiques. Le Stage 1 est calculé une fois puis réutilisé. Aucun
résultat n'est remplacé par le QR témoin. La campagne principale contient donc
`10 prompts × 1 seed × 2 recettes = 20 résultats`.

## Lancement depuis Windows

Ouvrir le tunnel :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
.\scripts\lab-remote.ps1
```

Puis, dans un second PowerShell :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
python .\scripts\e022-paper-vs-prooftag.py `
  --payload "https://ptag.io/t/e022" `
  --family all `
  --seeds 61001 `
  --launch
```

Remplacer l'URL d'exemple par une URL Prooftag courte réelle. Ouvrir ensuite
`http://127.0.0.1:18080/lab`, noter chaque image et exporter le CSV.

## Notation et décision

Faire trois essais de scan par image, avec le même téléphone, la même distance
et la même luminosité. Noter esthétique, fidélité au prompt, discrétion du QR,
ainsi que flou, saturation, grille visible ou hors-prompt.

Le classement est lexicographique : taux de scan téléphone, lectures originales
des décodeurs, SSR robuste normalisé, notes humaines, puis CLIPScore et
CLIP-Aesthetic. Le rapport final donnera aussi les résultats séparés des familles
`simple` et `atypical`, et les dix paires image par image.
