# E005 — vraie boucle DDIM guidée par SRPG

## Pourquoi E005 existe

E004 ajoutait une deuxième passe ControlNet puis recopiait localement son résultat. Ce n'était
pas le guidage SRPG de DiffQRCoder. La campagne du 21 juillet a confirmé l'échec : la sortie
guidée restait illisible, l'erreur module augmentait et la réparation finale devenait plus
visible. E005 remplace cette approximation par un gradient calculé **dans chaque timestep**.

## Chaîne réellement exécutée

```text
sortie artistique x_hat de l'étape 1
  -> encodage VAE + bruit déterministe au timestep T
  -> 40 timesteps DDIM
       -> ControlNet + UNet prédisent epsilon
       -> calcul différentiable de z0|t
       -> décodage VAE de z0|t
       -> SRL QR + LPIPS(x0|t, x_hat)
       -> gradient de cette loss par rapport à zt
       -> epsilon_guidé = epsilon + sqrt(1-alpha_t) * gradient
       -> garde-fou sur la norme du delta de bruit
       -> pas DDIM vers z(t-1)
  -> décodage de l'image SRPG
  -> validation indépendante par 2 décodeurs × 13 scénarios
  -> si l'erreur QR réelle baisse : réparations ciblées depuis SRPG
  -> sinon : SRPG reste mesuré et archivé, mais les réparations repartent du brut
  -> sélection de l'image lisible ayant la plus faible MAE
```

Le code correspond aux équations 7 à 11 et à l'algorithme 1 de DiffQRCoder pour la partie SRPG.
Il fige les poids du VAE, de l'UNet, du ControlNet et du text encoder ; seul le latent courant
reçoit un gradient. Le gradient checkpointing est activé pour l'UNet et ControlNet afin de tenir
sur la RTX 4000 Ada 20 Go.

## Paramètres E005a

| Paramètre | Valeur | Origine / rôle |
|---|---:|---|
| Timesteps DDIM | 40 | protocole DiffQRCoder |
| strength img2img | 1,0 | bruit au timestep maximal de l'étape 2 |
| poids ControlNet | 1,35 | réglage QR Code Monster de l'article |
| lambda SRL | 500 | meilleur point SSR de l'ablation publiée |
| lambda LPIPS | 3 | compromis perceptuel publié |
| poids motifs fonctionnels | 4 | garde-fou Prooftag additionnel |
| cible erreur centrale | 0 | guidage tant qu'un module central est faux |
| delta bruit RMS maximal | 2,0 | protection contre un gradient explosif |
| MAE maximale | 0,20 | porte interne avant réutilisation comme base |
| amélioration module minimale | 10 % relatif | évite le faux positif de E004 |

Ces valeurs sont une **hypothèse E005a**, pas un réglage de production. Le mode reste désactivé
dans le ConfigMap.

## Ce qui est mesuré

Pour chaque image et chaque tentative :

- erreur centrale, SRL, LPIPS, RMS du gradient et RMS du delta de bruit à chacun des 40 pas ;
- nombre de pas ayant déclenché le clip du gradient ;
- erreur module réelle avant/après sur l'image PIL finale ;
- pixels modifiés, MAE, durée et pic d'allocation CUDA ;
- résultat des 26 couples décodeur/scénario sur l'image SRPG, même si la porte interne la rejette ;
- image `attempt_1_srpg.png`, courbe HTML, `refinements.csv` et `srpg-steps.csv`.

Prometheus et Grafana suivent aussi les résultats, la durée P95, l'erreur avant/après, la VRAM et
les diagnostics par pas. Une alerte signale les erreurs, une inefficacité supérieure à 75 % et
un pic d'allocation dépassant 18 Gio.

## Protocole sans confusion causale

E005a compare sur le même commit et une seule tentative déterministe par cas :

1. baseline avec ancien guidage, SRPG et raffinement latent désactivés ;
2. SRPG seul, avec ancien guidage et raffinement latent désactivés.

Cette sentinelle exécute donc 6 boucles SRPG, pas 18. Les réparations globales restent disponibles
sur l'unique tentative afin de vérifier la non-régression de livraison. Une campagne à trois
tentatives n'est autorisée qu'après passage des portes 1–4, car elle triplerait le coût GPU sans
apporter d'information si l'implémentation ne converge pas.

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
make pause-vllm
make benchmark-e005
```

Depuis le PC Windows :

```powershell
.\scripts\benchmark-remote.ps1 -E005
```

Le script restaure les trois options Kubernetes à leur valeur du ConfigMap même après une erreur.
Après la campagne, reprendre vLLM explicitement avec `make resume-vllm` uniquement si le GPU ne
doit plus rester attribué au service QR.

## Portes de décision

E005a n'est retenue pour ablation que si :

1. les six événements `srpg_completed` existent et contiennent exactement 40 pas finis ;
2. aucune valeur de loss ou de gradient n'est NaN/Inf et aucune erreur CUDA/OOM n'apparaît ;
3. l'erreur module réelle moyenne baisse d'au moins 10 % relatif ;
4. au moins une image SRPG gagne un scénario de lecture par rapport au brut ;
5. les six livraisons finales restent à 26/26 sans mauvais payload ;
6. le pic d'allocation reste inférieur à 18 Gio et la durée est documentée ;
7. l'inspection visuelle confirme une intégration plus naturelle, sans grille supplémentaire.

Si 1–2 échouent, on corrige l'implémentation avant toute ablation. Si 3–4 échouent, E005b fait
varier une seule dimension : lambda SRL `400/500/600`, puis LPIPS `0/2/3/5`, puis strength. Une
campagne de dataset ou un entraînement reste bloqué tant qu'aucune configuration ne franchit ces
portes.

## Différence encore ouverte avec DiffQRCoder

E005a n'implémente pas QArt, qui transforme le QR de contrôle pour le rapprocher de l'image de
l'étape 1. Le dépôt officiel annoncé dans l'article ne fournit pas une implémentation exploitable
de cette transformation dans notre pile. E005 utilise donc le QR exact comme condition
ControlNet. Les résultats E005 sont ceux de Prooftag sur sa RTX ; ils ne peuvent pas être assimilés
au SSR de 99–100 % publié par l'article. QArt constitue E005b seulement après validation de la
boucle SRPG elle-même.
