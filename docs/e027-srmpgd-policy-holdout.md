# E027 — comparaison cascade, sélection complète et SR-MPGD forcé

## Question testée

E026J a montré que le rang 1 du conseiller atteint 30/30 lectures QR-Verify, mais que SR-MPGD ne
modifie réellement que 3 sorties sur 24 demandes. E027 répond à une question plus directe :

> Lorsque la scannabilité est prioritaire et l'esthétique secondaire, faut-il toujours livrer la
> sortie demandée SR-MPGD, ou faut-il conserver Stage 1 et Stage 2 comme candidats ?

Le test ne change ni le modèle, ni le ControlNet, ni la géométrie. Il utilise Cetus-Mix Whalefall,
QR Monster v2, DiffQRCoder épinglé et la cible binaire exacte existante.

## Appariement obligatoire

Chaque contexte est défini par un prompt, un payload et une seed. Les trois états sont produits
dans la même campagne et dans cet ordre :

1. `e027_stage1` génère le Stage 1 ;
2. `e027_stage2` réutilise exactement ce Stage 1 et exporte son latent propre ;
3. `e027_srmpgd` importe exactement ce latent Stage 2, puis exécute le SR-MPGD robuste.

Le cache du laboratoire vérifie le SHA-256 du latent. Une source Stage 2 manquante ou différente
est une erreur technique, pas un échec QR ordinaire.

SR-MPGD utilise quatre itérations maximales, gamma 100, LPIPS 0,10, les limites de déplacement
latent et les gardes de saturation existantes. L'itération zéro fait partie des candidats. Le
raffinement s'arrête si cette itération franchit déjà la porte QR-Verify fixée à 0,80.

## Holdout

Le protocole par défaut contient :

- 100 prompts déterministes nouveaux, moitié simples et moitié atypiques ;
- trois seeds nouvelles : `743001`, `857001`, `971001` ;
- 300 contextes prompt/seed ;
- trois états par contexte, soit 900 essais persistants ;
- cinq campagnes de vingt prompts, chacune reprenable après une coupure.

Le plan persistant ne contient pas le payload clair. Son identifiant dépend du hash du payload,
des prompts, des seeds, des profils complets et du seuil QR.

## Sélection lexicographique

La fonction `select_e027_candidate` classe les images dans l'ordre suivant :

1. payload exact restitué par QR-Verify ;
2. tolérance QR-Verify ;
3. garde de saturation ;
4. HPS v2.1 ;
5. CLIP-Aesthetic ;
6. CLIPScore ;
7. saturation plus faible ;
8. durée plus faible.

Une image n'est livrable que si le payload est exact et si sa tolérance atteint 0,80. Une image
ayant un meilleur score esthétique ne peut jamais dépasser une image QR valide ou plus robuste.

## Trois politiques rejouées

### Cascade

Stage 1 est livré immédiatement s'il franchit la porte robuste. Sinon, Stage 2 et SR-MPGD sont
considérés et la sélection lexicographique choisit le meilleur état. E027 génère volontairement
les trois états pour conserver un test apparié ; le coût de la cascade est donc simulé dans le
rapport avant d'être implanté comme arrêt physique de production.

### Chaîne complète avec sélection

Les trois états sont toujours calculés. Le meilleur état mesuré est livré, même s'il s'agit de
Stage 1 ou de l'itération zéro du Stage 2.

### SR-MPGD forcé

La sortie du profil SR-MPGD est toujours prise. Si SR-MPGD conserve l'itération zéro, cette sortie
est honnêtement étiquetée SRPG. Cette politique mesure directement le coût de la règle « toujours
prendre SR-MPGD ».

## Critère de décision

La politique SR-MPGD forcée ne sera promue que si elle ne provoque aucune régression QR face à la
sélection complète et si son avantage de tolérance justifie ses pertes esthétiques et son coût.
La cascade ne sera promue comme politique de production que si son intervalle de confiance QR
reste compatible avec la chaîne complète.

Avec zéro échec sur 300 contextes, la borne basse unilatérale naïve à 95 % approche 99 %. Les trois
seeds d'un même prompt ne sont toutefois pas totalement indépendantes. Le rapport publie donc
aussi un verdict plus conservateur par prompt : un prompt ne réussit que si ses trois seeds
franchissent la porte. Aucun résultat ne doit être annoncé avant la fin des 300 triplets, et une
revendication statistique de 99 % demandera ensuite environ 300 prompts réellement indépendants.

## Lancement

Déployer sur le serveur depuis un dépôt propre :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git fetch origin
git switch main
git pull --ff-only origin main
bash scripts/deploy-e027-notebook.sh
```

Puis, sur le PC Windows :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
git pull
.\scripts\notebook-remote.ps1 -Notebook 22_e027_srmpgd_policy_holdout.ipynb
```

Dans Jupyter, lancer **Run > Run All Cells**. Une reprise utilise la même commande et relance les
cellules ; les lots terminés sont ignorés.

## Artefacts

Le notebook produit :

- `e027-state-results.csv` : les 900 états mesurés ;
- `e027-policy-decisions.csv` : les trois décisions pour chacun des 300 contextes ;
- `e027-pairing-audit.csv` : états manquants et complétude technique ;
- `e027-policy-report.json` : taux, intervalles, coûts et comparaisons appariées ;
- `e027-policy-scorecard.png` ;
- `e027-gallery/images` : tous les PNG ;
- les planches appariées, les échecs et les désaccords avec SR-MPGD forcé ;
- `manifest.json` et l'archive `.tar.gz` complète.

## Limites

- QR-Verify demeure un test logiciel, pas un scan de téléphone physique.
- HPS, CLIP-Aesthetic et CLIPScore sont des proxys, pas une note humaine.
- Les 300 contextes doivent rester hors du prochain entraînement jusqu'à la décision finale.
- Le coût de la cascade est simulé sur un run qui calcule tous les états afin de garantir une
  comparaison équitable.
