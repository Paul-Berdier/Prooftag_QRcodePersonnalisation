# E030 — QR-Verify répétable et cascade Stage 2

## Pourquoi E030 remplace une nouvelle campagne GPU

E029 v4 a produit 180 états correctement appariés, mais son analyse a révélé que QR-Verify
pouvait attribuer des scores différents à un raster strictement identique. Dans ces conditions,
les écarts proches du seuil `0,80` et les faibles gains attribués à SR-MPGD ne sont pas
scientifiquement identifiables.

E030 ne génère donc aucune image. Il ferme d'abord ce défaut de mesure en rescannant les images
E029 existantes sur CPU. Les décisions E029 qui sont déjà établies restent verrouillées :

- Stage 1 n'est jamais livré ;
- le Stage 2 fixe est la première tentative ;
- SR-MPGD n'est pas un secours automatique puisqu'il n'a sauvé aucune porte dans E029 ;
- le conseiller n'est utilisé qu'en Stage 2 alternatif après l'échec de la recette fixe.

Les résultats E029 complets et les raisons de cette décision sont dans
[`e029-results-2026-08-20.md`](e029-results-2026-08-20.md).

## Mesure conservatrice

Chaque raster RGB unique est évalué cinq fois par `antfu/qr-verify@0.2.0`. Une répétition contient
les 37 presets du bridge épinglé. E030 conserve :

- les 37 résultats de chacune des cinq répétitions ;
- le minimum, la moyenne et le maximum des scores par répétition ;
- le nombre de presets instables ;
- l'intersection des presets qui restituent le payload exact lors des cinq répétitions.

La tolérance conservatrice est la taille de cette intersection divisée par 37. La porte exige :

1. au moins une lecture exacte lors de chacune des cinq répétitions ;
2. une tolérance conservatrice supérieure ou égale à `0,80` ;
3. un risque de saturation inférieur ou égal à `0,05` ;
4. le même hash de raster dans toutes les observations.

Le cache est adressé par le hash des pixels RGB, le hash du payload, la version du moteur, le hash
du bridge et de son lockfile, la version du scorer, le nombre de répétitions et les 37 presets.
Une reprise après coupure ne rescannera donc pas les images déjà terminées avec exactement le même
protocole. Le payload décodé brut n'est jamais écrit dans le cache.

## Politiques rejouées

E030 compare quatre cascades sur les images E029 déjà calculées :

1. Stage 2 fixe, première seed ;
2. Stage 2 fixe puis Stage 2 alternatif, première seed ;
3. Stage 2 fixe puis nouvelle seed ;
4. Stage 2 fixe, Stage 2 alternatif si nécessaire, puis nouvelle seed.

Une tentative suivante n'est comptée que si la précédente échoue. Aucune de ces politiques ne
peut sélectionner Stage 1 ou SR-MPGD. L'expérience reste un replay logiciel sur dix prompts : une
politique gagnante devra ensuite être confirmée prospectivement, puis physiquement.

## Source et reprise

Le notebook cherche d'abord le dernier export E029 v4 complet sur le PVC, sous
`/data/e029-srmpgd-raster`. Il vérifie les 180 lignes, les 180 images et chaque checksum avant de
lire quoi que ce soit. Une archive `.tar.gz` explicite reste possible avec
`PROOFTAG_E030_SOURCE_ARCHIVE`; son extraction est sélective et refuse les chemins dangereux. Le
modèle Joblib de 1,35 Go n'est jamais extrait.

Le payload E029 est `https://ptag.io/t/e029`. Son texte est vérifié contre le SHA-256 et la longueur
du manifeste avant la première lecture. Une source utilisant un autre payload s'arrête au lieu
d'inférer le texte depuis une sortie de décodeur.

## Lancement

Après commit et push, sur le serveur Linux :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git fetch origin
git switch main
git pull --ff-only origin main
bash scripts/deploy-e030-notebook.sh
```

Le script construit uniquement l'image notebook immuable. E030 tourne sur CPU et ne change ni les
réplicas de vLLM ni ceux de l'API. Depuis PowerShell sur le PC :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
git pull
.\scripts\notebook-remote.ps1 -Notebook 25_e030_reliable_qrverify_cascade.ipynb
```

Dans Jupyter, utiliser **Run > Run All Cells**. Pour fermer uniquement le notebook et restaurer
l'état mémorisé :

```powershell
.\scripts\notebook-remote.ps1 -Stop
```

## Sorties

Les résultats persistants sont écrits sous `/data/e030-reliable-qrverify/<run-id>` :

- journal JSONL reprenable et progression ;
- inventaire source et scores des rasters uniques ;
- candidats enrichis avec les cinq observations ;
- décisions et agrégats des quatre politiques ;
- graphes d'instabilité et de taux de livraison ;
- planche et PNG des candidats livrés ;
- rapport JSON/Markdown ;
- manifeste avec commit Git, tag et digest de l'image runtime, ainsi que les checksums ;
- archive finale et fichier `.sha256`.

Le dossier `/workspace/downloads` ne sert qu'à rendre une copie facilement téléchargeable depuis
Jupyter. La preuve canonique reste sur le PVC afin de survivre à l'arrêt ou au remplacement du pod.
