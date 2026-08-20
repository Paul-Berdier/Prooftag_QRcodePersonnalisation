# E029 — récupération exacte du raster Stage 2

## Pourquoi E029 est nécessaire

L'archive E028 contient 1 170 résultats complets, mais elle révèle une erreur de protocole dans
le post-traitement SR-MPGD. Parmi les 450 branches SR-MPGD, 335 ont retenu l'itération zéro.
Pourtant leur image ne correspondait pas pixel pour pixel à leur Stage 2 parent.

La cause est précise : l'implémentation décodait à nouveau le latent Stage 2 avec le VAE pour
construire l'état zéro. Un VAE est imparfait ; cette reconstruction modifiait le raster avant la
première descente de gradient. Le résultat était ensuite nommé « no-op », alors qu'il ne l'était
pas. Sur E028, SR-MPGD améliorait la tolérance QR dans 49 cas, la dégradait dans 344 et la laissait
inchangée dans 57. La porte de livraison était gagnée dans 12 cas mais perdue dans 207. Ces nombres
ne mesurent donc pas proprement l'algorithme publié.

## Correction

`run_srmpgd` reçoit maintenant le PNG Stage 2 original en plus de son latent :

- le latent reste utilisé pour la loss différentiable et les itérations `i > 0` ;
- l'itération zéro utilise une copie exacte du raster Stage 2 ;
- si l'itération zéro gagne, son SHA-256 de pixels doit être identique à celui du Stage 2 ;
- le backend publie `diffqrcoder_srmpgd_iteration_zero_exact=1` ;
- une divergence déclenche une erreur au lieu de produire une fausse sortie SRPG.

Cette règle est appliquée au backend DiffQRCoder et au backend ControlNet générique.

## Campagne de récupération

E029 rejoue une campagne plus petite avant tout nouvel entraînement :

- 10 prompts de reprise ;
- 3 seeds appariées ;
- une chaîne fixe et une chaîne conseillée ;
- pour chaque chaîne : Stage 1, Stage 2, SR-MPGD ;
- 6 états par contexte, donc 180 générations au total.

Les Stage 1 et Stage 2 doivent être régénérés : l'archive E028 contient les PNG et les métriques,
mais aucun latent `.safetensors`. Un SR-MPGD différentiable ne peut pas être repris fidèlement à
partir du PNG seul.

Le notebook arrête l'analyse si :

1. la réutilisation Stage 1/Stage 2 n'est pas prouvée ;
2. un SR-MPGD sélectionne l'itération zéro avec un hash différent de son Stage 2 parent ;
3. le marqueur backend d'identité exacte manque.

La comparaison QR-first est ensuite calculée comme avant : payload exact, tolérance QR-Verify,
absence de saturation, puis HPS v2.1, CLIP-Aesthetic et CLIPScore. Stage 1 reste interdit à la
livraison.

## Exécution

Sur le serveur Linux, après commit et push :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git fetch origin
git switch main
git pull --ff-only origin main
bash scripts/deploy-e029-notebook.sh
```

Sur le PC Windows :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
git pull
.\scripts\notebook-remote.ps1 -Notebook 24_e029_srmpgd_exact_raster_recovery.ipynb
```

Dans Jupyter, vérifier `COLLECTION_PAYLOAD`, puis utiliser **Run > Run All Cells**. La reprise est
persistée sous `/data/e029-srmpgd-raster/<plan-id>` et l'archive finale est copiée dans
`/workspace/downloads`.

## Décision après E029

On ne réentraîne pas le conseiller SR-MPGD sur les libellés E028 corrompus. Si l'invariant E029
passe, ses données et les futures campagnes corrigées pourront remplacer progressivement ces
observations. La campagne large ne sera relancée qu'après examen du taux de porte gagné/perdu et
des planches visuelles E029.
