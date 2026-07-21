# Benchmark reproductible

Le benchmark exécute six cas fixes représentant plusieurs styles graphiques, longueurs de
payload et versions QR. Chaque cas autorise jusqu'à trois seeds déterministes. Les paramètres
restent identiques entre deux versions du code afin que les écarts mesurés proviennent du
pipeline et non du hasard.

Le protocole 3.0 inscrit dans chaque rapport sa version, le hash SHA-256 des cas et les
paramètres complets de génération. Une comparaison dont le hash ou la version diffère doit
être présentée comme une nouvelle campagne, pas comme une régression directe.

`PROOFTAG_QR_BENCHMARK_MAX_ATTEMPTS` peut borner une campagne sentinelle ; sa valeur est inscrite
dans `environment.json` et `summary.json`. E005a fixe cette valeur à 1 pour exécuter exactement
six boucles SRPG avant d'autoriser une campagne trois fois plus coûteuse.

## Depuis le serveur

Après le déploiement d'une nouvelle image :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
make benchmark
```

La commande ouvre elle-même un port-forward temporaire, génère les six images et crée :

- les images finales, brutes et variantes disponibles pour chaque tentative ;
- la réponse JSON et le snapshot Prometheus de chaque génération ;
- `summary.csv`, `variants.csv`, `variant-failures.csv`, `validations.csv`,
  `refinements.csv`, `srpg-steps.csv` et `comparison.csv` ;
- les informations Git, GPU, runtime et Kubernetes ;
- un échantillonnage GPU chaque seconde (`gpu-samples.csv`) avec utilisation, VRAM,
  température et puissance ;
- `report.html`, avec galerie et graphiques ;
- une archive `.tar.gz` prête à être transférée.

Les résultats sont conservés dans `benchmark-results/`. À partir de la deuxième exécution,
le rapport calcule automatiquement les écarts avec le benchmark précédent.

Le taux « premier essai » mesure la qualité intrinsèque d'une seed. Le taux « livraison
finale » mesure la capacité réelle du service après régénération et fallback. Le rapport
indique séparément le nombre de cas ayant nécessité une correction globale.

Le protocole 3.0 ajoute le taux strict de `raw`, `guided`, `srpg` et de la variante SRL disponible,
le nombre de sauvetages obtenus par le latent et leurs taux moyens. Ces indicateurs ne doivent
pas être remplacés par le taux final : ils mesurent le progrès réel du modèle avant fallback.

Les variantes `rounded_*` sont essayées en premier : elles corrigent la luminance avec des
formes arrondies à bords fondus pour réduire l'effet de grille. Les variantes `perceptual_*`
puis binaires restent disponibles comme replis de robustesse. `selected_variant` et
`variant-failures.csv` permettent de suivre le palier retenu pour chaque image.

Quand E004 est activée, la galerie conserve `raw`, `guided_control`, `guided_mask`,
`guided_unprojected`, `guided_projected`, `guided`,
`guided_latent_srl` si elle existe, puis l'image finale. `refinements.csv` enregistre le statut,
la durée, les paramètres, les erreurs modules et les écarts visuels de chaque étape. Les modes
restent désactivés dans le manifeste de production tant qu'une campagne d'ablation n'a pas
identifié des paramètres sûrs.

Pour exécuter automatiquement une baseline et E004 sur le même commit, puis restaurer les
valeurs désactivées même en cas d'erreur :

```bash
make benchmark-e004
```

E005 compare de la même façon une baseline au vrai guidage SRPG seul. Chaque sortie SRPG est
validée même si sa porte interne la rejette ; le rapport ne confond donc plus « loss améliorée »
et « QR effectivement lu ». La galerie inclut `attempt_1_srpg.png`, une courbe des 40 pas et
`srpg-steps.csv` contient les losses, gradients et clips :

```bash
make benchmark-e005
```

## Une commande depuis le PC Windows

Depuis PowerShell, dans le dépôt local :

```powershell
.\scripts\benchmark-remote.ps1
```

Cette commande lance le benchmark sur `paul@pcIA`, copie l'archive dans
`Downloads\prooftag-benchmarks`, l'extrait puis ouvre le rapport. Utiliser `-NoOpen` pour ne
pas ouvrir le navigateur automatiquement.

Pour E004, la commande suivante lance les deux campagnes distantes, rapatrie les deux archives
et ouvre le rapport guidé, qui compare automatiquement la seconde campagne à la baseline :

```powershell
.\scripts\benchmark-remote.ps1 -E004
```

Pour E005 :

```powershell
.\scripts\benchmark-remote.ps1 -E005
```

Un serveur ou un chemin différent peut être indiqué explicitement :

```powershell
.\scripts\benchmark-remote.ps1 `
  -Server paul@192.168.1.20 `
  -RemoteRepository /home/paul/apps/Prooftag_QRcodePersonnalisation
```

Une clé SSH est recommandée pour éviter les demandes répétées de mot de passe par `ssh` et
`scp`.
