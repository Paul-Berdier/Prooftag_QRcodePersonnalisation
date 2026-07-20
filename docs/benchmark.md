# Benchmark reproductible

Le benchmark exécute six cas fixes représentant plusieurs styles graphiques, longueurs de
payload et versions QR. Les prompts, paramètres et seeds restent identiques entre deux
versions du code afin que les écarts mesurés proviennent du pipeline et non du hasard.

## Depuis le serveur

Après le déploiement d'une nouvelle image :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
make benchmark
```

La commande ouvre elle-même un port-forward temporaire, génère les six images et crée :

- les images finales, brutes et variantes disponibles ;
- la réponse JSON et le snapshot Prometheus de chaque génération ;
- `summary.csv`, `variants.csv`, `validations.csv` et `comparison.csv` ;
- les informations Git, GPU, runtime et Kubernetes ;
- un échantillonnage GPU chaque seconde (`gpu-samples.csv`) avec utilisation, VRAM,
  température et puissance ;
- `report.html`, avec galerie et graphiques ;
- une archive `.tar.gz` prête à être transférée.

Les résultats sont conservés dans `benchmark-results/`. À partir de la deuxième exécution,
le rapport calcule automatiquement les écarts avec le benchmark précédent.

## Une commande depuis le PC Windows

Depuis PowerShell, dans le dépôt local :

```powershell
.\scripts\benchmark-remote.ps1
```

Cette commande lance le benchmark sur `paul@pcIA`, copie l'archive dans
`Downloads\prooftag-benchmarks`, l'extrait puis ouvre le rapport. Utiliser `-NoOpen` pour ne
pas ouvrir le navigateur automatiquement.

Un serveur ou un chemin différent peut être indiqué explicitement :

```powershell
.\scripts\benchmark-remote.ps1 `
  -Server paul@192.168.1.20 `
  -RemoteRepository /home/paul/apps/Prooftag_QRcodePersonnalisation
```

Une clé SSH est recommandée pour éviter les demandes répétées de mot de passe par `ssh` et
`scp`.
