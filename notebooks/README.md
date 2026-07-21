# Notebooks Prooftag QR

Les deux notebooks n'ont pas le même rôle :

- `01_srpg_step_by_step.ipynb` analyse une archive de benchmark déjà produite. Il ne génère rien.
- `02_generate_live_on_gpu.ipynb` exécute réellement le modèle sur la RTX du serveur et montre
  chaque étape au fur et à mesure.

## Génération réelle depuis le PC Windows

Le navigateur s'ouvre sur le PC, mais le kernel Python, Stable Diffusion, ControlNet et CUDA
s'exécutent dans Kubernetes sur `pcIA`. Il ne faut donc pas lancer le notebook 02 avec le Python
Windows.

Une fois la version notebook construite et déployée sur le serveur, lancer depuis PowerShell dans
le dépôt local :

```powershell
cd "C:\Users\p.berdier\Documents\Paul Berdier\codage\Prooftag_QRcodePersonnalisation"
.\scripts\notebook-remote.ps1
```

Le tunnel utilise `http://127.0.0.1:18888` par défaut afin de ne pas entrer en conflit avec un
Jupyter Windows déjà lancé sur le port 8888. Si 18888 est occupé, choisir explicitement un autre
port : `./scripts/notebook-remote.ps1 -LocalPort 18889`.

Cette commande :

1. mémorise l'état de l'API QR et de vLLM ;
2. les arrête pour libérer l'unique GPU ;
3. démarre le pod Jupyter avec la RTX ;
4. crée un tunnel SSH privé ;
5. ouvre directement `02_generate_live_on_gpu.ipynb` sur le PC.

Sans clé SSH, une seconde fenêtre s'ouvre pour le tunnel : saisir le mot de passe `paul@pcIA`
dans cette fenêtre et la laisser ouverte pendant la session Jupyter. Elle sera fermée par la
commande `-Stop`.

Dans Jupyter, utiliser **Run > Run All Cells**. Le notebook fabrique alors, sans archive :

1. le QR de contrôle ;
2. la diffusion artistique brute ;
3. sa validation ;
4. la seconde diffusion SRPG avec aperçu `x0` et carte d'erreur à chacun des 40 pas ;
5. les courbes de loss et d'erreur de modules ;
6. chaque réparation candidate ;
7. la validation multi-décodeur et multi-dégradation de chaque candidate ;
8. la sélection finale et les exports dans `results/notebook-runs/<date>-<seed>`.

À la fin, arrêter Jupyter et restaurer exactement les nombres de réplicas précédents :

```powershell
.\scripts\notebook-remote.ps1 -Stop
```

## Première installation ou mise à jour sur le serveur

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
git pull
docker build -t prooftag-qr:dev .
docker build -f Dockerfile.notebook -t prooftag-qr-notebook:dev .
docker save prooftag-qr:dev prooftag-qr-notebook:dev | sudo k3s ctr images import -
bash scripts/create-database-secret.sh
kubectl apply -k deploy/k8s
kubectl get deployment/prooftag-qr-notebook -n qr-core
```

Le Deployment notebook reste à zéro réplique tant que la commande PowerShell ne le démarre pas.
Les modèles réutilisent le PVC `prooftag-qr-model-cache` et les résultats persistent dans le PVC
`prooftag-qr-data`, sous `/data/notebook-runs`.

## Analyse d'une ancienne archive sur Windows

Le notebook 01 reste utile pour comparer une campagne déjà rapatriée :

```powershell
python -m pip install -e ".[notebook]"
$env:PROOFTAG_QR_BENCHMARK_ARCHIVE = "$HOME\Downloads\prooftag-benchmarks\20260721T090541Z-0b3c040b.tar.gz"
jupyter lab notebooks\01_srpg_step_by_step.ipynb
```

Il réutilise le dossier déjà extrait ou écrit dans `.prooftag-notebook-cache` à côté de l'archive.
Ce notebook d'analyse n'utilise pas le GPU du serveur.
