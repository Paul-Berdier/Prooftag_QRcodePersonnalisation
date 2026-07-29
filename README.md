# Prooftag QR esthétique

Socle expérimental et opérationnel pour générer des QR codes intégrés à des images,
mesurer leur qualité et ne publier que les résultats validés.

## État actuel

La version `0.1.0` fournit désormais un laboratoire volontairement recentré sur
DiffQRCoder et sur l’algorithme décrit dans son papier :

- une API FastAPI ;
- un laboratoire Web persistant pour composer, lancer, comparer et noter des campagnes ;
- un générateur QR de référence (`backend=qr`) ;
- DiffQRCoder épinglé au commit `e24ea73ee2e13c7e6e87cb422e8b11784e70ae00` ;
- Cetus-Mix Whalefall + QR Monster v2 pour le Stage 1 ;
- le Stage 2 DDIM/SRPG initialisé depuis le latent VAE bruité du Stage 1 ;
- une cible QArt reconstruite et explicitement distinguée du code QArt non publié ;
- un SR-MPGD séparé conforme aux équations 13–14, avec validation de chaque état ;
- une géométrie entière QR v3/M/masque 4 conforme aux exemples publics ;
- aucune réparation déterministe ou superposition du QR témoin dans les profils du Web Lab ;
- des garde-fous de divergence et de saturation sans réparation cachée de l’image ;
- une validation exacte du payload par OpenCV, ZBar et ZXing-cpp ;
- treize scénarios de dégradation ;
- PostgreSQL en production, avec migrations Alembic et sauvegardes quotidiennes ;
- SQLite pour les tests et le développement local ;
- une base relationnelle contenant runs, tentatives, validations et qualité ;
- des exports JSON et CSV ;
- des métriques Prometheus, alertes et un dashboard Grafana ;
- une image Docker CUDA et des ressources Kubernetes adaptées au cluster Prooftag.

Les anciennes réparations locales, projections de modules, fusions FreeQR et
ControlNet alternatifs restent documentés, mais ils ne font plus partie des
profils actifs du Web Lab. Voir [`docs/e005-srpg.md`](docs/e005-srpg.md),
[`docs/e006-parameter-search.md`](docs/e006-parameter-search.md),
[`docs/e007-contextual-optimizer.md`](docs/e007-contextual-optimizer.md),
[`docs/e008-controlnet-bakeoff.md`](docs/e008-controlnet-bakeoff.md),
[`docs/research-roadmap.md`](docs/research-roadmap.md) pour le programme complet et
[`docs/experiment-log.md`](docs/experiment-log.md) pour les résultats, erreurs et décisions.

## Laboratoire Web

Le laboratoire est servi par la même API à l'adresse `/lab`. Il exécute
séquentiellement chaque combinaison recette × prompt × seed, conserve les
configurations et mesures dans PostgreSQL, puis permet de valider à la chaîne
chaque image : esthétique bonne/mauvaise, scan téléphone positif/négatif, note
et commentaire. SSR, payload exact, MER, CLIPScore et CLIP-aesthetic restent
séparés des verdicts humains. CLIP est calculé sur CPU pour réserver la VRAM à
la diffusion.

Depuis le PC Windows, lorsque le Deployment est prêt sur le serveur :

```powershell
.\scripts\lab-remote.ps1
```

Pour fermer le tunnel :

```powershell
.\scripts\lab-remote.ps1 -Stop
```

Le fonctionnement, les profils disponibles et les limites d'interprétation sont détaillés dans
[`docs/web-lab.md`](docs/web-lab.md).

## API

Lancer localement :

```bash
python -m pip install -e '.[dev]'
uvicorn prooftag_qr.api:app --reload --port 8080
```

Générer un QR de référence :

```bash
curl -X POST http://127.0.0.1:8080/v1/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "payload": "https://example.prooftag.test/t/abc123",
    "prompt": "engraved botanical illustration, high contrast",
    "backend": "qr",
    "error_correction": "H",
    "seed": 42
  }'
```

Endpoints principaux :

| Endpoint | Fonction |
|---|---|
| `POST /v1/generations` | Génération et validation |
| `GET /v1/generations` | Tableau des exécutions récentes |
| `GET /v1/generations/{id}` | Détails, tentatives et validations |
| `GET /v1/generations/{id}/image` | Image acceptée ou meilleur candidat |
| `GET /v1/generations/{id}/artifacts` | Liste des variantes et diagnostics sauvegardés |
| `POST /v1/generations/{id}/physical-validations` | Enregistrer un scan terrain |
| `GET /v1/generations/{id}/physical-validations` | Tableau des essais physiques |
| `GET /lab` | Interface du laboratoire comparatif |
| `GET /v1/lab/schema` | Profils et paramètres proposés |
| `POST /v1/lab/campaigns` | Créer et mettre en file une campagne |
| `GET /v1/lab/campaigns/{id}` | Progression, essais, images et notes |
| `GET /v1/lab/campaigns/{id}/results.csv` | Export complet de la campagne |
| `GET /v1/reports/summary` | Agrégats persistants |
| `GET /v1/reports/runs.csv` | Export pour analyse externe |
| `GET /metrics` | Métriques Prometheus |
| `GET /docs` | Documentation OpenAPI interactive |

## Principe d'acceptation

Une image est `accepted` uniquement lorsque :

1. tous les décodeurs disponibles lisent exactement le payload original ;
2. le taux de réussite sur l'ensemble des dégradations atteint
   `PROOFTAG_QR_VALIDATION_MIN_PASS_RATE` ;
3. aucune substitution de payload n'est tolérée.

Avec la configuration de production (`1.0`), toutes les combinaisons
décodeur/scénario doivent réussir. Une image rejetée reste dans l'historique pour permettre
l'analyse du modèle, mais ne doit pas être publiée par l'application appelante.

## Déploiement

Le projet ne déploie pas de second Prometheus, Grafana, Loki ou MinIO. Voir
[`docs/deployment.md`](docs/deployment.md) pour le déploiement K3s et la permutation avec
vLLM, [`docs/web-lab.md`](docs/web-lab.md) pour le laboratoire,
[`docs/metrics.md`](docs/metrics.md) pour le catalogue de mesures et
[`docs/benchmark.md`](docs/benchmark.md) pour générer et rapatrier un rapport comparatif en
une commande.

## Benchmark après une modification

Depuis le serveur, `make benchmark` génère six cas reproductibles, toutes les images et les
mesures comparatives. Depuis le PC Windows, `.\scripts\benchmark-remote.ps1` lance le même
benchmark à distance, copie l'archive, l'extrait et ouvre le rapport HTML.

La campagne causale E005 (baseline puis SRPG seul) s'exécute avec `make benchmark-e005` sur le
serveur ou `.\scripts\benchmark-remote.ps1 -E005` depuis Windows.

Cinq notebooks séparent maintenant clairement les usages. Le notebook
[`01_srpg_step_by_step.ipynb`](notebooks/01_srpg_step_by_step.ipynb) ne fait que relire une archive.
Le notebook [`02_generate_live_on_gpu.ipynb`](notebooks/02_generate_live_on_gpu.ipynb) exécute au
contraire toute la génération sur la RTX du serveur : diffusion brute, validation, pas SRPG
visibles en direct, réparations, validation de chaque candidate et sélection finale. Le navigateur
reste sur Windows grâce à un tunnel SSH. Le notebook
[`03_srpg_parameter_search.ipynb`](notebooks/03_srpg_parameter_search.ipynb) crible 17 profils,
confirme les trois meilleurs et conserve les validations automatiques et physiques.
Le notebook [`04_e007_contextual_optimizer.ipynb`](notebooks/04_e007_contextual_optimizer.ipynb)
ajoute la recherche TPE complète, CLIP-aesthetic, CLIPScore et un mini-modèle contextuel, avec
livraison interdite sous 26/26. Les commandes exactes sont dans
[`notebooks/README.md`](notebooks/README.md).

Le notebook [`05_controlnet_model_bakeoff.ipynb`](notebooks/05_controlnet_model_bakeoff.ipynb)
compare équitablement Dion, QR Code Monster v1/v2 et Nacholmo v2 avant de fixer le ControlNet
utilisé par E007.

## Tests

```bash
make install
make lint
make test
```

## Modèles et confidentialité

Les payloads ne sont jamais écrits en clair dans SQLite ou les logs. Seul leur SHA-256 est
persisté. Le QR contenu dans l'image reste naturellement décodable. Les prompts sont
conservés pour la reproductibilité des expériences ; ils ne doivent pas contenir de secret.
