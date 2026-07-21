# Prooftag QR esthétique

Socle expérimental et opérationnel pour générer des QR codes intégrés à des images,
mesurer leur qualité et ne publier que les résultats validés.

## État actuel

La version `0.1.0` fournit :

- une API FastAPI ;
- un générateur QR de référence (`backend=qr`) ;
- une baseline artistique Stable Diffusion 1.5 + ControlNet (`backend=controlnet`) ;
- une seconde diffusion img2img guidée par les modules QR incorrects, expérimentale et
  désactivée par défaut ;
- une vraie boucle DDIM/SRPG différentiable à 40 pas, avec SRL + LPIPS et garde-fous VRAM/QR,
  expérimentale et désactivée par défaut ;
- un verrouillage des motifs fonctionnels et une réparation adaptative des modules incorrects
  ou peu contrastés, d'abord par luminance et formes arrondies fondues dans l'illustration,
  puis par profils binaires de secours ;
- une régénération avec une nouvelle seed avant tout fallback de réparation globale ;
- plusieurs tentatives avec conservation automatique du meilleur résultat ;
- une validation exacte du payload par OpenCV et ZBar ;
- treize scénarios de dégradation ;
- PostgreSQL en production, avec migrations Alembic et sauvegardes quotidiennes ;
- SQLite pour les tests et le développement local ;
- une base relationnelle contenant runs, tentatives, validations et qualité ;
- des exports JSON et CSV ;
- des métriques Prometheus, alertes et un dashboard Grafana ;
- une image Docker CUDA et des ressources Kubernetes adaptées au cluster Prooftag.

La réparation locale actuelle conserve teinte et texture, remplace les centres carrés visibles
par des superellipses à bords progressifs et constitue le garde-fou structurel de secours :
chaque variante est validée et, parmi celles qui atteignent le seuil strict, la moins modifiée
par rapport à l'image artistique est livrée.

La campagne E004 a rejeté la seconde diffusion localisée : elle augmentait l'erreur QR et le
nombre de pixels réparés. E005 implémente désormais le gradient SRPG dans chaque timestep DDIM,
teste sa sortie avec tous les décodeurs et garde la chaîne brute comme secours. Tous les modes de
recherche sont désactivés par défaut. Voir [`docs/e005-srpg.md`](docs/e005-srpg.md),
[`docs/research-roadmap.md`](docs/research-roadmap.md) pour le programme complet et
[`docs/experiment-log.md`](docs/experiment-log.md) pour les résultats, erreurs et décisions.

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
| `POST /v1/generations/{id}/physical-validations` | Enregistrer un scan terrain |
| `GET /v1/generations/{id}/physical-validations` | Tableau des essais physiques |
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
vLLM, [`docs/metrics.md`](docs/metrics.md) pour le catalogue de mesures et
[`docs/benchmark.md`](docs/benchmark.md) pour générer et rapatrier un rapport comparatif en
une commande.

## Benchmark après une modification

Depuis le serveur, `make benchmark` génère six cas reproductibles, toutes les images et les
mesures comparatives. Depuis le PC Windows, `.\scripts\benchmark-remote.ps1` lance le même
benchmark à distance, copie l'archive, l'extrait et ouvre le rapport HTML.

La campagne causale E005 (baseline puis SRPG seul) s'exécute avec `make benchmark-e005` sur le
serveur ou `.\scripts\benchmark-remote.ps1 -E005` depuis Windows.

Deux notebooks séparent maintenant clairement les usages. Le notebook
[`01_srpg_step_by_step.ipynb`](notebooks/01_srpg_step_by_step.ipynb) ne fait que relire une archive.
Le notebook [`02_generate_live_on_gpu.ipynb`](notebooks/02_generate_live_on_gpu.ipynb) exécute au
contraire toute la génération sur la RTX du serveur : diffusion brute, validation, 40 pas SRPG
visibles en direct, réparations, validation de chaque candidate et sélection finale. Le navigateur
reste sur Windows grâce à un tunnel SSH. Les commandes exactes sont dans
[`notebooks/README.md`](notebooks/README.md).

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
