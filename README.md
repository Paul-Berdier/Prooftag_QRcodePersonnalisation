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
- un QR binaire exact comme cible Stage 2, seule solution locale garantissant le payload ;
- aucune imitation visuelle de QArt : le transformateur Reed–Solomon du papier
  n’est pas publié dans le dépôt DiffQRCoder ;
- un SR-MPGD séparé conforme aux équations 13–14, avec validation de chaque état ;
- une géométrie entière QR v3/M/masque 4 conforme aux exemples publics ;
- aucune réparation déterministe ou superposition du QR témoin dans les profils du Web Lab ;
- des garde-fous de divergence et de saturation sans réparation cachée de l’image ;
- une validation automatique unique par `antfu/qr-verify@0.2.0` et son scanner
  WeChat WASM, avec contrôle exact du payload sur 37 presets déterministes ;
- CLIP-Aesthetic, CLIPScore et HPS v2.1 comme objectifs esthétiques secondaires ;
- E026, un conseiller prompt → paramètres entraîné sur les configurations réellement observées ;
- E026W, une collecte Kubernetes reprenable et bornée à sept jours ;
- E027, un holdout apparié de 300 contextes qui compare cascade, sélection complète et SR-MPGD
  forcé avec QR-Verify prioritaire ;
- E028, une cascade hiérarchique conseillée par prompt où Stage 1 n'est jamais livrable et où
  chaque Stage 2/SR-MPGD réutilise exactement son parent ;
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
et commentaire. Le score QR-Verify, sa lecture directe et la MER restent séparés
des verdicts humains. CLIP-Aesthetic, CLIPScore et HPS v2.1 sont conservés comme
mesures secondaires ; l'évaluation humaine reste la référence esthétique.

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

Une image est `accepted` lorsque `antfu/qr-verify` restitue le payload attendu
sur au moins un de ses 37 presets. Le preset original sans filtre est affiché
séparément. Le score QR-Verify correspond à la fraction de presets exacts parmi
ceux que le QR témoin supporte ; il sert au classement, pas de probabilité de
scan téléphone. Aucune substitution de payload n'est tolérée.

Le protocole, les versions épinglées et les limites sont documentés dans
[`docs/e024-qr-verify.md`](docs/e024-qr-verify.md). Les validations téléphone
restent enregistrées séparément. Une image rejetée reste dans l'historique pour
permettre l'analyse du modèle, mais ne doit pas être publiée par l'application
appelante.

E025 rétablit `clip_similarity`, `CLIPScore`, `CLIP-Aesthetic` et HPS v2.1
comme mesures d'image secondaires. Elles ne modifient jamais le verdict
QR-Verify. Le protocole et la commande de re-test sont dans
[`docs/e025-quality-scoring.md`](docs/e025-quality-scoring.md).

Le conseiller E026 et sa collecte autonome d'une semaine sont documentés dans
[`docs/e026-prompt-parameter-advisor.md`](docs/e026-prompt-parameter-advisor.md) et
[`docs/e026-week-unattended.md`](docs/e026-week-unattended.md).
Le notebook E026 exécute ensuite E026I : top-3 conseillé contre Stage 1 sur dix prompts inconnus,
avec reprise, images, QR-Verify, CLIP-Aesthetic, CLIPScore et HPS v2.1.

Le notebook E027 mesure séparément Stage 1, Stage 2 et SR-MPGD sur le même latent, puis rejoue
trois politiques de livraison sans permettre à l'esthétique de dépasser la porte QR. Protocole :
[`docs/e027-srmpgd-policy-holdout.md`](docs/e027-srmpgd-policy-holdout.md).

E028 corrige la politique de livraison : Stage 1 reste uniquement une source esthétique. Le
conseiller choisit les paramètres de Stage 1, de Stage 2 et de SR-MPGD pour chaque prompt, puis
compare une chaîne fixe, une chaîne top-1 et plusieurs chaînes conseillées. Protocole :
[`docs/e028-hierarchical-prompt-advisor.md`](docs/e028-hierarchical-prompt-advisor.md).

L'audit E028 a ensuite montré qu'un SR-MPGD ayant sélectionné l'itération zéro redécodait malgré
tout le latent Stage 2 et pouvait changer le raster. E029 impose une identité pixel pour pixel du
no-op, lie chaque campagne distante à son `plan_id`, refuse toute régénération silencieuse d'un
Stage 1 manquant et le vérifie sur 180 états appariés avant tout réentraînement SR-MPGD. Protocole :
[`docs/e029-srmpgd-exact-raster-recovery.md`](docs/e029-srmpgd-exact-raster-recovery.md).

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
