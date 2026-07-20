# Déploiement sur `pcIA`

## Architecture retenue

| Composant | Emplacement |
|---|---|
| API et modèle | `Deployment/prooftag-qr`, namespace `qr-core` |
| Historique | PostgreSQL 16 dédié, PVC `local-path-retain` de 20 Gio |
| Migrations | InitContainer Alembic avant le démarrage de l'API |
| Sauvegardes SQL | `CronJob` quotidien, conservation locale de 30 jours |
| Images | PVC `prooftag-qr-data`, `local-path-retain` |
| Cache Hugging Face | PVC `prooftag-qr-model-cache`, `local-path-retain` |
| Séries temporelles | Prometheus existant, rétention 10 jours |
| Dashboard | Grafana existant via ConfigMap `grafana_dashboard=1` |
| Logs | stdout JSON, collecté par Promtail vers Loki |
| GPU | runtime `nvidia`, ressource exclusive `nvidia.com/gpu: 1` |
| Artefacts distants | MinIO existant, intégration optionnelle |

SQLite reste utilisé localement et dans les tests. PostgreSQL est imposé par la configuration
Kubernetes afin de permettre les requêtes analytiques, les migrations et l'accès ultérieur
en lecture seule depuis Grafana.

## Construction et import dans K3s

La construction ne nécessite pas d'arrêter vLLM :

```bash
cd ~/apps/Prooftag_QRcodePersonnalisation
docker build -t prooftag-qr:dev .
docker save prooftag-qr:dev | sudo k3s ctr images import -
sudo k3s ctr images list | grep prooftag-qr
```

Docker et K3s utilisent des magasins d'images distincts ; l'import est donc obligatoire en
l'absence de registre privé.

## Validation des manifests sans mutation

```bash
kubectl kustomize deploy/k8s > /tmp/prooftag-qr-rendered.yaml
kubectl apply --dry-run=server -f /tmp/prooftag-qr-rendered.yaml
```

## Secret de la base de données

Le mot de passe n'est jamais inscrit dans Git. Créer le Secret une seule fois :

```bash
bash scripts/create-database-secret.sh
kubectl get secret prooftag-qr-database -n qr-core
```

Le script conserve le Secret existant lors des exécutions suivantes. Ne le supprime pas tant
que le PVC PostgreSQL existe : recréer un autre mot de passe ne modifierait pas automatiquement
le mot de passe enregistré dans la base existante.

## Libérer le GPU puis déployer

```bash
bash scripts/gpu-workload.sh pause-vllm
kubectl apply -k deploy/k8s
kubectl rollout status statefulset/prooftag-qr-postgres -n qr-core --timeout=300s
kubectl rollout status deployment/prooftag-qr -n qr-core --timeout=900s
kubectl get pods,pvc,service -n qr-core
```

Le premier appel ControlNet télécharge les modèles dans le PVC et prendra nettement plus de
temps que les appels suivants. Suivre le démarrage :

```bash
kubectl logs -n qr-core deployment/prooftag-qr -f
kubectl port-forward -n qr-core service/prooftag-qr-svc 8080:8080
```

Dans un second terminal :

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/metrics
```

## Restituer le GPU à vLLM

Le pod QR doit d'abord être ramené à zéro, sinon Kubernetes conservera le GPU :

```bash
kubectl scale deployment/prooftag-qr -n qr-core --replicas=0
bash scripts/gpu-workload.sh resume-vllm
```

Pour reprendre les essais QR :

```bash
bash scripts/gpu-workload.sh pause-vllm
kubectl scale deployment/prooftag-qr -n qr-core --replicas=1
kubectl rollout status deployment/prooftag-qr -n qr-core --timeout=900s
```

Les PVC des deux applications restent montables et leurs caches ne sont pas supprimés.

## Sauvegardes PostgreSQL

Le CronJob `prooftag-qr-postgres-backup` produit chaque nuit un dump PostgreSQL au format
custom et supprime les dumps locaux de plus de 30 jours. Contrôler le premier dump :

```bash
kubectl create job --from=cronjob/prooftag-qr-postgres-backup \
  prooftag-qr-postgres-backup-manual -n qr-core
kubectl logs -n qr-core job/prooftag-qr-postgres-backup-manual
kubectl get jobs -n qr-core
```

Ce PVC reste sur le même serveur : il protège contre une erreur logique, pas contre la perte
du disque ou de la machine. Une copie périodique vers MinIO ou un stockage hors serveur devra
être ajoutée avant la production.

## MinIO optionnel

Le stockage local est activé par défaut. Pour utiliser MinIO, créer un bucket et un compte
dédiés, puis un Secret dans `qr-core` sans inscrire les identifiants dans Git. Ajouter ensuite
les variables suivantes au Deployment :

- `PROOFTAG_QR_ARTIFACT_STORE=s3`
- `PROOFTAG_QR_S3_ENDPOINT=http://minio.data-core.svc.cluster.local:9000`
- `PROOFTAG_QR_S3_BUCKET=prooftag-qr`
- `PROOFTAG_QR_S3_ACCESS_KEY`, depuis un Secret ;
- `PROOFTAG_QR_S3_SECRET_KEY`, depuis un Secret.

## Point de vigilance Prometheus

Le Prometheus existant n'a pas de stockage persistant et sa rétention est de dix jours. Les
mesures détaillées restent donc dans SQLite même après disparition des séries temporelles.
La persistance de Prometheus pourra être ajoutée ultérieurement au chart
`kube-prometheus-stack`, mais ce changement concerne toute la plateforme et n'est pas requis
pour le premier prototype.
