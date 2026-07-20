# Déploiement sur `pcIA`

## Architecture retenue

| Composant | Emplacement |
|---|---|
| API et modèle | `Deployment/prooftag-qr`, namespace `qr-core` |
| Historique SQLite et images | PVC `prooftag-qr-data`, `local-path-retain` |
| Cache Hugging Face | PVC `prooftag-qr-model-cache`, `local-path-retain` |
| Séries temporelles | Prometheus existant, rétention 10 jours |
| Dashboard | Grafana existant via ConfigMap `grafana_dashboard=1` |
| Logs | stdout JSON, collecté par Promtail vers Loki |
| GPU | runtime `nvidia`, ressource exclusive `nvidia.com/gpu: 1` |
| Artefacts distants | MinIO existant, intégration optionnelle |

SQLite est adapté à un unique worker GPU sur ce cluster mono-nœud. Une migration vers
PostgreSQL sera nécessaire avant plusieurs réplicas ou plusieurs nœuds.

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

## Libérer le GPU puis déployer

```bash
bash scripts/gpu-workload.sh pause-vllm
kubectl apply -k deploy/k8s
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

