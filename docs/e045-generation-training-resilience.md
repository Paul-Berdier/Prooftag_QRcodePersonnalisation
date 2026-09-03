# E045 — contrat de résilience génération et entraînement

## Pourquoi un contrat spécifique

Une génération GPU peut échouer après Stage 1, pendant Stage 2, pendant SR-MPGD
ou pendant le scoring. Un entraînement peut échouer après plusieurs heures. La
reprise ne doit ni recommencer inutilement, ni réutiliser silencieusement un état
incompatible.

`prooftag_qr.resilient_experiment` fournit le contrat commun.

## Identité d'une tâche

Une tâche est définie par :

```text
run_id
task_id
kind
spec_json canonique
spec_hash SHA-256
source_commit
max_attempts
artifact_dir
```

Changer le batch, le modèle, le dataset, le seed, un paramètre de génération ou
une ressource scientifique crée une **nouvelle spécification**. Le même `task_id`
ne peut pas être réutilisé avec un autre hash.

## États

```text
pending
  ↓ claim transactionnel
running + lease + heartbeat
  ├─ succeeded
  ├─ retry_wait     (transitoire et budget restant)
  ├─ failed         (terminal non-opérateur)
  └─ blocked        (OOM, disque, contrat, checksum, entrée absente)
```

Un worker disparu laisse expirer son lease. La reprise transforme cette tâche en
`pending` uniquement si son budget de tentatives n'est pas épuisé.

## Classification

### Transitoire — reprise bornée

Exemples :

- timeout réseau ;
- HTTP 429/500/502/503/504 ;
- connexion réinitialisée ;
- pod évincé ou nœud perdu.

### Ressource — blocage opérateur

Exemples :

- CUDA OOM ;
- mémoire système épuisée ;
- disque plein ou quota ;
- trop de fichiers ouverts ;
- watchers inotify épuisés.

Un OOM n'est **jamais** relancé avec le même batch, la même précision et les mêmes
ressources. L'opérateur crée une nouvelle spécification.

### Déterministe — correction obligatoire

Exemples :

- schéma ou dimension invalide ;
- parent/latent manquant ;
- payload ou checksum différent ;
- assertion d'appariement ;
- configuration impossible.

### Inconnu

Une seule reprise prudente est autorisée. Un second échec bloque la tâche.

## Tentatives et promotion

```text
attempts/<task-id>/<attempt-no>/
    checkpoint
    logs
    métriques
    sorties provisoires
```

Une tentative n'est promue vers `final/<task-id>` qu'après :

1. présence de tous les fichiers requis ;
2. validation du payload et des hashes ;
3. écriture et fsync du manifeste ;
4. `os.replace` atomique.

Un dossier final existant n'est jamais écrasé.

## Checkpoint d'entraînement obligatoire

Le futur E047 doit enregistrer atomiquement :

```text
model_state
optimizer_state
scheduler_state
gradient_scaler_state
epoch
global_step
best_metric
RNG Python / NumPy / Torch CPU / CUDA
dataset_hash
split_hash
architecture_hash
source_commit
runtime_image_digest
```

Une reprise est refusée si l'un des quatre hashes scientifiques change.

## Checkpoint de génération obligatoire

Chaque contexte E046 doit pouvoir reprendre à la frontière de :

```text
QR/blueprint validé
Stage 1 validé
Stage 2 + latent validés
chaque checkpoint SR-MPGD validé
scoring logiciel terminé
captures téléphone importées
sélection Pareto terminée
```

Un fichier d'étape n'est pas considéré terminé sans son manifeste.

## Échecs dans le dénominateur

Une erreur technique reste une observation du plan. Les rapports publient :

- résultats planifiés ;
- résultats générés ;
- erreurs techniques par catégorie ;
- taux sur le plan complet ;
- taux sur les sorties générées, uniquement comme diagnostic.

Aucune moyenne pandas ne doit ignorer silencieusement les lignes absentes.

## Commandes de reprise

Même commit, même plan :

```bash
bash scripts/run-e045-foundation.sh
```

Le script reprend les tâches incomplètes.

Après correction de code :

- conserver l'ancien plan en lecture seule ;
- construire une nouvelle image/commit ;
- créer un nouveau plan ou un finalizer explicitement versionné ;
- importer seulement les artefacts dont les contrats passent.

Il est interdit de supprimer automatiquement `/data/e045-*`.
