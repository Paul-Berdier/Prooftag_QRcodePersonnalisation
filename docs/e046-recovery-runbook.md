# E046 — reprise après incident

## Règle générale

Ne jamais supprimer `/data/e046-controlled-best-generator-v1`.

Relancer avec le même profil :

```bash
PROOFTAG_E046_PROFILE=smoke \
bash scripts/run-e046-controlled-campaign.sh resume
```

ou :

```bash
PROOFTAG_E046_PROFILE=pilot \
bash scripts/run-e046-controlled-campaign.sh resume
```

Le plan dépend du profil, du commit et du manifeste E045.

## Échec pendant un parent GPU

Les fichiers de tentative restent dans :

```text
<plan>/attempts/parents/
```

Un parent déjà promu avec `GENERATION_COMPLETE.json` est ignoré. Une tentative
incomplète n'est jamais publiée comme parent final.

## Échec pendant SR-MPGD

Chaque trajectoire constitue un Job indépendant. Les autres trajectoires
terminées restent intactes. Les images et latents i0..iN sont écrits avant le
scoring CPU.

Un OOM ne reçoit aucun retry Kubernetes (`backoffLimit: 0`). Après diagnostic,
modifier la recette ou les ressources dans un nouveau commit/plan plutôt que
relancer indéfiniment la même spécification.

## Échec HPS / QR-Verify

Le scoring est une phase CPU séparée. Relancer la même commande ne régénère pas
les images. Les candidats déjà dotés de `SCORING_COMPLETE.json` sont ignorés.

## Voir l'état

```bash
bash scripts/run-e046-controlled-campaign.sh status
```

Pendant un Job GPU, l'API reste arrêtée afin de laisser la RTX au Job. Le statut
affiche alors Job et Pod sans essayer de prendre le GPU.

```bash
bash scripts/run-e046-controlled-campaign.sh logs
```

Les logs ne sont pas suivis avec `-f`, ce qui évite de créer inutilement des
watchers.

## Timeout opérateur

Le script quitte après le timeout mais ne supprime pas le Job actif. Vérifier :

```bash
kubectl get job,pod \
  -n qr-core \
  -l prooftag.io/experiment=e046-controlled-best-generator-v1 \
  -o wide
```

Puis reprendre lorsque le Job est terminé.

## Vérification finale

```bash
bash scripts/run-e046-controlled-campaign.sh verify
```

La sortie valide doit contenir :

```json
{
  "missing": [],
  "mismatched": [],
  "valid": true
}
```


## Restauration des services après perte de session SSH

Si la connexion SSH disparaît pendant un Job actif, le script laisse
volontairement l'API à zéro pour ne pas reprendre la RTX au Job. Lorsque le Job
est terminé :

```bash
bash scripts/run-e046-controlled-campaign.sh restore-runtime
```

Le runtime QR revient à :

```text
API=1
notebook=0
vLLM=0
```

Une sortie normale ou un Job échoué déjà terminé déclenche cette restauration
automatiquement.


## Relance explicite d'un Job échoué

Un Job Kubernetes déjà en état `Failed` n'est pas supprimé et recréé
automatiquement. Le script affiche ses logs et s'arrête.

Pour un incident confirmé transitoire uniquement :

```bash
PROOFTAG_E046_RETRY_FAILED=1 \
PROOFTAG_E046_PROFILE=smoke \
bash scripts/run-e046-controlled-campaign.sh resume
```

Pour un OOM, un parent/latent absent, un checksum invalide ou une erreur de
configuration, ne pas utiliser ce drapeau : corriger le code ou la recette,
commiter, redéployer et créer un nouveau plan.
