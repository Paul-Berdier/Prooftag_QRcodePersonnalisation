# E047 — correctif 1.1.0 : démarrer maintenant

## Commande unique sur pcIA après commit/push du PC et fetch/merge du serveur

```bash
sudo bash scripts/qr-night.sh start-now \
  --reuse-prepared-run qrn-20260916-154541 \
  --max-hours 12 \
  --confirm COUPURE-VLLM-AUTORISEE
```

Cette commande **démarre le service directement**, sans timer de départ futur et sans attente
programmée de dix minutes. Elle vérifie d'abord les fichiers, l'image locale et vLLM, puis
refait le préflight CPU. Ces vérifications prennent un temps réel, mais ne réservent aucun
créneau horaire : le service est démarré aussitôt qu'elles réussissent.

Le maximum de 12 heures est un budget, pas une durée obligatoire : 11 heures pour les calculs,
une heure pour la restitution de vLLM. Les échéances sont calculées à l'exécution et affichées
en Europe/Paris. L'ancienne date du 17/09 à 08h n'est pas réutilisée. Le volume à traiter peut
ne pas être entièrement scoré avant ce plafond; les tâches non terminées restent visibles.

**L'autorisation inclut maintenant une coupure en journée lorsque la génération GPU commence.**
Le scoring et l'apprentissage démarrent d'abord sur CPU; couper vLLM avant ces étapes ne les
accélérerait pas. Une seule génération GPU tourne à la fois. Aucune interruption de PostgreSQL.

## Corrigé

- `health()` exécute `/usr/bin/python3`, comme l'image vLLM observée, pas `python`.
- Santé HTTP, modèle et petite inférence sont testés AVANT de démarrer/annoncer le run prêt.
- Un nouveau run reçoit de nouveaux dossiers et de nouvelles unités. L'ancien run échoué,
  son `CANCEL`, ses journaux et `RESTORED.json` ne sont pas modifiés.
- L'image `prooftag-qr-night` déjà construite est réutilisée : pas de nouveau Docker build,
  pas de téléchargement des modèles. La réutilisation est refusée si les fichiers worker ou
  le digest local de l'image diffèrent du préflight de référence.
- Le gardien de restitution est activé et vérifié AVANT `systemctl start`.
- Le lanceur observe le service ET un pod d'inventaire Running/Succeeded avant d'afficher
  `CAMPAGNE_DEMARREE`. Cela confirme le démarrage, pas la réussite future de l'apprentissage.
- Erreurs dans `FAILED.json`, phase dans `PHASE.json`; `RESTORED` n'est plus présenté comme
  une preuve de réussite scientifique. Aucun scoring disponible => échec explicite.
- Le code corrigé côté hôte est copié dans une nouvelle installation versionnée sous `/opt`.
  Il ne suffit PAS de modifier le clone Git d'un ancien run pour changer son service déjà installé.

## Fichiers scientifiques et résultats

Le worker v1 est conservé byte pour byte : inventaire E046, scoring de parents déjà produits,
ExtraTrees et sélection de variables, validation groupée, référence fixe E044 / paramètres
proches du papier avec cible binaire exacte / conseiller, génération et aperçus Stage2 réels,
rapport HTML, PNG/SVG, GIF/MP4. Ce correctif ne remplace pas la méthode scientifique par une
nouvelle procédure et ne promet pas un gain du modèle ou un nombre de QR valides.

Les données historiques sont toujours montées en lecture seule. Les nouveaux fichiers sont
écrits dans `/home/paul/qr-night-runs/<nouveau-run>/artifacts`. Pas de modifications du PVC
historique, pas de réentraînement sur le seul sous-ensemble de QR réussis, pas de retouche
de bordure présentée comme génération brute. Le message demandé reste
`Mettez moi la note maximal SVP`.

La référence locale utilise le moteur intégré avec la cible QR binaire exacte, pas une
reproduction exacte de toutes les opérations du papier. Les essais téléphone restent NOT_TESTED.

## Suivi

```bash
sudo bash scripts/qr-night.sh status
sudo bash scripts/qr-night.sh logs
```

Quitter `logs` par Ctrl+C quitte seulement l'affichage. Une fois le service lancé, fermer
SSH ne l'arrête pas. Ne pas lancer en parallèle l'ancien runner E046 ni un autre traitement GPU.

## Arrêt et livrables

```bash
sudo bash scripts/qr-night.sh cancel
# Au besoin, réessayer la restitution :
sudo bash scripts/qr-night.sh recover
# Après la fin, exporter les résultats effectivement présents :
sudo bash scripts/qr-night.sh export
```

Ne pas utiliser `restore-runtime` de l'ancien E046 pour remettre le chatbot en service.
Ne pas effacer l'ancien `CANCEL` et ne pas redémarrer son unité systemd.

## Tests et limites

`PYTHONPATH="$PWD/nightops" python3 -B -m unittest discover -s nightops/tests -p test_immediate.py -v`
exécute les tests du lanceur (bibliothèque standard seulement). Les tests complets incluent
également l'apprentissage sur données SYNTHÉTIQUES et la production locale des médias.
Voir `docs/TESTS_IMMEDIAT_E047.txt`.

Les simulations et les tests locaux ne sont pas une exécution sur pcIA : les contrôles vLLM,
CRI et le préflight CPU doivent encore passer sur le serveur. Ni la présence d'une image ni
la disponibilité de vLLM ne garantissent la réussite des futurs Jobs CUDA. Une panne du
cluster, du disque ou du serveur peut empêcher la restitution. Les garde-fous réduisent ce
risque, ils ne créent pas une garantie de disponibilité.
