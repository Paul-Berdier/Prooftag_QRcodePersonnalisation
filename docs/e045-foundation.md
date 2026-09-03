# E045 — fondation du générateur adaptatif et reprise sûre

## Objet

E045 ne lance ni une nouvelle génération de QR, ni un entraînement. Il prépare les
deux conditions qui manquent avant de recommencer :

1. **une base historique canonique**, où les expériences E000–E044 sont indexées,
   dédupliquées, classées et reliées à leurs artefacts réels ;
2. **un contrat de reprise**, réutilisable par les futures générations E046 et les
   entraînements E047/E048.

La création d'un conseiller plus grand sans ces deux fondations répéterait les
incidents déjà observés : parents Stage 2 manquants, no-op SR-MPGD comptés comme
recettes distinctes, scores absents ignorés, labels contradictoires, OOM relancées
à l'identique et holdouts réutilisés pour entraîner.

## Sortie

Par défaut :

```text
/data/e045-foundation-v1/<plan-id>/
├── plan.json
├── state.sqlite
├── foundation.sqlite
├── experiment-registry.json
├── parameter-space.json
├── artifact-inventory.csv
├── canonical-observations.csv
├── canonical-observations.jsonl
├── duplicate-images.csv
├── label-conflicts.csv
├── srmpgd-noop-images.csv
├── phone-labels/
├── resilience-selftest/
├── data-card.json
├── recovery-runbook.json
├── summary.json
├── report.md
├── artifact-manifest.json
└── COMPLETE.json
```

`/data/e045-foundation-v1/LATEST.json` pointe vers le dernier plan.

## Ce que l'inventaire fait

Le runner parcourt les artefacts de recherche sous `/data` sans modifier les
sources. Il exclut les caches et sa propre sortie. Pour chaque fichier utile il
conserve :

- chemin absolu et relatif ;
- expérience déduite ;
- taille et date ;
- SHA-256 lorsque la taille autorise le calcul ;
- hash des pixels RGB, dimensions et mode des images ;
- erreur isolée si un fichier est illisible.

Les gros checkpoints restent indexés même si leur SHA n'est pas recalculé. La
limite est explicite dans `plan.json`.

## Observations canoniques

Les CSV/JSON/JSONL historiques sont convertis vers un schéma commun :

- expérience, méthode et étape ;
- texte/hash du prompt et famille ;
- hash/longueur du payload ;
- seed traitée comme catégorie ;
- configuration effective et son hash ;
- image et hash raster ;
- état technique ;
- QR-Verify, original exact et métriques d'image ;
- captures téléphone lorsqu'elles existent ;
- politique d'utilisation du registre ;
- éligibilité conseiller, surrogate téléphone, hard negative ou évaluation seule.

Le record canonique ne remplace pas le fichier source. Il conserve toujours
`source_path`, `source_record_index` et `source_record_hash`.

## Quarantaine et holdouts

Les données restent visibles, mais ne sont pas toutes entraînables :

- E016 invalidé : quarantaine ;
- E002/E004/E005/E011/E032 : hard negatives ou méthodologie seulement ;
- E021/E022/E027–E031 : évaluation seule ;
- E044 : apprentissage logiciel possible après audit, mais aucun label téléphone
  positif n'est créé à partir de QR-Verify ;
- un hash de pixels portant des labels contradictoires est exclu jusqu'à résolution ;
- un SR-MPGD pixel-identique à Stage 2 est marqué no-op.

## Quiet zone

E045 verrouille le changement de politique :

```text
interdit :
image terminée → effacement d'un anneau → marge artificielle

cible :
composition plus grande → région claire créée dès Stage 1
→ QR et sa zone calme intégrés à la scène
```

Les quatre modules de marge nécessaires restent une contrainte de conception.
Ils ne justifient pas de supprimer une partie de l'image finale.

## Autorisations

Même avec `COMPLETE.json`, E045 laisse :

```text
advisor_training_authorized=false
phone_surrogate_training_authorized=false
generation_campaign_authorized=false
production_ready=false
```

Ces portes ne changent qu'après revue humaine du notebook, résolution des conflits
et import de captures téléphone.


## Import téléphone après la fondation

Un plan `COMPLETE` reste immuable. Les captures ajoutées ensuite sont importées dans :

```text
/data/e045-foundation-v1/phone-imports/<hash-du-csv>/
```

Puis `PHONE_LATEST.json` pointe vers le dernier import valide. Le notebook préfère
ce pointeur sans réécrire le plan historique :

```bash
python scripts/e045-import-phone-captures.py \
  /data/e045-phone-captures.csv \
  --output-root /data/e045-foundation-v1
```
