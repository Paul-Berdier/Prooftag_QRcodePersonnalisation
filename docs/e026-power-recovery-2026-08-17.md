# E026 — récupération après la coupure électrique du 17 août 2026

## Constat factuel

Le PVC et PostgreSQL ont survécu. L'inventaire effectué après le redémarrage montre :

- plan `02b9402fba845d79` et payload hash inchangé ;
- échéance de collecte dépassée le 14 août 2026 à 01:24 UTC ;
- 31 exports CSV et 14 880 lignes brutes sur le PVC ;
- 77 campagnes E026 en base : 60 terminées, 13 interrompues, 3 annulées et une terminée avec
  erreurs ;
- des campagnes complètes au moins jusqu'au lot 16 et un lot 17 partiel ;
- aucun modèle `prooftag-e026-parameter-advisor.joblib` encore produit.

Le `state.json` n'est pas une source fiable pour cet incident : il indique seulement six lots
terminés alors que PostgreSQL en contient davantage. Il ne doit pas servir à supprimer ou à
relancer des données.

## Cause

Le notebook et le Job Kubernetes E026 ont été actifs sur le même plan. Les deux runners ont
partagé le même dossier d'état et ont soumis des reprises simultanées. La coupure a ensuite
redémarré l'API, qui marque volontairement les campagnes actives `interrupted` parce que le
payload en clair n'est pas persisté en base.

Les lignes restent récupérables, mais plusieurs campagnes représentent le même triplet
`prompt/recette/seed`. Compter les 14 880 lignes brutes comme 14 880 observations indépendantes
biaiserait le conseiller.

## Décision de récupération

1. ne lancer aucune nouvelle génération ;
2. sélectionner en base toutes les campagnes E026 ayant le même hash de payload ;
3. réexporter les campagnes terminales vers `exports-recovered` ;
4. conserver une seule observation logique par `(payload, prompt, configuration, seed, ECC)`, en
   privilégiant celle qui possède QR-Verify et le plus grand nombre de métriques ;
5. entraîner et valider le conseiller uniquement sur ce dataset dédupliqué ;
6. conserver les CSV bruts et `recovery-summary.json` comme preuve d'audit.

## Prévention

`WeekCampaignRunner` prend désormais un verrou exclusif `runner.lock` sur le PVC pendant son
exécution. Un second notebook ou Job visant le même plan s'arrête explicitement au lieu de lancer
des campagnes concurrentes. Le verrou système est automatiquement libéré après un crash ou une
coupure électrique.
