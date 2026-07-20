# Baselines versionnées

Ce dossier contient les résultats de référence retenus pour décider si une modification est
conservée. Les archives complètes, trop volumineuses pour Git, restent dans
`benchmark-results/` sur le serveur et dans `Downloads/prooftag-benchmarks` sur le poste.

Une baseline versionnée doit indiquer le commit, le protocole, le matériel, les paramètres,
les résultats agrégés et les anomalies connues. Un taux de livraison mesuré sur six cas ne
constitue jamais une preuve de robustesse terrain.

| Identifiant | Commit | État | Usage |
|---|---|---|---|
| `2026-07-20-83641dff` | `83641dff` | mesuré sur GPU | référence avant régénération et réparations arrondies |

À partir du protocole 2.0, `summary.json` enregistre également la version du protocole, le hash
des cas et tous les paramètres de génération.
