# E021 — généralisation sur prompts atypiques

Date : 4 août 2026.

## Point de départ E020

La campagne `d36b2724` confirme l’appariement : SRPG, SR-MPGD officiel et
SR-MPGD robuste partagent le latent
`fe92b169fdb13dd2c6548dfaf0a80636c24448ee4e0a8e3861dadbd603dcd6c2`.
Le SSR reste à `1/39` pour les trois sorties. La loss officielle vaut zéro,
alors que la loss robuste atteint `0,0154397`, mais aucune des quatre
itérations n’améliore le SSR et l’état zéro reste sélectionné.

Cette campagne ne prétend donc pas résoudre la loss. Elle mesure si ce constat
reste vrai en dehors des quatre familles de prompts historiques.

## Jeux de prompts

Le lot `core` contient six sujets inédits : coupe de nautile, colibri mécanique,
bibliothèque en apesanteur, archipel tissé, marais salants fractals et serre
lunaire brutaliste.

Le lot `stress` force des cas difficiles : image presque blanche, scène presque
noire, op art répétitif, macro iridescente, pose longue et diorama extrêmement
dense.

Chaque prompt compare quatre sorties appariées : Stage 1, SRPG, SR-MPGD
officiel et SR-MPGD robuste. Le QR témoin n’est pas dupliqué par défaut, car la
validation calcule déjà sa capacité pour chaque résultat. L’option
`--include-reference` permet néanmoins de l’afficher.

## Lancement progressif

Commencer par les six prompts `core`, soit 24 essais :

```bash
python scripts/e021-atypical-prompts.py \
  --api http://127.0.0.1:18080 \
  --payload 'https://ptag.io/t/e021' \
  --set core \
  --launch
```

Ne lancer `stress` qu’après export et inspection du premier CSV :

```bash
python scripts/e021-atypical-prompts.py \
  --api http://127.0.0.1:18080 \
  --payload 'https://ptag.io/t/e021' \
  --set stress \
  --launch
```

Deux seeds ne sont justifiés qu’après ces deux lots. Ils doublent le coût et se
passent avec `--seeds 51001,52001`.
