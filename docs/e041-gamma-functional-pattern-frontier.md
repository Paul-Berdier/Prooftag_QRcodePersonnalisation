# E041 — Gamma et motifs fonctionnels

E041 répond à deux questions laissées ouvertes par E040 :

1. `gamma=1000` était-il réellement utile ou seulement un contrôle historique ?
2. le faible SSR vient-il en partie de motifs fonctionnels (finder/timing/alignment/format) trop discrets ?

## Nouveau prompt

E041 ne réutilise pas le parent greenhouse. Il génère un parent frais avec :

> a sunlit botanical reading room inside a glass conservatory, oak shelves, climbing vines, terracotta pots and a small writing desk, refined editorial interior photograph

Seed `71041`, payload `https://ptag.io/t/e041`.

Ce changement est volontaire. **Les scores E041 ne sont donc pas appariés aux scores E040.** E040 reste un contrôle historique uniquement. À l'intérieur d'E041, tous les gamma partagent exactement le même parent frais.

## Phase A — gamma

- loss : `scanaware_v2` identique E039/E040 ;
- radius latent : `0.20` ;
- LPIPS weight : `0.01` ;
- checkpoints : `i0..i8` ;
- gamma : `50, 100, 250, 500, 1000, 2000`.

`1000` est présent pour continuité historique mais n'est plus imposé.

Le runner enregistre aussi `raw_step_rms`, `projected_step_rms`, `accepted_alpha` et indique si la projection de trust-region a effectivement été active. Cela permet de savoir si un gros gamma est simplement écrasé par la projection/backtracking.

## Phase B — motifs fonctionnels

Pour limiter le coût GPU, E041 prend les trois meilleurs checkpoints Phase A, provenant de gamma distincts, puis applique les six facteurs :

`0.00, 0.05, 0.10, 0.15, 0.20, 0.30`.

Ce traitement appelle le mécanisme existant `prepare_scan_ready_image(... functional_pattern_tone_factor=...)` : seuls les motifs fonctionnels sont renforcés ; les data modules ne sont jamais projetés sur l'image.

## Sélection

Priorité :

1. visual guard PASS ;
2. QR-Verify exact `/37` ;
3. original exact ;
4. erreur centre des motifs fonctionnels ;
5. MER ;
6. LPIPS.

CLIPScore, CLIP-Aesthetic, HPS, clipping RGB/pixels et saturation restent audités. L'advisor est enregistré comme recommandation seulement. E016 n'est utilisé que s'il est `research_usable` et ne remplace jamais QR-Verify.

## Sorties

`/data/e041-gamma-functional-pattern-frontier-v1`

avec parent Stage 1/2, 54 checkpoints Phase A, 18 variantes Phase B, preuves QR-Verify, quality scores, traces, `gamma-projection-summary.json`, verdict et `pipeline/99-FINAL-QR.png`.

`production_ready=false` et `generalization_authorized=false` restent obligatoires.
