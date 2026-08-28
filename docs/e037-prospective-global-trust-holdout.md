# E037 — prospective global-trust mini-holdout

E037 is the preregistered generalization check authorized by E036.

## Frozen method

The method is not tuned on holdout results:

- E036 research winner: `e036_gamma1000_global_trust`;
- raw proposal gamma: `1000`;
- official DiffQRCoder SRL revision: `e24ea73ee2e13c7e6e87cb422e8b11784e70ae00`;
- latent RMS trust radius: `0.050`;
- LPIPS budget: `0.050`;
- core MAE budget: `0.050`;
- LPIPS soft weight: `0.01`;
- four recorded updates;
- QR version 3, mask 4, module size 20, padding 78, EC M.

## Holdout

Ten fixed prompt/seed pairs cover different visual domains: courtyard, station, wine cellar,
alpine cabin, ramen shop, botanical library, lighthouse, workshop, Paris cafe and vineyard.
All use the fixed payload `https://ptag.io/t/e037` so the QR geometry remains identical.

Each case generates a fresh DiffQRCoder Stage 1 and public SRPG Stage 2 parent before the
frozen E036 trust-region refinement is applied.

## Evidence

For every case E037 stores:

- Stage 1 PNG;
- Stage 2 parent PNG and latent safetensors;
- four-update E036-global trace;
- final PNG and latent;
- side-by-side parent/final comparison;
- conservative QR-Verify evidence (37 presets x 3 repetitions);
- module error, upstream margin, LPIPS, core-MAE and latent-RMS metrics.

The notebook displays the global contact sheet and all ten side-by-side comparisons.

## Preregistered decision

- any visual-budget violation -> `STOP_VISUAL_BUDGET_GENERALIZATION_FAILURE`;
- at least 5/10 final cases with >=1 conservative exact preset ->
  `GENERALIZES_PREPARE_E038_SCANNER_ROBUSTNESS`;
- otherwise at least one exact case or >=7/10 module-error improvements ->
  `PARTIAL_GENERALIZATION_PREPARE_E038_SCANNER_AWARE_TRUST`;
- otherwise -> `NO_GENERALIZATION_REVISIT_OBJECTIVE`.

E037 never marks a result production-ready and never authorizes automatic expansion.
