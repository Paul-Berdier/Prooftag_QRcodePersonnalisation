# E038 — SR-MPGD SSR / aesthetic frontier search

## Question

Find the SR-MPGD recipe that maximizes software scan success (conservative QR-Verify exact presets)
while preserving the image. E038 is **not** a holdout/generalization experiment.

## Fixed elements

- exact immutable E035 Stage-2 parent latent;
- pinned DiffQRCoder revision `e24ea73ee2e13c7e6e87cb422e8b11784e70ae00`;
- QR version 3, mask 4, module size 20, crop/padding 78;
- FP32 VAE working copy;
- four updates;
- `gamma = 1000` for every raw proposal;
- LPIPS VGG soft weight `0.01`;
- QR-Verify 37 presets × 3 repetitions;
- no best-state oracle: final recorded state is i4.

## Same-parent controls shown visually

1. parent FP32;
2. E035 paper loss;
3. E035 official-upstream loss without trust region;
4. E036 global trust `r=.050`;
5. E036 strict trust `r=.025`;
6. E036 local-preserve `r=.050`.

E033/E034 are shown only as numeric history because E034's historical parent reproduction was not
byte-identical to the later frozen parent.

## Phase A — trust-region radius

Official upstream SRL only, all other settings fixed:

- `r=.075`
- `r=.100`
- `r=.125`
- `r=.150`
- `r=.200`
- `r=.300`

This fills the missing experimental space between E036 global (`.050`) and E035 unbounded
(approximately `.442` final latent RMS in the executed single case).

## Phase B — new QR objectives

At moderate radii, compare:

- `full_r100`: upstream centre SRL + weak full-module luminance margin;
- `robust_r100`: upstream SRL + differentiable blur/downscale/brightness/contrast terms;
- `hybrid_r100`: full-module + robust scan terms;
- `hybrid_r150`: same hybrid objective with a larger latent region.

The robust terms are intentionally weak additions. They do not replace the official upstream SRL.

## Metrics

Every new final candidate records:

- conservative QR-Verify exact presets and SSR `/37`;
- original/direct decode when evidence exposes it;
- upstream SRL;
- full-module error count/rate;
- upstream active module count;
- LPIPS;
- latent delta RMS;
- changed-pixel ratio and mean absolute change;
- clipped-pixel and RGB-channel clipping changes;
- saturation change;
- CLIPScore;
- CLIP-Aesthetic;
- HPS v2.1;
- final native PNG and complete per-iteration trace.

## Ranking

No scalar score mixes aesthetics and scanning. Ranking is lexicographic:

1. candidate must pass the preregistered visual guards;
2. maximize conservative QR-Verify exact presets;
3. prefer an exact original/direct decode;
4. minimize full-module errors;
5. minimize LPIPS, then latent displacement.

E035 upstream unbounded remains visible as a high-SSR negative aesthetic control even if it cannot
win.

## Visual guards for research ranking

- LPIPS <= 0.05;
- mean absolute image change <= 0.08;
- clipped pixel ratio increase <= 0.005;
- RGB clipped-channel ratio increase <= 0.005;
- absolute saturation-mean change <= 0.08;
- high-saturation ratio increase <= 0.05;
- CLIPScore drop <= 0.03;
- CLIP-Aesthetic drop <= 0.25;
- HPS v2.1 drop <= 0.02 when available.

These are research guards, not production acceptance criteria.

## Output

The notebook `33_e038_srmpgd_ssr_aesthetic_frontier.ipynb` presents one table for all methods,
a large contact sheet, every native final image, SSR/LPIPS and SSR/radius plots, and the complete
trace of the selected research winner.
