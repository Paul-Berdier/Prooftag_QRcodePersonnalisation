# E046 — politique de score QR

## Principal

```text
engine = qr-scanner-wechat via qr-verify@0.2.0
payload exact
37 presets
3 répétitions conservatrices
```

Champs :

```text
wechat_exact_presets
wechat_exact_rate
wechat_original_exact
wechat_repetitions
wechat_engine
```

## Secondaires

Les métriques suivantes sont diagnostiques ou contraintes :

```text
module_error_rate
full_module_error_count
finder/timing/alignment diagnostics
CLIPScore
CLIP-Aesthetic
HPSv2
LPIPS
clipping
saturation
quiet-zone luminance/texture
```

Aucun mélange de votes OpenCV/ZBar/ZXing n'est utilisé comme cible.

## Vérité finale

```text
téléphone réel > WeChat logiciel > diagnostics structurels
```

E046 ne possède aucun label téléphone. Toute sortie garde :

```text
phone_truth_available=false
production_ready=false
```
