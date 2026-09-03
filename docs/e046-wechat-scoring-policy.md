# E046 — politique de score QR

## Gate de validité automatique

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

## Objectifs visuels et sémantiques

Après le gate WeChat, le gagnant de chaque prompt est choisi automatiquement avec :

```text
40 % robustesse WeChat
25 % CLIPScore / respect du prompt
20 % HPSv2
15 % CLIP-Aesthetic
```

Les métriques suivantes restent aussi enregistrées comme diagnostics ou contraintes :

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

Aucun mélange de votes OpenCV/ZBar/ZXing n'est utilisé. La beauté ne peut jamais compenser un QR qui échoue au gate WeChat.

## Vérité finale

```text
téléphone réel > WeChat logiciel > diagnostics structurels
```

E046 ne possède aucun label téléphone. Toute sortie garde :

```text
phone_truth_available=false
production_ready=false
```
