from __future__ import annotations
import json
from pathlib import Path

OUT = Path('notebooks/39_e042_decoder_failure_localization.ipynb')

def md(text: str):
    return {'cell_type':'markdown','metadata':{},'source':[line+'\n' for line in text.strip().splitlines()]}

def code(text: str):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in text.strip().splitlines()]}

cells = [
md('''# E042 — Localisation de l’échec du décodeur

Cette expérience **ne génère pas un nouveau QR**. Elle réutilise 9 latents E041 et localise l’échec entre détection, quiet zone, binarisation, reconstruction de grille et bits/ECC. QR-Verify reste la référence. Les reconstructions target-assisted sont uniquement diagnostiques.'''),
code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image as IPImage, Markdown
RESULTS_DIR = Path('/data/e042-decoder-failure-localization-v1')
verdict_path = RESULTS_DIR / 'verdict.json'
if not verdict_path.is_file():
    raise FileNotFoundError(f'E042 non terminé: {verdict_path}')
verdict = json.loads(verdict_path.read_text(encoding='utf-8'))
display(verdict)'''),
md('''## 1. Conclusion principale'''),
code('''conclusion = json.loads((RESULTS_DIR/'diagnose/diagnostic-conclusion.json').read_text(encoding='utf-8'))
display(Markdown(f"**Primary blocker:** `{conclusion['primary_blocker']}`"))
display(pd.DataFrame([{
    'quiet_zone_rescues': len(conclusion['quiet_zone_rescues']),
    'binarization_rescues': len(conclusion['binarization_rescues']),
    'grid_rescues': len(conclusion['grid_reconstruction_rescues']),
    'opencv_current_detected': conclusion['opencv_current_detected_count'],
    'min_mean_module_errors': conclusion['minimum_target_assisted_mean_module_errors'],
    'min_format_errors': conclusion['minimum_target_assisted_format_errors'],
    'min_data_errors': conclusion['minimum_target_assisted_data_errors'],
}]))
display(pd.DataFrame({'recommended_e043_loss_components': conclusion['recommended_e043_loss_components']}))'''),
md('''## 2. Matrice décodeur × transformation'''),
code('''matrix = pd.read_csv(RESULTS_DIR/'diagnose/decoder-stage-matrix.csv')
cols = ['state_id','gamma','iteration','variant','opencv_detected','opencv_exact','zbar_exact','zxingcpp_exact','wechat_exact','qr_verify_one_shot_exact_presets','qr_verify_one_shot_original_exact']
display(matrix[cols].sort_values(['state_id','variant']).reset_index(drop=True))'''),
md('''## 3. Résumé structurel par état'''),
code('''states = json.loads((RESULTS_DIR/'diagnose/state-diagnostics.json').read_text(encoding='utf-8'))
rows=[]
for d in states:
    s=d['state']; st=d['structure']; q=s.get('quiet_zone_overwrite') or {}
    rows.append({
        'state_id':s['state_id'],'gamma':s['gamma'],'iteration':s['iteration'],
        'mean050_errors':st['mean_threshold_050']['total_error_count'],
        'mean_best_threshold':st['mean_best']['threshold'],
        'mean_best_errors':st['mean_best']['total_error_count'],
        'format_errors_best':st['mean_best']['format_error_count'],
        'data_errors_best':st['mean_best']['data_error_count'],
        'intra_module_std_mean':st['intra_module_std_mean'],
        'qz_overwrites_core_edge':q.get('legacy_quiet_zone_overwrites_exact_core'),
        'qz_core_edge_changed_ratio':q.get('changed_pixel_ratio_exact_core_edge_2px'),
    })
structure_df=pd.DataFrame(rows).sort_values(['mean_best_errors','gamma','iteration'])
display(structure_df)'''),
md('''## 4. Erreurs par sous-région QR'''),
code('''region_rows=[]
for d in states:
    s=d['state']; best=d['structure']['mean_best']
    region_rows.append({
        'state_id':s['state_id'],'gamma':s['gamma'],'iteration':s['iteration'],
        **{f'{r}_errors':best[f'{r}_error_count'] for r in ['finder','separator','timing','alignment','format','fixed_dark','data']}
    })
display(pd.DataFrame(region_rows).sort_values(['format_errors','data_errors']))'''),
md('''## 5. Rescues QR-Verify'''),
code('''rescues = json.loads((RESULTS_DIR/'diagnose/conservative-rescue-checks.json').read_text(encoding='utf-8'))
if rescues:
    display(pd.DataFrame([{
        'state_id':r['state_id'],'variant':r['variant'],
        'one_shot_exact':r['diagnostic_one_shot_exact_presets'],
        'conservative_exact':r['conservative']['conservative_exact_presets'],
        'preset_count':r['conservative']['preset_count'],
    } for r in rescues]))
else:
    display(Markdown('Aucun transform diagnostic n’a obtenu de preset QR-Verify exact en one-shot.'))'''),
md('''## 6. Contact sheet de l’état représentatif'''),
code('''sheet = RESULTS_DIR/'diagnose/pipeline/decoder-localization-contact-sheet.png'
display(IPImage(filename=str(sheet), width=1100))'''),
md('''## 7. Cartes d’erreurs du représentant'''),
code('''rep=verdict['representative_state']
for name in ['module-error-map-mean050.png','module-error-map-mean-best.png']:
    display(Markdown(f'### {name}'))
    display(IPImage(filename=str(RESULTS_DIR/'diagnose/states'/rep/name), width=520))'''),
md('''## Lecture scientifique

- **Quiet-zone rescue** → corriger la géométrie 736/78 avant toute nouvelle loss.
- **Otsu/adaptive rescue** → la texture/binarisation est le verrou ; E043 doit optimiser une marge sous plusieurs seuils.
- **Grid reconstruction rescue** → les bits sont présents mais le scanner ne peut pas récupérer la grille proprement ; pénaliser variance intra-module + géométrie cellule.
- **Aucun rescue** avec erreurs data/format restantes → travailler directement les bits de format/data et une loss plus proche de l’ECC.

Aucun résultat E042 n’autorise production ou généralisation.''')
]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print(OUT)
