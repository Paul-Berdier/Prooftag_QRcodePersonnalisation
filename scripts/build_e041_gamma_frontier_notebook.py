#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks/37_e041_gamma_functional_frontier.ipynb'

def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def code(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)}
cells = [
    md('# E041 — Gamma × motifs fonctionnels\n\nNouveau prompt, nouveau parent Stage 1/2. E040 reste un contrôle historique non apparié.\n'),
    code("""from pathlib import Path\nimport json, os\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import Image as DisplayImage, Markdown, display\nRESULTS_DIR = Path(os.environ.get('E041_RESULTS_DIR','/data/e041-gamma-functional-pattern-frontier-v1'))\nprint(RESULTS_DIR)\n"""),
    md('## 1. Verdict'),
    code("""v = json.loads((RESULTS_DIR/'verdict.json').read_text(encoding='utf-8'))\ndisplay(v)\n"""),
    md('## 2. Phase A — effet de gamma'),
    code("""a = pd.read_csv(RESULTS_DIR/'phase-a-scoring/comparison.csv')\ncols=[c for c in ['variant','gamma','iteration','qr_verify_exact_presets','original_exact','full_module_error_count','functional_center_error_rate','data_center_error_rate','lpips','accepted_alpha','projection_was_active','visual_guard_pass'] if c in a.columns]\ndisplay(a[cols].sort_values(['gamma','iteration']).reset_index(drop=True))\nfig,ax=plt.subplots(figsize=(10,6))\nfor gamma,g in a.groupby('gamma'):\n    g=g.sort_values('iteration'); ax.plot(g['iteration'],g['qr_verify_exact_presets'],marker='o',label=str(int(gamma)))\nax.set_xlabel('checkpoint i'); ax.set_ylabel('QR-Verify exact /37'); ax.set_title('E041 — SSR par gamma'); ax.legend(title='gamma'); plt.tight_layout(); plt.show()\n"""),
    md('## 3. Projection / backtracking'),
    code("""proj = pd.read_json(RESULTS_DIR/'gamma-projection-summary.json')\ndisplay(proj)\n"""),
    md('## 4. Phase B — renforcement sélectif'),
    code("""b = pd.read_csv(RESULTS_DIR/'phase-b-scoring/comparison.csv')\nb = b[b['phase']=='B'].copy()\ncols=[c for c in ['variant','base_checkpoint','gamma','iteration','functional_tone_factor','qr_verify_exact_presets','original_exact','full_module_error_count','functional_center_error_rate','data_center_error_rate','lpips','clip_aesthetic','clip_score','hpsv2_1','visual_guard_pass'] if c in b.columns]\ndisplay(b[cols].sort_values(['gamma','functional_tone_factor']).reset_index(drop=True))\n"""),
    md('## 5. Gagnant'),
    code("""display(Markdown(f\"**Winner:** {v['winner_variant']} — gamma={v['selected_gamma']} — tone={v['selected_functional_tone_factor']} — SSR={v['winner_ssr_exact_presets']}/37 — original={v['winner_original_exact']}\"))\ndisplay(DisplayImage(filename=str(RESULTS_DIR/'pipeline/99-FINAL-QR.png'), width=850))\n"""),
    md('## 6. Pipeline contact sheet'),
    code("""display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/full-pipeline-contact-sheet.png'), width=1500))\n"""),
]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8'); print(OUT)
