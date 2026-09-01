#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks/38_e041_final_pipeline_visualizer.ipynb'

def md(s): return {'cell_type':'markdown','metadata':{},'source':s.splitlines(True)}
def code(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)}
cells=[
 md('# E041 — Pipeline finale visuelle\n\nQR → condition → Stage 1 → Stage 2 → gamma/scan-aware → renforcement fonctionnel → final.\n'),
 code("""from pathlib import Path\nimport json, os\nfrom IPython.display import Image as DisplayImage, Markdown, display\nRESULTS_DIR=Path(os.environ.get('E041_RESULTS_DIR','/data/e041-gamma-functional-pattern-frontier-v1'))\nv=json.loads((RESULTS_DIR/'verdict.json').read_text(encoding='utf-8'))\n"""),
 md('## 1. QR référence'), code("display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/01-qr-reference.png'),width=650))\n"),
 md('## 2. Condition ControlNet'), code("display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/02-control-condition.png'),width=650))\n"),
 md('## 3. Stage 1 — nouveau prompt'), code("display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/03-stage1.png'),width=800))\n"),
 md('## 4. Stage 2 SRPG'), code("display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/04-stage2.png'),width=800))\n"),
 md('## 5. Stage 2 scan-ready'), code("display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/05-stage2-scan-ready.png'),width=800))\n"),
 md('## 6. Gamma sélectionné + checkpoint'), code("display(Markdown(f\"**gamma={v['selected_gamma']} / i{v['selected_iteration']}**\"))\n"),
 md('## 7. Renforcement motifs fonctionnels'), code("display(Markdown(f\"**functional_pattern_tone_factor={v['selected_functional_tone_factor']}**\"))\n"),
 md('## 8. QR FINAL'), code("display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/99-FINAL-QR.png'),width=850))\ndisplay(v)\n"),
 md('## 9. Vue complète'), code("display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/full-pipeline-contact-sheet.png'),width=1500))\n"),
 md('## 10. Advisor'), code("advisor=json.loads((RESULTS_DIR/'advisor-preview.json').read_text(encoding='utf-8')); display(advisor)\n"),
]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},'nbformat':4,'nbformat_minor':5}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8'); print(OUT)
