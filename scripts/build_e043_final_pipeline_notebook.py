from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'notebooks/42_e043_final_pipeline_visualizer.ipynb'
doc=json.loads(p.read_text(encoding='utf-8'))
p.write_text(json.dumps(doc,indent=1),encoding='utf-8')
print(p)
