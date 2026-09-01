from __future__ import annotations
import json
from pathlib import Path
OUT=Path('notebooks/40_e042_diagnostic_pipeline_visualizer.ipynb')
def md(t): return {'cell_type':'markdown','metadata':{},'source':[x+'\n' for x in t.strip().splitlines()]}
def code(t): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[x+'\n' for x in t.strip().splitlines()]}
cells=[
md('''# E042 — Pipeline visuelle du diagnostic

Ce notebook montre **ce que voit le décodeur** à chaque transformation. Il ne présente pas un nouveau QR final.'''),
code('''from pathlib import Path
import json
from IPython.display import display, Image as IPImage, Markdown
R=Path('/data/e042-decoder-failure-localization-v1')
v=json.loads((R/'verdict.json').read_text(encoding='utf-8'))
display(v)
P=R/'diagnose/pipeline' '''),
md('''## Chaîne de localisation'''),
code('''steps=[
('01-exact-reference.png','Référence binaire exacte — contrôle QR-Verify'),
('02-current-scan-ready.png','Raster E041 réellement scoré'),
('03-raw-vae.png','Même latent, VAE brut sans post-traitement quiet-zone'),
('04-exact-qz.png','Quiet zone reconstruite sur géométrie exacte padding=78'),
('05-otsu.png','Binarisation globale Otsu'),
('06-adaptive.png','Binarisation adaptive'),
('07-grid-mean-050.png','Grille canonique inférée par moyenne cellule, seuil .50'),
('08-grid-mean-best.png','Grille moyenne avec meilleur seuil target-assisted'),
('09-grid-center-best.png','Grille centre avec meilleur seuil target-assisted'),
('10-module-error-map-mean050.png','Erreurs modules au seuil .50'),
('11-module-error-map-mean-best.png','Erreurs modules au meilleur seuil'),
]
for filename,caption in steps:
    display(Markdown(f'### {caption}'))
    display(IPImage(filename=str(P/filename), width=760))'''),
md('''## Contact sheet'''),
code('''display(IPImage(filename=str(P/'decoder-localization-contact-sheet.png'), width=1200))'''),
md('''## Conclusion'''),
code('''c=json.loads((R/'diagnose/diagnostic-conclusion.json').read_text(encoding='utf-8'))
display(Markdown(f"**Primary blocker : `{c['primary_blocker']}`**"))
display(Markdown('**E043 proposé :** ' + ', '.join(c['recommended_e043_loss_components'])))'''),
]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}},'nbformat':4,'nbformat_minor':5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
print(OUT)
