#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'notebooks/36_final_qr_pipeline_visualizer.ipynb'
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(True)}
def code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.splitlines(True)}
cells=[
md("""# Pipeline QR finale — vue de bout en bout\n\nCe notebook montre la chaîne réelle ayant mené au QR final E040 : **QR exact → condition ControlNet → Stage 1 → Stage 2 → SR-MPGD i0…i8 → sélection du meilleur checkpoint → QR final**.\n\nIl affiche aussi le modèle advisor E026/E031 et le surrogate E016 avec leur rôle exact. Le parent Stage 1/Stage 2 reste figé pour que l'optimisation E040 soit comparable à E035–E039.\n"""),
code("""import json, os\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image as DisplayImage, Markdown, display\nRESULTS_DIR=Path(os.environ.get('E040_RESULTS_DIR','/data/e040-srmpgd-checkpoint-frontier-v1'))\nmanifest=json.loads((RESULTS_DIR/'pipeline-manifest.json').read_text(encoding='utf-8'))\nverdict=json.loads((RESULTS_DIR/'verdict.json').read_text(encoding='utf-8'))\ndisplay(Markdown(f\"**Payload :** `{manifest['payload']}`  \\\n**Prompt :** {manifest['prompt']}  \\\n**γ :** {verdict['gamma']}\"))\n"""),
md("## 1. La pipeline entière"),
code("""display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/full-pipeline-contact-sheet.png'),width=1500))\n"""),
md("## 2. QR exact / condition ControlNet"),
code("""display(Markdown('### QR exact payload')); display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/01-qr-reference.png'),width=600))\ndisplay(Markdown('### Condition binaire envoyée au ControlNet')); display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/02-control-condition.png'),width=700))\n"""),
md("## 3. Stage 1 — diffusion artistique"),
code("""p=RESULTS_DIR/'pipeline/03-stage1.png'\nif p.is_file(): display(DisplayImage(filename=str(p),width=760))\nelse: display(Markdown('Stage 1 archivé absent dans cette image de déploiement.'))\n"""),
md("## 4. Stage 2 — SRPG"),
code("""display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/04-stage2.png'),width=760))\n"""),
md("## 5. SR-MPGD — tous les checkpoints"),
code("""df=pd.read_csv(RESULTS_DIR/'checkpoint-comparison.csv')\nrecipe=verdict['research_winner_recipe']; selected=int(verdict['winner_iteration'])\nfor i in range(9):\n row=df[(df.method==recipe)&(df.iteration==i)].iloc[0]\n label=' ⭐ FINAL' if i==selected else ''\n display(Markdown(f\"### SR-MPGD i{i}{label} — SSR={int(row.qr_verify_exact_presets)}/37 · MER={int(row.full_module_error_count)}/841 · LPIPS={row.lpips:.4f} · safe={row.visual_guard_pass}\"))\n display(DisplayImage(filename=str(RESULTS_DIR/recipe/'images'/f'iteration-{i:03d}.png'),width=760))\n"""),
md("## 6. QR FINAL sélectionné"),
code("""display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/99-FINAL-QR.png'),width=850))\ndisplay(verdict)\n"""),
md("## 7. Pourquoi ce checkpoint a gagné ?"),
code("""raw=json.loads((RESULTS_DIR/'checkpoint-comparison.json').read_text(encoding='utf-8'))\nw=next(x for x in raw if x['checkpoint']==verdict['research_winner_checkpoint'])\ndisplay(pd.DataFrame([{'metric':'SSR /37','value':w['qr_verify_exact_presets']},{'metric':'original_exact','value':w['original_exact']},{'metric':'MER /841','value':w['full_module_error_count']},{'metric':'LPIPS','value':w['lpips']},{'metric':'latent RMS','value':w['latent_delta_rms']},{'metric':'CLIPScore','value':w.get('clip_score')},{'metric':'CLIP-Aesthetic','value':w.get('clip_aesthetic')},{'metric':'HPS','value':w.get('hpsv2_1')},{'metric':'E016 mean p(scan)','value':w.get('surrogate_mean_success_probability')},{'metric':'visual safe','value':w['visual_guard_pass']}]))\ndisplay(Markdown('### Détail de la garde visuelle')); display(w.get('visual_guard_checks'))\n"""),
md("## 8. Modèle advisor E026/E031"),
code("""advisor=json.loads((RESULTS_DIR/'advisor-preview.json').read_text(encoding='utf-8'))\nif advisor.get('available'):\n display(Markdown('Le modèle est chargé et recommande des paramètres **avant génération**. Il ne certifie jamais le QR.'))\n display(pd.DataFrame(advisor.get('recommendations',[])))\nelse: display(advisor)\n"""),
md("## 9. Modèle E016 différentiable"),
code("""status=json.loads((RESULTS_DIR/'e016-surrogate-status.json').read_text(encoding='utf-8')); display(status)\nif status.get('research_usable'):\n display(Markdown('E016 score les checkpoints en plus des vrais décodeurs. Dans E040 il est volontairement **secondaire à QR-Verify**.'))\n"""),
md("## 10. Carte finale de la chaîne"),
code("""display(Markdown('''```text\nPrompt + payload\n      ↓\nAdvisor E026/E031 (recommandation)\n      ↓\nQR exact → condition QR Monster\n      ↓\nStage 1 : Cetus-Mix + ControlNet\n      ↓\nStage 2 : SRPG\n      ↓ latent exact z0\nSR-MPGD scan-aware, γ=1000\n      ↓ i0..i8\nQR-Verify + garde esthétique + score E016\n      ↓\nMEILLEUR CHECKPOINT SÛR\n      ↓\n99-FINAL-QR.png\n```'''))\n"""),
]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
OUTPUT.parent.mkdir(parents=True,exist_ok=True); OUTPUT.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8'); print(OUTPUT)
