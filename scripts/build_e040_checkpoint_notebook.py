#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/35_e040_srmpgd_checkpoint_frontier.ipynb"

def md(s): return {"cell_type":"markdown","metadata":{},"source":s.splitlines(True)}
def code(s): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":s.splitlines(True)}

cells = [
md("""# E040 — SR-MPGD checkpoint frontier\n\nE040 garde **γ=1000**, la loss `scanaware_v2` d'E039 et évalue chaque checkpoint `i0..i8` pour cinq rayons `0.150..0.250`. Le gagnant est le meilleur **checkpoint sûr**, pas forcément le dernier.\n\nLe CNN E016, lorsqu'il est promu `research_usable` par sa propre model card, est affiché comme score secondaire. QR-Verify reste la vérité terrain logicielle.\n"""),
code("""from __future__ import annotations\nimport json, os\nfrom pathlib import Path\nimport matplotlib.pyplot as plt\nimport pandas as pd\nfrom IPython.display import Image as DisplayImage, Markdown, display\nRESULTS_DIR = Path(os.environ.get('E040_RESULTS_DIR','/data/e040-srmpgd-checkpoint-frontier-v1'))\nprint('E040 résultats :', RESULTS_DIR)\n"""),
md("## 1. Verdict"),
code("""verdict = json.loads((RESULTS_DIR/'verdict.json').read_text(encoding='utf-8'))\ndisplay(Markdown('**E040 terminé : résultats disponibles.**'))\ndisplay(verdict)\n"""),
md("## 2. Tous les checkpoints — vrai SSR + esthétique"),
code("""df = pd.read_csv(RESULTS_DIR/'checkpoint-comparison.csv')\ncols = ['checkpoint','radius','iteration','gamma','qr_verify_exact_presets','ssr','original_exact','full_module_error_count','lpips','latent_delta_rms','clip_score','clip_aesthetic','hpsv2_1','surrogate_mean_success_probability','visual_guard_pass','acceptance_reason']\ncols=[c for c in cols if c in df.columns]\ndisplay(df[cols].sort_values(['qr_verify_exact_presets','visual_guard_pass','lpips'], ascending=[False,False,True]).reset_index(drop=True))\n"""),
md("## 3. Pourquoi un checkpoint est-il rejeté par la garde visuelle ?"),
code("""raw = json.loads((RESULTS_DIR/'checkpoint-comparison.json').read_text(encoding='utf-8'))\nguard_rows=[]\nfor row in raw:\n    checks=row.get('visual_guard_checks') or {}\n    guard_rows.append({'checkpoint':row['checkpoint'],'SSR /37':row['qr_verify_exact_presets'],'LPIPS':row['lpips'],'safe':row['visual_guard_pass'],**checks})\nguards=pd.DataFrame(guard_rows)\ndisplay(guards.sort_values(['SSR /37','safe'],ascending=[False,False]).reset_index(drop=True))\n"""),
md("## 4. Meilleur checkpoint par rayon"),
code("""best = pd.DataFrame(json.loads((RESULTS_DIR/'best-checkpoint-per-radius.json').read_text(encoding='utf-8')))\nif not best.empty:\n    display(best[['checkpoint','radius','iteration','qr_verify_exact_presets','full_module_error_count','lpips','visual_guard_pass']])\n"""),
md("## 5. SSR selon rayon et itération"),
code("""fig, ax = plt.subplots(figsize=(11,6))\nfor radius, group in df.groupby('radius'):\n    group=group.sort_values('iteration')\n    ax.plot(group['iteration'],group['qr_verify_exact_presets'],marker='o',label=f'r={radius:.3f}')\nax.set_xlabel('checkpoint SR-MPGD')\nax.set_ylabel('QR-Verify exact presets / 37')\nax.set_title('E040 — le meilleur checkpoint n’est pas forcément le dernier')\nax.legend()\nplt.tight_layout(); plt.show()\n"""),
md("## 6. SSR ↔ LPIPS"),
code("""fig, ax = plt.subplots(figsize=(9,6))\nfor radius, group in df.groupby('radius'):\n    ax.scatter(group['lpips'],group['qr_verify_exact_presets'],label=f'r={radius:.3f}')\nax.set_xlabel('LPIPS vs parent'); ax.set_ylabel('SSR /37'); ax.set_title('E040 — frontier scan / esthétique')\nax.legend(); plt.tight_layout(); plt.show()\n"""),
md("## 7. Pipeline complète en une image"),
code("""display(DisplayImage(filename=str(RESULTS_DIR/'pipeline/full-pipeline-contact-sheet.png'), width=1500))\n"""),
md("## 8. Trajectoire complète du gagnant"),
code("""winner_recipe=verdict['research_winner_recipe']\nwinner_iteration=int(verdict['winner_iteration'])\nfor i in range(9):\n    row=df[(df.method==winner_recipe)&(df.iteration==i)].iloc[0]\n    display(Markdown(f\"### i{i} — SSR={int(row.qr_verify_exact_presets)}/37 · LPIPS={row.lpips:.4f} · safe={row.visual_guard_pass}\"))\n    display(DisplayImage(filename=str(RESULTS_DIR/winner_recipe/'images'/f'iteration-{i:03d}.png'), width=720))\n    if i==winner_iteration: display(Markdown('**← CHECKPOINT FINAL SÉLECTIONNÉ**'))\n"""),
md("## 9. Modèle E016 — score des checkpoints"),
code("""status=json.loads((RESULTS_DIR/'e016-surrogate-status.json').read_text(encoding='utf-8'))\ndisplay(status)\nif status.get('research_usable'):\n    scores=pd.DataFrame([{'checkpoint':k,**v} for k,v in json.loads((RESULTS_DIR/'e016-surrogate-scores.json').read_text(encoding='utf-8')).items()])\n    display(scores.sort_values('mean_success_probability',ascending=False).head(20))\nelse:\n    display(Markdown('E016 n’est pas utilisé pour départager tant que sa propre card ne le marque pas `research_usable`.'))\n"""),
md("## 10. Advisor E026/E031 — recommandation prospective"),
code("""advisor=json.loads((RESULTS_DIR/'advisor-preview.json').read_text(encoding='utf-8'))\ndisplay(advisor)\n"""),
md("## 11. E039 vs E040"),
code("""control=json.loads((RESULTS_DIR/'e039-control.json').read_text(encoding='utf-8'))\nwinner=df[df.checkpoint==verdict['research_winner_checkpoint']].iloc[0]\nsummary=pd.DataFrame([\n {'method':'E039 safe winner','SSR /37':control['verdict'].get('winner_ssr_exact_presets'),'LPIPS':control['winner'].get('lpips'),'MER':control['winner'].get('full_module_error_count')},\n {'method':'E040 best checkpoint','SSR /37':winner.qr_verify_exact_presets,'LPIPS':winner.lpips,'MER':winner.full_module_error_count},\n])\ndisplay(summary)\n"""),
]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"}},"nbformat":4,"nbformat_minor":5}
OUTPUT.parent.mkdir(parents=True,exist_ok=True); OUTPUT.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8'); print(OUTPUT)
