"""Aggregate completed E044 prompt jobs without loading diffusion models."""
from __future__ import annotations
import argparse,csv,json,os,shutil
from datetime import UTC,datetime
from pathlib import Path
from typing import Any
from PIL import Image,ImageDraw
from .e044_multiprompt_best_pipeline import EXPERIMENT,GAMMAS,PROMPTS,SEED

def _atomic_json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8'); os.replace(tmp,path)

def _write_csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows: path.write_text('',encoding='utf-8'); return
    fields=sorted({k for r in rows for k in r})
    with path.open('w',encoding='utf-8',newline='') as s:
        w=csv.DictWriter(s,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def _rank_key(r): return (-int(r.get('qr_verify_exact_presets',0)),-int(bool(r.get('original_exact'))),int(r.get('full_module_error_count',10**9)),float(r.get('lpips',1e9)),-float(r.get('clip_aesthetic') or -1e9),int(r.get('iteration',10**9)))

def _sheet(path,items,columns=4):
    if not items:return
    thumb=320; label=58; rows=(len(items)+columns-1)//columns
    sheet=Image.new('RGB',(columns*thumb,rows*(thumb+label)),'white'); d=ImageDraw.Draw(sheet)
    for i,(title,img,sub) in enumerate(items):
        r,c=divmod(i,columns); tile=img.convert('RGB').copy(); tile.thumbnail((thumb-12,thumb-12),Image.Resampling.LANCZOS)
        x=c*thumb+(thumb-tile.width)//2; y=r*(thumb+label)+(thumb-tile.height)//2; sheet.paste(tile,(x,y)); ty=r*(thumb+label)+thumb
        d.text((c*thumb+8,ty+4),title[:42],fill='black'); d.text((c*thumb+8,ty+26),sub[:48],fill='black')
    path.parent.mkdir(parents=True,exist_ok=True); sheet.save(path,format='PNG',optimize=False,compress_level=9)

def aggregate(*,root:Path,source_commit:str):
    verdicts=[]; all_rows=[]; summaries=[]
    for p in PROMPTS:
        pdir=root/'prompts'/p['id']; complete=pdir/'COMPLETE.json'; comp=pdir/'scoring/comparison.json'
        if not complete.is_file() or not comp.is_file(): raise FileNotFoundError(f'E044 prompt incomplete: {p["id"]}')
        v=json.loads(complete.read_text(encoding='utf-8')); rows=json.loads(comp.read_text(encoding='utf-8')); verdicts.append(v); all_rows.extend(rows)
        safe=[r for r in rows if bool(r.get('visual_guard_pass'))]; best=sorted(safe,key=_rank_key)[0]
        summaries.append({'prompt_id':p['id'],'family':p['family'],'prompt':p['text'],'best_variant':best['variant'],'best_gamma':best['gamma'],'best_iteration':best['iteration'],'best_ssr_exact_presets':best['qr_verify_exact_presets'],'best_ssr':best['ssr'],'best_original_exact':best['original_exact'],'best_lpips':best['lpips'],'best_clip_score':best.get('clip_score'),'best_clip_aesthetic':best.get('clip_aesthetic'),'best_hpsv2_1':best.get('hpsv2_1'),'best_module_errors':best['full_module_error_count'],'best_image_path':best['image_path']})
    safe_all=[r for r in all_rows if bool(r.get('visual_guard_pass'))]; winner=sorted(safe_all,key=_rank_key)[0]; raw=sorted(all_rows,key=_rank_key)[0]
    _atomic_json(root/'prompt-verdicts.json',verdicts); _atomic_json(root/'prompt-summary.json',summaries); _atomic_json(root/'comparison-all.json',all_rows)
    csvrows=[]
    for r in all_rows:
        f=dict(r); f['visual_guard_checks']=json.dumps(f.get('visual_guard_checks') or {},ensure_ascii=False,sort_keys=True); f['decoder_diagnostics']=json.dumps(f.get('decoder_diagnostics') or {},ensure_ascii=False,sort_keys=True); csvrows.append(f)
    _write_csv(root/'comparison-all.csv',csvrows)
    gs=[]
    for g in GAMMAS:
        sub=[r for r in all_rows if float(r.get('gamma',0.0))==float(g) and bool(r.get('visual_guard_pass'))]
        gs.append({'gamma':g,'safe_checkpoint_count':len(sub),'prompt_count_with_any_exact':len({r['prompt_id'] for r in sub if int(r['qr_verify_exact_presets'])>0}),'max_ssr_exact_presets':max((int(r['qr_verify_exact_presets']) for r in sub),default=0),'mean_ssr':sum(float(r['ssr']) for r in sub)/len(sub) if sub else 0.0,'projection_active_count':sum(bool(r.get('projection_was_active')) for r in sub)})
    _atomic_json(root/'gamma-summary.json',gs)
    pipe=root/'pipeline'; pipe.mkdir(parents=True,exist_ok=True); shutil.copy2(Path(str(winner['image_path'])),pipe/'99-FINAL-QR.png'); shutil.copy2(Path(str(winner['latent_path'])),pipe/'99-FINAL-latent.safetensors')
    wp=root/'prompts'/str(winner['prompt_id']); shutil.copy2(wp/'parent/stage1.png',pipe/'01-WINNER-stage1.png'); shutil.copy2(wp/'parent/stage2.png',pipe/'02-WINNER-stage2.png'); shutil.copy2(wp/'parent/stage2-exact-qz.png',pipe/'03-WINNER-stage2-exact-qz.png')
    _sheet(pipe/'best-by-prompt-contact-sheet.png',[(s['prompt_id'],Image.open(s['best_image_path']).convert('RGB'),f"g={int(s['best_gamma'])} i={s['best_iteration']} SSR={s['best_ssr_exact_presets']}/37") for s in summaries])
    v={'experiment':EXPERIMENT,'created_at_utc':datetime.now(UTC).isoformat(),'source_commit':source_commit,'seed':SEED,'prompt_count':len(PROMPTS),'gamma_grid':list(GAMMAS),'checkpoint_count':sum(int(x['checkpoint_count']) for x in verdicts),'scored_image_count':len(all_rows),'safe_image_count':len(safe_all),'prompts_with_any_qrverify_success':sum(int(s['best_ssr_exact_presets']>0) for s in summaries),'winner_prompt_id':winner['prompt_id'],'winner_prompt_family':winner['prompt_family'],'winner_variant':winner['variant'],'winner_gamma':winner['gamma'],'winner_iteration':winner['iteration'],'winner_ssr_exact_presets':winner['qr_verify_exact_presets'],'winner_ssr':winner['ssr'],'winner_original_exact':winner['original_exact'],'winner_visual_guard_pass':winner['visual_guard_pass'],'winner_lpips':winner['lpips'],'winner_clip_score':winner.get('clip_score'),'winner_clip_aesthetic':winner.get('clip_aesthetic'),'winner_hpsv2_1':winner.get('hpsv2_1'),'raw_best_prompt_id':raw['prompt_id'],'raw_best_variant':raw['variant'],'raw_best_ssr_exact_presets':raw['qr_verify_exact_presets'],'paper_comparison_kind':'documented methodological comparison; E044 is not claimed paper-exact','authoritative_scanner':'qr-verify@0.2.0 conservative 37-preset exact-payload scoring','multi_prompt_screen':True,'multi_seed_generalization':False,'production_ready':False,'generalization_authorized':False,'next_action':'REVIEW_COMPLETE_NOTEBOOK_AND_PROMPT_SENSITIVITY_BEFORE_ANY_NEW_LOSS'}
    _atomic_json(root/'verdict.json',v); (root/'report.md').write_text(f"# E044 — multi-prompt best-pipeline benchmark\n\n- prompts: **{len(PROMPTS)}**\n- checkpoints SR-MPGD: **{v['checkpoint_count']}**\n- scored images: **{len(all_rows)}**\n- prompts with any QR-Verify success: **{v['prompts_with_any_qrverify_success']}**\n- winner: **{winner['prompt_id']} / {winner['variant']}**\n- SSR: **{winner['qr_verify_exact_presets']}/37**\n\nE044 is a shared-seed prompt screen, not multi-seed generalization.\nProduction and generalization remain false.\n",encoding='utf-8'); return v

def _cli():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); p.add_argument('--source-commit',required=True); a=p.parse_args(); print(json.dumps(aggregate(root=a.root,source_commit=a.source_commit),ensure_ascii=False,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(_cli())
