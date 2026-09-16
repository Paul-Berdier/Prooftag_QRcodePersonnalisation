"""Container tasks on the existing, pinned QR stack. /data is mounted READ ONLY."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import time
import traceback

from . import BASE_COMMIT, VERSION
from .common import read, write, sha, digest, git_blob, eligibility, utc, normalized_prompt

STOP = False

def on_signal(*_):
    global STOP
    STOP = True


def expired(deadline):
    return STOP or time.time() >= deadline


def upstream():
    from prooftag_qr import e046_large_campaign as e
    return e


def load(cfg):
    e = upstream()
    d, plan = e.load_plan(Path(cfg["source_root"]), cfg["source_plan_id"])
    if plan["source_commit"] != BASE_COMMIT:
        raise RuntimeError("Plan incompatible avec le commit audité")
    return e, d, plan


def validate_runtime():
    attestation = Path("/app/prooftag-build-commit.txt").read_text().strip()
    if attestation != BASE_COMMIT:
        raise RuntimeError("L'image de base ne correspond pas au commit audité")
    expected = {
       "e046_large_campaign.py": "9e937224f0e66899fcd782587408d0a4571d178a",
       "e046_campaign.py": "55e68fef4429f8fa822952667ef62e3d6389977d",
       "diffqrcoder_backend.py": "9e5b48d1f9eb0531e5f08e319966e1dc027d5976",
       "e046_catalog.py": "b9856096995e23b298c07ba9f3e0ecc505793154",
    }
    import prooftag_qr
    root = Path(prooftag_qr.__file__).parent
    for name, h in expected.items():
        if git_blob(root / name) != h:
            raise RuntimeError(f"Fichier scientifique incompatible : {name}")
    return {"base_commit": attestation, "upstream_git_blobs": expected,
            "addon_version": VERSION, "addon_image": os.getenv("QRNIGHT_IMAGE"),
            "packages": {p: importlib.metadata.version(p) for p in
                ("torch", "diffusers", "transformers", "numpy", "scikit-learn", "joblib", "matplotlib")}}


def inventory(work, cfg, deadline):
    e, d, plan = load(cfg)
    items = []
    recipes = e._recipe_map(plan)
    for c in plan["candidates"]:
        if expired(deadline):
            break
        p = d / "parents" / c["id"]
        valid = e._promotion_valid(p, e.PARENT_GENERATION_REQUIRED, deep=True)
        scored = valid and e._promotion_valid(p / "scoring", e.SCORING_REQUIRED, deep=True)
        items.append({"candidate_id": c["id"], "prompt_id": c["prompt_id"],
                      "generated_valid": valid, "scored_valid": scored})
    result = {"source_plan": str(d), "source_plan_sha256": sha(d / "plan.json"),
              "expected": len(plan["candidates"]), "checked": len(items),
              "partial": len(items) != len(plan["candidates"]), "items": items,
              "generated": sum(x["generated_valid"] for x in items),
              "scored": sum(x["scored_valid"] for x in items), "at": utc()}
    write(work / "inventory.json", result)
    return result


def preflight(work, cfg):
    runtime = validate_runtime()
    e, d, plan = load(cfg)
    import imageio_ffmpeg
    from prooftag_qr.e046_catalog import PARENT_RECIPES
    from prooftag_qr.quality_scoring import quality_scorer_from_settings
    # The funny payload is tested as plain text; no shortener or payload rewrite.
    allowed = []
    for ecc in ("M", "Q", "H"):
        c = dict(plan["candidates"][0], payload=cfg["payload"])
        try:
            b = e._blueprint(c, replace(PARENT_RECIPES[4], error_correction=ecc))
            allowed.append(ecc)
        except Exception:
            pass
    if "M" not in allowed:
        raise RuntimeError("Le payload ne tient pas dans la géométrie de référence M")
    scorer = e._new_qr_scorer(work, plan)
    try:
        c = dict(plan["candidates"][0], payload=cfg["payload"])
        b = e._blueprint(c, PARENT_RECIPES[4])
        q = e._score_qr_cached(plan_dir=work, image=b.image, payload=cfg["payload"], scorer=scorer, plan=plan)
        fields = e._qr_fields(q)
        if fields["wechat_exact_presets"] != 37 or not fields["wechat_original_exact"]:
            raise RuntimeError("Autotest QR classique / payload exact échoué")
    finally:
        scorer.close()
    # Warm CPU quality stack and verify local checkpoints before authorizing ANY GPU outage.
    first = next((c for c in plan["candidates"] if
                  (d / "parents" / c["id"] / "images/stage2-raw.png").is_file()), None)
    if first is None:
        raise RuntimeError("Aucune image Stage2 présente dans le plan demandé")
    from PIL import Image
    settings = e._settings_for_plan(plan, first, e._parent_recipe(plan, first))
    quality = quality_scorer_from_settings(settings, device="cpu")
    with Image.open(d / "parents" / first["id"] / "images/stage2-raw.png") as im:
        scores, prov = e._quality_scores(images={"raw": im.convert("RGB")}, prompt=first["prompt"],
                                        settings=settings, scorer=quality)
    if any(scores["raw"].get(k) is None for k in ("clip_score", "clip_aesthetic", "hpsv2_1")):
        raise RuntimeError("Scoring qualité incomplet")
    result = {"status": "PASS", "at": utc(), "runtime": runtime, "source_plan": plan["plan_id"],
              "plan_sha256": sha(d / "plan.json"), "allowed_payload_ecc": allowed,
              "plain_qr": fields, "quality_warmup": scores, "quality_provenance": prov,
              "ffmpeg": imageio_ffmpeg.get_ffmpeg_exe(), "gpu_used": False}
    write(work / "preflight.json", result)
    return result


def score_one(work, cfg, e, d, plan, c, qr_scorer, quality_scorer):
    p = d / "parents" / c["id"]
    output = work / "scores" / (c["id"] + ".json")
    if output.is_file():
        prior = read(output)
        if prior["source_plan_sha256"] == sha(d / "plan.json") and prior["image_file_sha256"] == sha(p / "images/stage2-raw.png"):
            return prior
        raise RuntimeError("Score dérivé existant incompatible; ne pas écraser")
    from PIL import Image
    recipe = e._parent_recipe(plan, c)
    metadata = read(p / "parent-metadata.json")
    with Image.open(p / "images/stage1-raw.png") as im:
        s1 = im.convert("RGB")
    with Image.open(p / "images/stage2-raw.png") as im:
        s2 = im.convert("RGB")
    started = time.perf_counter()
    # Reuse only a complete, valid, same-contract historical score.
    reusable = None
    if e._promotion_valid(p / "scoring", e.SCORING_REQUIRED, deep=True):
        provenance = read(p / "scoring/quality-provenance.json")
        rows = read(p / "scoring/comparison.json")
        if provenance.get("scientific_contract_sha256") == plan["runtime_scientific_contract"]["contract_sha256"]:
            reusable = next((r for r in rows if r.get("variant") == "stage2_raw"
                             and r.get("image_file_sha256") == sha(p / "images/stage2-raw.png")), None)
    if reusable is not None:
        row, evidence, quality_provenance = reusable, read(p / "scoring/qr-verify-evidence.json"), provenance
    else:
        settings = e._settings_for_plan(plan, c, recipe)
        quality, quality_provenance = e._quality_scores(images={"stage2_raw": s2}, prompt=c["prompt"],
                                      settings=settings, scorer=quality_scorer)
        evidence = e._score_qr_cached(plan_dir=work, image=s2, payload=c["payload"], scorer=qr_scorer, plan=plan)
        row = e._stage2_parent_row(plan=plan, candidate=c, recipe=recipe, parent_dir=p,
            parent_metadata=metadata, stage1=s1, stage2=s2, qr_evidence=evidence,
            quality=quality["stage2_raw"], stage1_quality={})
    record = {"row": row, "source_plan_sha256": sha(d / "plan.json"),
              "image_file_sha256": sha(p / "images/stage2-raw.png"), "evidence": evidence,
              "quality_provenance": quality_provenance, "reused": reusable is not None,
              "scoring_image": os.getenv("QRNIGHT_IMAGE"), "elapsed_s": time.perf_counter() - started, "at": utc()}
    write(output, record)
    return record


def score(work, cfg, shard, shards, deadline):
    validate_runtime()
    e, d, plan = load(cfg)
    inv = read(work / "inventory.json")
    good = {i["candidate_id"] for i in inv["items"] if i["generated_valid"]}
    # Balance coverage before results are known: config rounds, stable prompt shuffle.
    candidates = sorted((c for c in plan["candidates"] if c["id"] in good),
        key=lambda c: (c.get("config_index", 0), digest(c["prompt_id"])))
    candidates = candidates[shard::shards]
    from prooftag_qr.quality_scoring import quality_scorer_from_settings
    qr = e._new_qr_scorer(work, plan)
    quality = None
    done, errors = 0, 0
    started = time.time()
    try:
        for c in candidates:
            if expired(deadline - 30):
                break
            try:
                if quality is None:
                    quality = quality_scorer_from_settings(e._settings_for_plan(plan, c, e._parent_recipe(plan, c)), device="cpu")
                score_one(work, cfg, e, d, plan, c, qr, quality)
                done += 1
            except Exception as exc:
                errors += 1
                write(work / "failures" / f"score-{c['id']}.json", {"candidate_id": c["id"],
                      "type": type(exc).__name__, "error": str(exc)[:3000], "at": utc()})
                if errors >= 5 and done == 0:
                    raise RuntimeError("Scoring globalement en erreur; arrêt plutôt que faux labels") from exc
            write(work / f"score-progress-{shard}.json", {"done": done, "errors": errors,
                  "assigned": len(candidates), "elapsed_s": time.time() - started, "at": utc()})
            print(json.dumps({"shard": shard, "scored": done, "errors": errors}), flush=True)
    finally:
        qr.close()


def aggregate(work):
    rows = []
    for p in sorted((work / "scores").glob("*.json")):
        rows.append(read(p)["row"])
    write(work / "dataset/observations.json", rows)
    import csv
    fields = sorted({k for r in rows for k in r})
    out = work / "dataset/observations.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    write(work / "dataset/summary.json", {"rows": len(rows), "prompts": len({r['prompt_id'] for r in rows}),
          "valid34": sum(eligibility(r) for r in rows), "strict37": sum(eligibility(r, 37) for r in rows),
          "source_is_partial": True, "historical_complete_marker_written": False,
          "original_data_modified": False})
    return rows


def tasks(work, cfg):
    e, d, plan = load(cfg)
    from prooftag_qr.e046_catalog import ParentRecipe, PARENT_RECIPES
    from .learning import recommend
    import joblib
    prompts = cfg["demo_prompts"]
    known = {normalized_prompt(c["prompt"]) for c in plan["candidates"]}
    if any(normalized_prompt(p["prompt"]) in known for p in prompts):
        raise ValueError("Un prompt démo est déjà dans le catalogue entraînement")
    bundle = None
    modeldir = work / "model"
    if (modeldir / "model-manifest.json").is_file() and (modeldir / "training-report.json").is_file():
        manifest = read(modeldir / "model-manifest.json")
        if read(modeldir / "training-report.json").get("trained"):
            if sha(modeldir / "advisor.joblib") != manifest["advisor.joblib"]:
                raise RuntimeError("Modèle sauvegardé corrompu")
            bundle = joblib.load(modeldir / "advisor.joblib")
    allowed = set(read(work / "preflight.json")["allowed_payload_ecc"])
    recipes = [r for r in e._recipe_map(plan).values() if r.error_correction in allowed]
    queue = []
    # First seed over all prompts, then the next seed. One generation per method/case.
    for seed in cfg["seeds"]:
        for index, prompt in enumerate(prompts):
            c = {"prompt_id": prompt["id"], "prompt": prompt["prompt"], "prompt_family": "prospective_demo",
                 "prompt_group_id": prompt["id"], "payload": cfg["payload"],
                 "payload_length": len(cfg["payload"].encode("utf-8")), "seed": seed,
                 "qr_version": e.QR_VERSION}
            slate = [{**c, **e._recipe_features(r), "parent_recipe_id": r.id,
                      "error_correction": r.error_correction, "qr_mask_pattern": r.qr_mask_pattern} for r in recipes]
            variants = [("fixed_e044", PARENT_RECIPES[4], None),
                        ("paper_parameters_binary_target", PARENT_RECIPES[7], None)]
            if bundle:
                rec = recommend(bundle, slate)
                chosen = rec[0]
                r = next(r for r in recipes if r.id == chosen["candidate"]["parent_recipe_id"])
                variants.append(("advisor", r, {k: v for k, v in chosen.items() if k != "candidate"}))
                write(work / "rankings" / f"{prompt['id']}-{seed}.json", rec)
            # Counterbalance method order instead of always generating advisor last.
            n = len(variants); offset = index % n
            variants = variants[offset:] + variants[:offset]
            for method, recipe, prediction in variants:
                ident = f"{prompt['id']}-{seed}-{method.replace('_', '-')}"
                queue.append({"id": ident, "method": method, "recipe": asdict(recipe),
                     "candidate": {**c, "id": ident, "generation_group_id": ident,
                                   "parent_recipe_id": recipe.id}, "prediction": prediction})
    write(work / "generation-plan.json", {"frozen_at": utc(), "tasks": queue,
        "protocol": "one generation per method per prompt/seed, identical literal payload; raw only",
        "paper_reproduction_exact": False, "paper_difference": "binary exact QR target, public wrapper and frozen Prooftag models; QArt exact not available",
        "payload_shift_from_training": True, "phone_validated": False,
        "advisor_available": bundle is not None})
    return queue


def generate(work, cfg, task_id):
    validate_runtime()
    e, d, plan = load(cfg)
    from prooftag_qr.e046_catalog import ParentRecipe, NEGATIVE_PROMPT
    from prooftag_qr.diffqrcoder_backend import UpstreamDiffQRCoderBackend
    from prooftag_qr.schemas import GenerationRequest
    from prooftag_qr.quality_scoring import quality_scorer_from_settings
    from safetensors.torch import save_file
    import torch
    task = next(t for t in read(work / "generation-plan.json")["tasks"] if t["id"] == task_id)
    dest = work / "generated" / task_id
    if (dest / "result.json").is_file():
        return read(dest / "result.json")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "images").mkdir(exist_ok=True)
    (dest / "previews").mkdir(exist_ok=True)
    c, recipe = task["candidate"], ParentRecipe(**task["recipe"])
    settings = e._settings_for_plan(plan, c, recipe).model_copy(update={
        "srpg_save_step_previews": True, "srpg_preview_interval": 4,
        "srpg_quiet_zone_mode": "none", "srmpgd_enabled": False})
    backend = UpstreamDiffQRCoderBackend(settings)
    blueprint = e._blueprint(c, recipe)
    request = GenerationRequest(payload=c["payload"], prompt=c["prompt"], negative_prompt=NEGATIVE_PROMPT,
        backend="controlnet", error_correction=recipe.error_correction, seed=c["seed"],
        steps=recipe.stage1_steps, guidance_scale=recipe.stage1_guidance_scale,
        controlnet_scale=recipe.stage1_controlnet_scale, strength=1., max_attempts=1)
    write(dest / "task.json", task)
    started = time.perf_counter()
    qr = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA indisponible dans le Job GPU")
        e._seed_e046_strict(c["seed"])
        torch.cuda.reset_peak_memory_stats()
        s1 = backend.generate(request, blueprint, c["seed"]).convert("RGB")
        s1.save(dest / "images/stage1-raw.png")
        s2 = backend._run_stage2(s1, blueprint, request, c["seed"]).convert("RGB")
        s2.save(dest / "images/stage2-raw.png")
        state = backend.export_stage2_state()
        if state is None:
            raise RuntimeError("Latent Stage2 absent")
        save_file({"latent": state["latent"].detach().cpu().contiguous()}, str(dest / "stage2-latent.safetensors"))
        preview_names = []
        for key, image in sorted(backend._debug_artifacts.items()):
            if key.startswith("stage2_x0_estimate_step_"):
                image.save(dest / "previews" / (key + ".png")); preview_names.append(key)
        elapsed_gpu = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated())
        meta = {"source_commit": BASE_COMMIT, "runtime_image": os.getenv("QRNIGHT_IMAGE"),
                "runtime_image_digest": os.getenv("QRNIGHT_IMAGE_DIGEST"),
                "scientific_contract_sha256": plan["runtime_scientific_contract"]["contract_sha256"],
                "stage2_latent_tensor_sha256": state["latent_sha256"],
                "stage2_diagnostics": state.get("diagnostics", {}),
                "elapsed_s": elapsed_gpu, "cuda_peak_allocated_bytes": peak}
        write(dest / "parent-metadata.json", meta)
        # Release GPU model weights before CPU visual scores to avoid incidental memory peaks.
        backend._pipeline = None
        del backend, state
        gc.collect(); torch.cuda.empty_cache()
        scorer = quality_scorer_from_settings(settings, device="cpu")
        quality, provenance = e._quality_scores(images={"stage2_raw": s2}, prompt=c["prompt"], settings=settings, scorer=scorer)
        qr = e._new_qr_scorer(work, plan)
        evidence = e._score_qr_cached(plan_dir=work, image=s2, payload=c["payload"], scorer=qr, plan=plan)
        row = e._stage2_parent_row(plan=plan, candidate=c, recipe=recipe, parent_dir=dest,
           parent_metadata=meta, stage1=s1, stage2=s2, qr_evidence=evidence,
           quality=quality["stage2_raw"], stage1_quality={})
        row.update(method=task["method"], plan_id=cfg["run_id"], phone_validation="NOT_TESTED",
                   generation_seconds=elapsed_gpu, total_seconds=time.perf_counter() - started,
                   cuda_peak_allocated_bytes=peak, uniform_quiet_zone_replacement=False,
                   input_payload=c["payload"], preview_kind="real Stage2 x0 estimates, not successive final images")
        result = {"row": row, "task": task, "evidence": evidence, "quality_provenance": provenance,
                  "previews": preview_names, "at": utc(), "addon_version": VERSION}
        write(dest / "result.json", result)
        return result
    except BaseException as exc:
        write(dest / "FAILED.json", {"type": type(exc).__name__, "error": str(exc)[:4000], "at": utc()})
        raise
    finally:
        if qr:
            qr.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["preflight", "inventory", "score", "train", "tasks", "generate", "report"])
    p.add_argument("--config", default="/config/config.json")
    p.add_argument("--work", default="/night")
    p.add_argument("--deadline", type=float, default=time.time() + 600)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--task")
    a = p.parse_args(); work = Path(a.work); cfg = read(a.config)
    signal.signal(signal.SIGTERM, on_signal); signal.signal(signal.SIGINT, on_signal)
    import torch
    torch.set_num_threads(cfg["cpu_per_worker"])
    torch.set_num_interop_threads(1)
    if a.action == "preflight": preflight(work, cfg)
    elif a.action == "inventory": inventory(work, cfg, a.deadline)
    elif a.action == "score": score(work, cfg, a.shard, a.shards, a.deadline)
    elif a.action == "train":
        aggregate(work)
        from .learning import train
        train(work, cfg, a.deadline)
    elif a.action == "tasks": tasks(work, cfg)
    elif a.action == "generate": generate(work, cfg, a.task)
    elif a.action == "report":
        aggregate(work)
        from .report import report
        report(work, cfg, a.deadline)


if __name__ == "__main__":
    main()
