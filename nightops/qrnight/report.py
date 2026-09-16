"""Charts, exact-raster showcase, actual diffusion previews, and an offline report."""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
import html
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import imageio_ffmpeg
from .common import read, write, sha, eligibility, utc


def white_frame(image: Image.Image) -> bool:
    """Conservative DISPLAY filter, not a QR validity test. Does not edit pixels."""
    a = np.asarray(image.convert("RGB"))
    border = max(2, round(min(a.shape[:2]) * .04))
    strips = (a[:border], a[-border:], a[:, :border], a[:, -border:])
    return all(float(np.mean(np.min(s, axis=2) >= 244)) >= .97 for s in strips)


def report(work: Path, cfg: dict, deadline: float):
    out = work / "report"; out.mkdir(parents=True, exist_ok=True)
    for name in ("charts", "best", "animations"):
        (out / name).mkdir(exist_ok=True)
    rows = read(work / "dataset/observations.json") if (work / "dataset/observations.json").exists() else []
    inv = read(work / "inventory.json") if (work / "inventory.json").exists() else {}
    generated = [read(p) for p in sorted((work / "generated").glob("*/result.json"))]
    plan = read(work / "generation-plan.json") if (work / "generation-plan.json").exists() else {"tasks": []}
    charts = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / "charts" / (name + ".png"), dpi=180)
        fig.savefig(out / "charts" / (name + ".svg"))
        plt.close(fig); charts.append(name)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["Planifiés", "Vérifiés générés", "Scorés cette reprise", "Valides ≥34", "Stricts 37/37"]
    vals = [inv.get("expected", 0), inv.get("generated", 0), len(rows),
            sum(eligibility(r) for r in rows), sum(eligibility(r, 37) for r in rows)]
    ax.bar(labels, vals); ax.set_ylabel("Parents (pas des checkpoints)")
    ax.set_title("E046 partiel : disponibilité et portes logicielles")
    ax.tick_params(axis="x", labelrotation=15)
    for i, n in enumerate(vals): ax.text(i, n, str(n), ha="center", va="bottom")
    save(fig, "01_etat_dataset")
    if rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist([r["wechat_exact_presets"] for r in rows], bins=np.arange(-.5, 38, 1))
        ax.set(xlabel="Presets exacts sur 37 (WeChat / QR-Verify)", ylabel="Images brutes",
               title="Distribution de la robustesse logicielle — pas un taux téléphone")
        save(fig, "02_robustesse")
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, subset in [("Garde franchie", [r for r in rows if eligibility(r)]),
                              ("Non retenus", [r for r in rows if not eligibility(r)])]:
            subset = [r for r in subset if r.get("clip_aesthetic") is not None]
            if subset: ax.scatter([r["wechat_exact_presets"] for r in subset],
                                 [r["clip_aesthetic"] for r in subset], label=label, alpha=.5)
        ax.set(xlabel="Presets exacts /37", ylabel="CLIP-Aesthetic (proxy)", title="Esthétique et lecture : critères distincts")
        ax.legend(); save(fig, "03_compromis")
    training = read(work / "model/training-report.json") if (work / "model/training-report.json").is_file() else {"status": "TRAINING_NOT_FINISHED"}
    if (work / "model/hyperparameter-search.json").is_file():
        search = [r for r in read(work / "model/hyperparameter-search.json") if r["complete"]]
        if search:
            fig, ax = plt.subplots(figsize=(9, 5))
            names = [f"{'Sélection' if r['selection'] else 'Toutes'} / leaf={r['min_samples_leaf']}" for r in search]
            ax.bar(names, [r["mean_brier"] for r in search]); ax.set(ylabel="Brier moyen (plus bas = mieux)", title="Optimisation sur validation groupée, pas sur le test")
            save(fig, "04_optimisation")
    if training.get("trained"):
        pred = read(work / "model/test-predictions.json")
        fig, ax = plt.subplots(figsize=(7, 5))
        for key, label in (("p_raw", "Brut"), ("p", "Calibré ou brut si calibration absente")):
            bins = defaultdict(list)
            for r in pred: bins[min(int(r[key] * 5), 4)].append(r)
            ax.plot([np.mean([r[key] for r in v]) for _, v in sorted(bins.items())],
                    [np.mean([r['observed'] for r in v]) for _, v in sorted(bins.items())], marker="o", label=label)
        ax.plot([0, 1], [0, 1], linestyle="--", label="Idéal")
        ax.set(xlabel="Probabilité prédite", ylabel="Fréquence observée", title="Calibration — test par groupes indépendant"); ax.legend()
        save(fig, "05_calibration_test")
        import joblib
        bundle = joblib.load(work / "model/advisor.joblib")
        features = read(work / "model/features.json")["selected"]
        importance = bundle["classifier"]["model"].feature_importances_
        indices = np.argsort(importance)[-15:]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh([features[i] for i in indices], importance[indices])
        ax.set(title="Importances du modèle entraîné (descriptives, non causales)", xlabel="Importance MDI")
        save(fig, "06_variables")
    # Complete paired cases only for fair policy charts; omissions explicit below.
    by_case = defaultdict(dict)
    methods = sorted({t["method"] for t in plan["tasks"]})
    for r in generated:
        task = r["task"]; c = task["candidate"]
        by_case[(c["prompt_id"], c["seed"])][task["method"]] = r["row"]
    paired = [x for x in by_case.values() if all(m in x for m in methods)] if methods else []
    if paired:
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = [np.mean([eligibility(case[m], 37) for case in paired]) for m in methods]
        ax.bar(methods, vals); ax.set_ylim(0, 1)
        ax.set(title=f"Comparaison appariée — {len(paired)} cas complets seulement", ylabel="Fraction de sorties brutes au gate strict 37/37")
        ax.tick_params(axis="x", labelrotation=15)
        save(fig, "07_comparaison_appariee")
    counts = {m: {"planned": sum(t["method"] == m for t in plan["tasks"]),
                  "evaluated": sum(x["task"]["method"] == m for x in generated),
                  "strict37": sum(x["task"]["method"] == m and eligibility(x["row"], 37) for x in generated)} for m in methods}
    summary = {"at": utc(), "source_plan": cfg["source_plan_id"], "training": training,
              "generation_counts": counts, "complete_paired_cases": len(paired),
              "phone_validation": "NOT_TESTED", "paper_exact_reproduction": False,
              "literal_demo_payload": cfg["payload"]}
    # Select only unmodified, strict software successes, one per prompt, with no uniform white band.
    candidates = sorted(rows + [r["row"] for r in generated],
                        key=lambda r: (float(r.get("clip_aesthetic") or -1), float(r.get("clip_score") or -1)), reverse=True)
    best = []; seen = set(); excluded_frame = 0
    for row in candidates:
        if expired(deadline): break
        if not eligibility(row, 37) or row["prompt_id"] in seen: continue
        source = Path(row["image_path"])
        if not source.is_file(): continue
        if row.get("image_file_sha256") and sha(source) != row["image_file_sha256"]: continue
        with Image.open(source) as im:
            if white_frame(im): excluded_frame += 1; continue
        name = f"best_{len(best)+1:02d}.png"; shutil.copyfile(source, out / "best" / name)
        card = {k: row.get(k) for k in ("prompt", "prompt_id", "payload", "parent_recipe_id", "seed", "wechat_exact_presets", "wechat_original_exact", "clip_aesthetic", "clip_score", "hpsv2_1")}
        card.update(image="best/" + name, original_file_sha256=sha(source), phone_validation="NOT_TESTED")
        best.append(card); seen.add(row["prompt_id"])
        if len(best) >= cfg["max_showcase"]: break
    summary["showcase"] = {"selected": len(best), "white_frame_excluded": excluded_frame,
                             "meaning": "software strict + raw + visual guard; not a phone certificate"}
    write(out / "summary.json", summary); write(out / "best/manifest.json", best)
    with (out / "phone-tests-to-fill.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["file_sha256", "prompt_id", "device", "app", "distance", "angle", "lighting", "decoded_exact", "observed_payload", "operator"])
        for card in best: wr.writerow([card["original_file_sha256"], card["prompt_id"], "", "", "", "", "", "", "", ""])
    animations = []
    for record in generated[:4]:
        if expired(deadline): break
        dest = Path(record["row"]["image_path"]).parent.parent
        files = sorted((dest / "previews").glob("stage2_x0_estimate_step_*.png"))
        if not files: continue
        files = files[:30] + [dest / "images/stage2-raw.png"]
        frames = []
        for i, path in enumerate(files):
            with Image.open(path) as im:
                image = im.convert("RGB"); image.thumbnail((640, 640))
                frame = Image.new("RGB", (720, 740), "white")
                frame.paste(image, ((720-image.width)//2, 50))
                draw = ImageDraw.Draw(frame)
                label = path.stem if i < len(files)-1 else "SORTIE BRUTE FINALE"
                draw.text((16, 14), label, fill="black")
                draw.text((16, 706), "Vrais apercus x0 de cet essai - pas du temps reel - telephone non teste", fill="black")
                frames.append(frame)
        name = record["task"]["id"]
        gif = out / "animations" / (name + ".gif")
        durations = [500] * len(frames); durations[-1] = 2000
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations, loop=0)
        mp4 = out / "animations" / (name + ".mp4")
        with imageio.get_writer(mp4, fps=2, codec="libx264", quality=7, macro_block_size=2) as writer:
            for frame in frames: writer.append_data(np.asarray(frame))
            for _ in range(3): writer.append_data(np.asarray(frames[-1]))
        animations.append({"id": name, "gif": str(gif.relative_to(out)), "mp4": str(mp4.relative_to(out)),
                           "actual_frames": [str(p.relative_to(work)) for p in files]})
    write(out / "animations/manifest.json", animations)
    esc = lambda x: html.escape(str(x))
    body = ['<h1>QRGen — reprise nocturne E047</h1>',
       '<p>Résultats de recherche. Aucun scan téléphone n’est déduit du score logiciel. '
       'La référence « paper_parameters_binary_target » utilise les paramètres proches du papier avec une cible QR binaire exacte : ce n’est pas une reproduction exacte de QArt.</p>',
       f'<p>Conseiller : <strong>{esc(training.get("status"))}</strong>. Cas appariés complets : {len(paired)}.</p>',
       '<h2>Comptabilité de génération</h2><pre>' + esc(counts) + '</pre>',
       '<p>Une tâche absente ou interrompue n’est ni un succès ni un échec de scan. Les graphiques de comparaison utilisent uniquement les cas entièrement évalués pour toutes les méthodes.</p>',
       '<h2>Graphiques pour le PowerPoint</h2>']
    for name in charts: body.append(f'<figure><img src="charts/{name}.png"><figcaption>{esc(name)}</figcaption></figure>')
    body.append('<h2>Meilleures sorties admissibles — images non retouchées</h2>')
    if not best: body.append('<p><strong>Aucune image ne satisfait tous les critères de sélection. Aucun exemple de remplacement n’a été fabriqué.</strong></p>')
    for card in best:
        body.append(f'<article><img src="{card["image"]}" width="480"><p>{esc(card["prompt"])}</p><pre>{esc(card)}</pre></article>')
    body.append('<h2>Aperçus de diffusion réellement enregistrés</h2>')
    for a in animations: body.append(f'<h3>{esc(a["id"])}</h3><video controls loop width="620" src="{a["mp4"]}"></video><p><a href="{a["gif"]}">GIF</a></p>')
    body.append('<h2>Sources</h2><p>Code de base : Paul-Berdier/Prooftag_QRcodePersonnalisation, commit bfa4cdce4c6333e5214af6f3468a24a5374302c1. '
                '<a href="https://arxiv.org/abs/2409.06355">DiffQRCoder, arXiv:2409.06355</a>. '
                'Les scores publiés des auteurs ne sont pas fusionnés avec ces résultats locaux.</p>')
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>QRGen nuit</title><style>body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:20px}img{max-width:100%}pre{white-space:pre-wrap;font-size:13px}article,figure{border-bottom:1px solid #ccc;padding:24px 0}video{max-width:100%}</style>' + '\n'.join(body), encoding="utf-8")
    (out / "README.txt").write_text("Ouvrir index.html. Graphiques PNG et SVG ; QR originaux sélectionnés par gate logiciel strict, jamais retouchés.\n"
        "Les animations proviennent des aperçus x0 enregistrés : pas un enregistrement temps réel.\n"
        "Les tests téléphone restent à remplir dans phone-tests-to-fill.csv.\n", encoding="utf-8")
    write(out / "manifest.json", {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"})


def expired(deadline):
    return time.time() >= deadline - 20
