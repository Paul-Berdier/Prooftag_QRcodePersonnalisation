from prooftag_qr.e046_campaign import _pareto_front, _row_rank


def _row(name, exact, aesthetic, hps, lpips=0.01, safe=True, original=False, qz=False):
    return {
        "candidate_id": name,
        "variant": name,
        "eligible_final": safe,
        "visual_guard_pass": safe,
        "wechat_exact_presets": exact,
        "wechat_original_exact": original,
        "quiet_zone_delivery_guard_pass": qz,
        "quiet_zone_variant": "scene_preserving" if qz else "raw",
        "clip_aesthetic": aesthetic,
        "hpsv2_1": hps,
        "clip_score": 0.5,
        "lpips": lpips,
        "module_error_rate": 0.1,
    }


def test_rank_prioritizes_wechat_after_visual_guard():
    high_scan = _row("scan", 20, 4.0, 0.2)
    pretty = _row("pretty", 5, 6.0, 0.4)
    unsafe = _row("unsafe", 37, 8.0, 0.8, safe=False)
    ordered = sorted([pretty, unsafe, high_scan], key=_row_rank)
    assert ordered[0]["candidate_id"] == "scan"
    assert ordered[-1]["candidate_id"] == "unsafe"


def test_pareto_keeps_scan_and_aesthetic_tradeoff():
    rows = [
        _row("scan", 30, 4.0, 0.20),
        _row("pretty", 15, 6.0, 0.35),
        _row("dominated", 10, 3.0, 0.10, lpips=0.04),
    ]
    front = _pareto_front(rows)
    names = {row["candidate_id"] for row in front}
    assert names == {"scan", "pretty"}


def test_raw_artwork_wins_when_wechat_and_quality_are_tied():
    raw = _row("raw", 20, 5.0, 0.2, qz=False)
    scene = _row("scene", 20, 5.0, 0.2, qz=True)
    assert sorted([scene, raw], key=_row_rank)[0]["candidate_id"] == "raw"
