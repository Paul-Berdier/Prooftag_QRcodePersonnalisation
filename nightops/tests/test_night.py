import copy
import json
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch

from qrnight.common import schedule, stamp, write, read, digest, eligibility, gpu_requested, within
from qrnight import host


def config():
    base = Path(__file__).resolve().parents[1]
    c = read(base / "config.json")
    c.update(run_id="qrn-test", configmap="qrn-test-config", image="test:1", image_id="sha256:" + "0" * 64,
             artifacts="/home/paul/qr-night-runs/test/artifacts", install="/opt/prooftag-qr-nightly/test")
    c.update(schedule("2026-09-16 22:00", "2026-09-17 08:00", stamp("2026-09-16 17:00")))
    return c


class CoreTests(unittest.TestCase):
    def test_deadline(self):
        c = config()
        self.assertEqual(c["stop_paris"], "2026-09-17T07:00:00+02:00")
        self.assertEqual(c["ready_epoch"] - c["stop_epoch"], 3600)

    def test_late_start_refused(self):
        with self.assertRaises(ValueError):
            schedule("2026-09-16 22:00", "2026-09-17 08:00", stamp("2026-09-16 22:01"))

    def test_ambiguous_dst_refused(self):
        with self.assertRaises(ValueError): stamp("2026-10-25 02:30")

    def test_short_window_refused(self):
        with self.assertRaises(ValueError):
            schedule("2026-09-17 06:00", "2026-09-17 08:00", stamp("2026-09-16 17:00"))

    def test_exact_payload(self):
        self.assertEqual(config()["payload"], "Mettez moi la note maximal SVP")

    def test_cpu_job_no_gpu(self):
        j = host.make_job(config(), "score", "test", time.time() + 600)
        p = j["spec"]["template"]
        self.assertFalse(gpu_requested(p))
        self.assertNotIn("runtimeClassName", p["spec"])
        self.assertFalse(p["spec"]["automountServiceAccountToken"])

    def test_gpu_job_bounded(self):
        j = host.make_job(config(), "generate", "test", time.time() + 600, gpu=True)
        self.assertTrue(gpu_requested(j["spec"]["template"]))
        self.assertLessEqual(j["spec"]["activeDeadlineSeconds"], 600)
        self.assertEqual(j["spec"]["backoffLimit"], 0)
        self.assertNotIn("nodeName", j["spec"]["template"]["spec"])

    def test_sources_readonly(self):
        j = host.make_job(config(), "generate", "test", time.time() + 600, gpu=True)
        spec = j["spec"]["template"]["spec"]
        self.assertTrue(next(v for v in spec["volumes"] if v["name"] == "source")["persistentVolumeClaim"]["readOnly"])
        self.assertTrue(next(v for v in spec["containers"][0]["volumeMounts"] if v["mountPath"] == "/data")["readOnly"])
        self.assertEqual(spec["securityContext"]["runAsUser"], 10001)

    def test_recovery_independent_unit(self):
        files = host.unit_files(Path("/tmp/test"), config())
        self.assertIn("ExecStopPost", files["qrn-test.service"])
        self.assertIn("OnUnitInactiveSec=60s", files["qrn-test-guard.timer"])
        self.assertIn("Persistent=true", files["qrn-test-guard.timer"])
        self.assertIn("2026-09-17 05:00:00 UTC", files["qrn-test-guard.timer"])

    def test_mutation_refuses_changed_deployment(self):
        snap = {"vllm": {"namespace":"vllm", "name":"vllm", "uid":"old", "template_sha256":"x"}}
        d = {"metadata":{"uid":"new", "resourceVersion":"7"}, "spec":{"template":{}}}
        with patch.object(host, "obj", return_value=d), patch.object(host, "k") as k:
            with self.assertRaises(RuntimeError): host.scale_from_snapshot(config(), snap, 1)
            k.assert_not_called()

    def test_no_foreign_job_deletion(self):
        d = {"metadata":{"labels":{"prooftag.io/night-run":"another-night"}}}
        with patch.object(host, "obj", return_value=d), patch.object(host, "k") as k:
            with self.assertRaises(RuntimeError): host.delete_owned(config(), "foreign")
            k.assert_not_called()

    def test_gate_does_not_use_aesthetic_to_compensate(self):
        row = {"raw": True, "visual_guard_pass": True, "wechat_original_exact": False,
               "wechat_exact_presets": 37, "clip_aesthetic": 10000}
        self.assertFalse(eligibility(row))
        row["wechat_original_exact"] = True
        self.assertTrue(eligibility(row, 37))
        row["uniform_quiet_zone_replacement"] = True
        self.assertFalse(eligibility(row))

    def test_atomic_json_and_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"a.json"; write(p, {"x":2}); self.assertEqual(read(p), {"x":2})
            with self.assertRaises(ValueError): within("/etc/passwd", d)


class MLTests(unittest.TestCase):
    def test_feature_allowlist(self):
        from qrnight.learning import Encoder
        rows = [{"prompt":"red wine bottle", "payload_length":30, "clip_aesthetic":9.9, "seed":52},
                {"prompt":"blue moon", "payload_length":30, "clip_aesthetic":2.2, "seed":99}]
        names = Encoder().fit(rows).get_feature_names_out()
        self.assertNotIn("clip_aesthetic", names)
        self.assertNotIn("seed", names)

    def test_duplicate_union(self):
        from qrnight.learning import prepare_rows
        base = {"primary_training_observation":True,"source_kind":"parent","raw":True,
                "visual_guard_pass":True,"wechat_original_exact":True,"wechat_exact_presets":37}
        rows = [{**base,"prompt":"one","payload":"a","image_sha256":"same"},
                {**base,"prompt":"two","payload":"b","image_sha256":"same"},
                {**base,"prompt":"two","payload":"b","image_sha256":"other"}]
        out, groups, audit = prepare_rows(rows)
        self.assertEqual(len(out),2)
        self.assertEqual(len(set(groups)),1)

    def test_conflicting_raster_excluded(self):
        from qrnight.learning import prepare_rows
        b = {"primary_training_observation":True,"source_kind":"parent","raw":True,
             "visual_guard_pass":True,"wechat_exact_presets":37,"image_sha256":"same"}
        out, g, a = prepare_rows([{**b,"wechat_original_exact":True},{**b,"wechat_original_exact":False}])
        self.assertEqual(len(out),0)
        self.assertEqual(a["label_conflicting_rasters"],1)

    def test_single_class_metrics(self):
        import numpy as np
        from qrnight.learning import metrics
        m = metrics(np.array([1,1,1]), np.array([.8,.5,.9]))
        self.assertIsNone(m["roc_auc"])

    def test_white_band_display_filter(self):
        from PIL import Image, ImageDraw
        from qrnight.report import white_frame
        im = Image.new("RGB",(100,100),"white"); ImageDraw.Draw(im).rectangle((10,10,89,89),fill="black")
        self.assertTrue(white_frame(im))
        self.assertFalse(white_frame(Image.new("RGB",(100,100),"black")))


if __name__ == "__main__":
    unittest.main()
