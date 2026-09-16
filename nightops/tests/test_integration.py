"""Small CPU integration test with clearly SYNTHETIC observations, never claimed as project data."""
import tempfile
import time
from pathlib import Path
import unittest

from qrnight.common import write, read
from qrnight.learning import train, recommend


class IntegrationTests(unittest.TestCase):
    def test_training_serialization_partitions_and_report(self):
        import joblib
        from qrnight.report import report
        rows = []
        for g in range(40):
            for c in range(4):
                valid = (g + c) % 3 != 0
                rows.append({"candidate_id":f"p{g}c{c}","prompt_id":f"p{g}","prompt":f"synthetic still life object {g}",
                    "payload":f"test-{g}","payload_sha256":f"payload{g}","generation_group_id":f"p{g}c{c}",
                    "image_sha256":f"image{g}-{c}","image_path":"/nonexistent/SYNTHETIC.png",
                    "primary_training_observation":True,"source_kind":"parent","raw":True,
                    "visual_guard_pass":True,"wechat_original_exact":bool(valid),"wechat_exact_presets":37 if valid else 10,
                    "payload_length":10,"qr_version":3,"stage1_steps":40,"stage1_controlnet_scale":1+c*.1,
                    "error_correction":"M","qr_mask_pattern":c,"clip_aesthetic":4+g*.01+c*.1,
                    "clip_score":.5+c*.03,"hpsv2_1":.2,"module_error_rate":.1})
        cfg = {"min_training_rows":128,"min_training_groups":16,"minimum_class_count":12,
               "cpu_per_worker":2,"trees":16,"source_plan_id":"SYNTHETIC","payload":"test","max_showcase":3}
        with tempfile.TemporaryDirectory() as td:
            w=Path(td);write(w/"dataset/observations.json",rows)
            result=train(w,cfg,time.time()+300)
            self.assertTrue(result["trained"])
            self.assertTrue(result["serialization_roundtrip"])
            s=read(w/"model/splits.json")
            groups={k:{r["group"] for r in v} for k,v in s.items()}
            self.assertFalse(groups["fit"]&groups["test"])
            self.assertFalse(groups["calibration"]&groups["test"])
            self.assertFalse(groups["fit"]&groups["calibration"])
            bundle=joblib.load(w/"model/advisor.joblib")
            self.assertEqual(len(recommend(bundle,rows[:2])),2)
            report(w,cfg,time.time()+300)
            self.assertTrue((w/"report/index.html").is_file())
            self.assertEqual(read(w/"report/best/manifest.json"),[])
            self.assertGreaterEqual(len(list((w/"report/charts").glob("*.png"))),5)


if __name__ == "__main__": unittest.main()
