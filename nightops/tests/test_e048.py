from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


def _install_host_test_shims() -> None:
    """Allow dependency-light host test runners to import Linux/container modules.

    The E048 production supervisor only runs on Debian and the E048 worker only runs
    inside the pinned QR container. Windows lacks ``fcntl`` and pcIA host Python
    intentionally has no Pillow. These shims are test-only and deliberately raise if
    code tries to *use* the unavailable dependency, so they cannot turn a runtime
    dependency failure into a passing behavioral test.
    """
    try:
        import fcntl as _fcntl  # noqa: F401
    except ModuleNotFoundError:
        shim = types.ModuleType("fcntl")
        shim.LOCK_EX = 2
        shim.LOCK_NB = 4
        shim.LOCK_UN = 8

        def _missing_flock(*_args, **_kwargs):
            raise RuntimeError("fcntl indisponible: shim réservé aux tests d'import")

        shim.flock = _missing_flock
        sys.modules["fcntl"] = shim

    try:
        from PIL import Image as _image  # noqa: F401
    except ModuleNotFoundError:
        pil = types.ModuleType("PIL")

        def _module(name: str):
            module = types.ModuleType(f"PIL.{name}")

            def _missing(*_args, **_kwargs):
                raise RuntimeError(
                    f"Pillow/{name} indisponible: shim réservé aux tests d'import"
                )

            module.__getattr__ = lambda _attr: _missing
            return module

        pil.Image = _module("Image")
        pil.ImageDraw = _module("ImageDraw")
        pil.ImageFont = _module("ImageFont")
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = pil.Image
        sys.modules["PIL.ImageDraw"] = pil.ImageDraw
        sys.modules["PIL.ImageFont"] = pil.ImageFont


_install_host_test_shims()

from qrnight import e048_host, e048_worker


class E048ProfilesTests(unittest.TestCase):
    def setUp(self):
        config_path = Path(__file__).parents[1] / "e048_config.json"
        self.cfg = json.loads(config_path.read_text(encoding="utf-8"))

    def test_profiles_ordered_catalog_then_extended(self):
        profiles = e048_worker._profiles(self.cfg)
        self.assertEqual([p["id"] for p in profiles], [
            "catalog_g250_r100_i04",
            "catalog_g500_r200_i08",
            "catalog_g1000_r150_i08",
            "catalog_g500_r150_i08_visual",
            "extended_g500_i16",
            "extended_g1000_i24",
            "robust_i32",
        ])

    def test_qr_verify_every_iteration(self):
        self.assertEqual(self.cfg["e048_qr_verify_interval"], 1)
        self.assertTrue(all(p["qr_verify_interval"] == 1 for p in self.cfg["e048_profiles"]))

    def test_iterations_bounded(self):
        self.assertEqual([p["max_iterations"] for p in self.cfg["e048_profiles"]], [4, 8, 8, 8, 16, 24, 32])
        self.assertTrue(all(1 <= p["max_iterations"] <= 40 for p in self.cfg["e048_profiles"]))

    def test_high_score_tries_all_profiles(self):
        seq = e048_worker._profile_sequence(30, e048_worker._profiles(self.cfg))
        self.assertEqual(len(seq), 7)

    def test_mid_score_skips_only_gentlest_catalog_recipe(self):
        seq = e048_worker._profile_sequence(20, e048_worker._profiles(self.cfg))
        self.assertEqual([p["id"] for p in seq], [
            "catalog_g500_r200_i08", "catalog_g1000_r150_i08",
            "catalog_g500_r150_i08_visual", "extended_g500_i16",
            "extended_g1000_i24", "robust_i32",
        ])

    def test_low_score_focuses_scan_oriented_and_extended(self):
        seq = e048_worker._profile_sequence(5, e048_worker._profiles(self.cfg))
        self.assertEqual([p["id"] for p in seq], [
            "catalog_g500_r200_i08", "catalog_g1000_r150_i08",
            "extended_g500_i16", "extended_g1000_i24", "robust_i32",
        ])

    def test_strict_beats_non_strict(self):
        strict = {"strict_all": True, "wechat_exact_presets": 37, "wechat_original_exact": True,
                  "lpips_loss": .1, "mean_absolute_change": .1, "latent_delta_rms": .1}
        non = {"strict_all": False, "wechat_exact_presets": 36, "wechat_original_exact": True,
               "lpips_loss": 0, "mean_absolute_change": 0, "latent_delta_rms": 0}
        self.assertGreater(e048_worker._candidate_rank(strict), e048_worker._candidate_rank(non))

    def test_more_presets_beats_beauty_before_strict(self):
        a = {"strict_all": False, "wechat_exact_presets": 30, "wechat_original_exact": False,
             "lpips_loss": .2, "mean_absolute_change": .2, "latent_delta_rms": .2}
        b = {"strict_all": False, "wechat_exact_presets": 29, "wechat_original_exact": True,
             "lpips_loss": 0, "mean_absolute_change": 0, "latent_delta_rms": 0}
        self.assertGreater(e048_worker._candidate_rank(a), e048_worker._candidate_rank(b))

    def test_equal_qr_prefers_less_change(self):
        a = {"strict_all": True, "wechat_exact_presets": 37, "wechat_original_exact": True,
             "lpips_loss": .01, "mean_absolute_change": .01, "latent_delta_rms": .01}
        b = {"strict_all": True, "wechat_exact_presets": 37, "wechat_original_exact": True,
             "lpips_loss": .02, "mean_absolute_change": .02, "latent_delta_rms": .02}
        self.assertGreater(e048_worker._candidate_rank(a), e048_worker._candidate_rank(b))


class SourceValidationTests(unittest.TestCase):
    def make_source(self, count=3, payload="Mettez moi la note maximal SVP", duplicate=False, srmpgd=False):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        tasks = []
        for i in range(count):
            ident = "same" if duplicate else f"task-{i}"
            task = {"id": ident, "candidate": {"payload": payload}}
            tasks.append(task)
            d = root / "generated" / ident
            (d / "images").mkdir(parents=True, exist_ok=True)
            (d / "images/stage2-raw.png").write_bytes(b"png")
            (d / "stage2-latent.safetensors").write_bytes(b"latent")
            (d / "result.json").write_text(json.dumps({"row": {"diffqrcoder_srmpgd_enabled": srmpgd}}))
        (root / "generation-plan.json").write_text(json.dumps({"tasks": tasks}))
        return tmp, root

    def cfg(self, root, count):
        return {"e048_source_mount": str(root), "e048_expected_tasks": count,
                "payload": "Mettez moi la note maximal SVP"}

    def test_valid_source(self):
        tmp, root = self.make_source()
        self.addCleanup(tmp.cleanup)
        result = e048_worker.validate_source(self.cfg(root, 3))
        self.assertEqual(result["task_count"], 3)

    def test_missing_latent_rejected(self):
        tmp, root = self.make_source()
        self.addCleanup(tmp.cleanup)
        (root / "generated/task-1/stage2-latent.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "manquants"):
            e048_worker.validate_source(self.cfg(root, 3))

    def test_wrong_payload_rejected(self):
        tmp, root = self.make_source(payload="wrong")
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            e048_worker.validate_source(self.cfg(root, 3))

    def test_already_srmpgd_rejected(self):
        tmp, root = self.make_source(srmpgd=True)
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(RuntimeError, "already_srmpgd"):
            e048_worker.validate_source(self.cfg(root, 3))

    def test_wrong_task_count_rejected(self):
        tmp, root = self.make_source(count=2)
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(RuntimeError, "nombre de tâches"):
            e048_worker.validate_source(self.cfg(root, 36))

    def test_duplicate_ids_rejected(self):
        tmp, root = self.make_source(count=2, duplicate=True)
        self.addCleanup(tmp.cleanup)
        with self.assertRaisesRegex(RuntimeError, "dupliqués"):
            e048_worker.validate_source(self.cfg(root, 2))


class HostManifestTests(unittest.TestCase):
    def cfg(self):
        return {
            "image": "prooftag-qr-e048:test", "image_id": "sha256:test",
            "namespace": "qr-core", "run_id": "e048-test", "node": "pcia",
            "data_claim": "prooftag-qr-data", "cache_claim": "cache",
            "source_artifacts": "/home/paul/source", "artifacts": "/home/paul/out",
            "configmap": "e048-test-config", "tolerations": [],
        }

    def test_gpu_job_requests_one_gpu(self):
        m = e048_host.manifest(self.cfg(), "optimize", "e048-test-optimize", 9999999999, True)
        c = m["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(c["resources"]["limits"]["nvidia.com/gpu"], "1")
        self.assertEqual(m["spec"]["template"]["spec"]["runtimeClassName"], "nvidia")

    def test_source_is_read_only(self):
        m = e048_host.manifest(self.cfg(), "optimize", "e048-test-optimize", 9999999999, True)
        mounts = m["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        source = next(x for x in mounts if x["name"] == "source")
        self.assertTrue(source["readOnly"])

    def test_history_is_read_only(self):
        m = e048_host.manifest(self.cfg(), "optimize", "e048-test-optimize", 9999999999, True)
        mounts = m["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
        history = next(x for x in mounts if x["name"] == "history")
        self.assertTrue(history["readOnly"])

    def test_cpu_job_masks_cuda(self):
        m = e048_host.manifest(self.cfg(), "report", "e048-test-report", 9999999999, False)
        env = {x["name"]: x["value"] for x in m["spec"]["template"]["spec"]["containers"][0]["env"]}
        self.assertEqual(env["NVIDIA_VISIBLE_DEVICES"], "void")
        self.assertNotIn("runtimeClassName", m["spec"]["template"]["spec"])

    def test_service_has_exec_stop_recovery(self):
        cfg = self.cfg() | {"install": "/opt/test", "start_epoch": 1000, "stop_epoch": 2000}
        units = e048_host.unit_files(cfg)
        service = units["e048-test.service"]
        self.assertIn("ExecStopPost=/usr/bin/python3 -B -m qrnight.e048_host recover", service)

    def test_guard_timer_exists(self):
        cfg = self.cfg() | {"install": "/opt/test", "start_epoch": 1000, "stop_epoch": 2000}
        units = e048_host.unit_files(cfg)
        self.assertIn("e048-test-guard.timer", units)
        self.assertIn("OnUnitInactiveSec=60s", units["e048-test-guard.timer"])

    def test_no_service_account_token(self):
        m = e048_host.manifest(self.cfg(), "optimize", "e048-test-optimize", 9999999999, True)
        self.assertFalse(m["spec"]["template"]["spec"]["automountServiceAccountToken"])

    def test_no_pull(self):
        m = e048_host.manifest(self.cfg(), "optimize", "e048-test-optimize", 9999999999, True)
        c = m["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(c["imagePullPolicy"], "Never")


class StaticReleaseTests(unittest.TestCase):
    def test_dockerfile_is_additive(self):
        text = (Path(__file__).parents[1] / "Dockerfile.e048").read_text()
        self.assertIn("ARG BASE_IMAGE", text)
        self.assertIn("FROM ${BASE_IMAGE}", text)
        self.assertIn("COPY qrnight", text)
        self.assertNotIn("pip install", text)

    def test_shell_wrapper_uses_python3(self):
        text = (Path(__file__).parents[2] / "scripts/e048-srmpgd.sh").read_text()
        self.assertIn("python3 -B -m qrnight.e048_host", text)


class HardeningTests(unittest.TestCase):
    def setUp(self):
        config_path = Path(__file__).parents[1] / "e048_config.json"
        self.cfg = json.loads(config_path.read_text(encoding="utf-8"))

    def test_gpu_canary_is_configured(self):
        self.assertGreaterEqual(int(self.cfg["gpu_canary_timeout_seconds"]), 300)
        self.assertLessEqual(int(self.cfg["gpu_canary_timeout_seconds"]), 1800)

    def test_scientific_contract_covers_srmpgd_dependencies(self):
        required = {
            "srmpgd.py", "e035_loss_fidelity.py", "e035_losses.py",
            "e035_parent_artifact.py", "e036_trust_region.py",
            "e039_limiter_scanaware.py", "guidance.py", "qr.py", "quality.py",
        }
        self.assertEqual(set(e048_worker.E048_SCIENTIFIC_BLOBS), required)
        self.assertTrue(all(len(v) == 40 for v in e048_worker.E048_SCIENTIFIC_BLOBS.values()))

    def test_canary_action_is_gpu_job(self):
        cfg = {
            "image": "prooftag-qr-e048:test", "image_id": "sha256:test",
            "namespace": "qr-core", "run_id": "e048-test", "node": "pcia",
            "data_claim": "prooftag-qr-data", "cache_claim": "cache",
            "source_artifacts": "/home/paul/source", "artifacts": "/home/paul/out",
            "configmap": "e048-test-config", "tolerations": [],
        }
        manifest = e048_host.manifest(cfg, "canary", "e048-test-canary", 9999999999, True)
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        self.assertIn("qrnight.e048_worker", container["command"])
        self.assertIn("canary", container["command"])
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], "1")

    def test_host_requires_exact_e047_base_ancestor(self):
        self.assertEqual(e048_host.EXPECTED_BASE_COMMIT, "7de284647bc526d1b206e2be21614d724f51e29e")
        source = Path(e048_host.__file__).read_text(encoding="utf-8")
        self.assertIn('"merge-base", "--is-ancestor", EXPECTED_BASE_COMMIT', source)
        self.assertIn('"diff", "--quiet", EXPECTED_BASE_COMMIT', source)
        self.assertIn('cfg.get("repo_head") != EXPECTED_BASE_COMMIT', source)

    def test_worker_uses_diffusion_offload_and_real_canary(self):
        source = Path(e048_worker.__file__).read_text(encoding="utf-8")
        self.assertIn("_offload_diffusion_modules", source)
        self.assertIn("def gpu_canary", source)
        self.assertIn("optimize_one(canary_root", source)
        self.assertIn("GPU_CANARY_PASS.json", source)

    def test_host_canary_precedes_long_optimization(self):
        source = Path(e048_host.__file__).read_text(encoding="utf-8")
        self.assertLess(source.index('"phase": "GPU_CANARY"'), source.index('"phase": "SRMPGD_GPU"'))
        self.assertIn('canary_state != "success"', source)



if __name__ == "__main__":
    unittest.main()
