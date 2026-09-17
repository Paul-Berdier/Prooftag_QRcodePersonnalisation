"""Regression tests: stdlib only, no Docker/Kubernetes mutation, no GPU."""
import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qrnight import host, immediate
from qrnight.common import read, write


def cfg():
    return {"run_id":"qrn-test-now", "namespace":"qr-core", "vllm_namespace":"vllm",
            "vllm_deployment":"vllm", "install":"/opt/prooftag-qr-nightly/test",
            **immediate.immediate_window(1800000000, 12)}


class HealthTests(unittest.TestCase):
    def test_python3_absolute_path(self):
        with patch.object(host, "k", return_value=SimpleNamespace(stdout=json.dumps(
                {"health":True,"model":"test","tiny_inference":False}))) as k:
            self.assertEqual(host.health(cfg(), False)["model"], "test")
            args = k.call_args.args
            self.assertEqual(args[args.index("--")+1], "/usr/bin/python3")
            self.assertIn("--request-timeout=105s", args)
            ast.parse(args[-1])

    def test_both_health_paths_execute_real_python_http(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                calls.append(self.path)
                self.send_response(200); self.end_headers()
                body = {"data":[{"id":"test-model"}]} if self.path.endswith("models") else {}
                self.wfile.write(json.dumps(body).encode())
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, data))
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({"choices":[{"message":{"content":"OK"}}]}).encode())
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
        def exec_in_fake_container(*args, **kwargs):
            code = args[-1].replace("http://127.0.0.1:8000", f"http://127.0.0.1:{server.server_port}")
            return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=5)
        try:
            with patch.object(host, "k", side_effect=exec_in_fake_container):
                self.assertFalse(host.health(cfg(), False)["tiny_inference"])
                self.assertTrue(host.health(cfg(), True)["tiny_inference"])
            posts = [x for x in calls if isinstance(x, tuple)]
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0][1]["model"], "test-model")
            self.assertFalse(posts[0][1]["chat_template_kwargs"]["enable_thinking"])
        finally:
            server.shutdown(); server.server_close(); t.join()

    def test_missing_interpreter_not_silenced(self):
        with patch.object(host, "k", side_effect=RuntimeError("executable not found")):
            with self.assertRaises(RuntimeError): host.health(cfg(), False)

    def test_invalid_health_fails_closed(self):
        with patch.object(host, "k", return_value=SimpleNamespace(stdout='{"health":false}')):
            with self.assertRaises(RuntimeError): host.health(cfg(), False)

    def test_prepare_validates_health_before_build(self):
        tree = ast.parse(Path(host.__file__).read_text())
        f = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
        calls = {n.func.id:n.lineno for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertLess(calls["health"], calls["build_image"])


class WindowTests(unittest.TestCase):
    def test_immediate_has_no_future_start(self):
        s = immediate.immediate_window(1800000000, 12)
        self.assertEqual(s["start_epoch"], 1800000000)
        self.assertEqual(s["ready_epoch"] - s["stop_epoch"], 3600)
        self.assertEqual(s["ready_epoch"] - s["start_epoch"], 12*3600)

    def test_oversized_and_nan_budgets_refused(self):
        for x in [4, 18, -1, float("nan"), float("inf")]:
            with self.subTest(x=x), self.assertRaises(ValueError): immediate.immediate_window(1800000000,x)

    def test_no_start_timer_guard_retained(self):
        u = immediate.immediate_units(Path("/tmp/test"), cfg())
        self.assertNotIn("qrn-test-now.timer", u)
        self.assertIn("qrn-test-now-guard.timer", u)
        self.assertIn("ExecStopPost=", u["qrn-test-now.service"])
        self.assertIn("Type=exec", u["qrn-test-now.service"])
        self.assertIn("OnUnitInactiveSec=60s", u["qrn-test-now-guard.timer"])

    def test_guard_armed_before_direct_start(self):
        calls = []
        def cmd(args, **kwargs):
            calls.append([str(x) for x in args]); return SimpleNamespace(returncode=0, stdout="active")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); r=p/"state"; r.mkdir(); u=p/"units"; u.mkdir()
            with patch.object(host,"command",side_effect=cmd), patch.object(host,"heartbeat"):
                immediate.arm_now(r,cfg(),u)
            enable=next(i for i,c in enumerate(calls) if c[:2]==["systemctl","enable"])
            start=next(i for i,c in enumerate(calls) if c[:2]==["systemctl","start"])
            self.assertLess(enable,start)
            self.assertEqual(read(r/"IMMEDIATE_AUTHORIZED.json")["run_id"], cfg()["run_id"])
            self.assertFalse((u/"qrn-test-now.timer").exists())

    def test_guard_failure_prevents_service_start(self):
        calls=[]
        def cmd(args, **kwargs):
            a=[str(x) for x in args]; calls.append(a)
            return SimpleNamespace(returncode=int(a[:2]==["systemctl","is-active"]), stdout="inactive")
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=p/"r";r.mkdir();u=p/"u";u.mkdir()
            with patch.object(host,"command",side_effect=cmd),patch.object(host,"heartbeat"):
                with self.assertRaises(RuntimeError): immediate.arm_now(r,cfg(),u)
        self.assertFalse(any(c[:2]==["systemctl","start"] for c in calls))

    def test_needs_explicit_authorization(self):
        args=SimpleNamespace(confirm="")
        with patch.object(immediate,"_start_now") as run:
            with self.assertRaises(ValueError): immediate.start_now(args)
            run.assert_not_called()


class StateTests(unittest.TestCase):
    def test_runtime_exception_has_failed_marker(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with patch.object(host,"_run_pipeline",side_effect=RuntimeError("python missing")):
                with self.assertRaises(RuntimeError): host.run(p)
            m=read(p/"FAILED.json")
            self.assertFalse(m["scientific_success"])
            self.assertIn("python missing",m["error"])

    def test_restored_is_not_scientific_success(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c=cfg();c["artifacts"]=d;write(p/"config.json",c)
            write(p/"RESTORED.json",{"vllm_was_not_stopped":True})
            out=io.StringIO()
            with contextlib.redirect_stdout(out),patch.object(host,"command"):
                host.status(p)
            self.assertIn("INTERROMPUE_OU_NON_DEMARREE",out.getvalue())

    def test_failed_marker_rejects_false_launch_success(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write(p/"FAILED.json",{"error":"test"})
            with self.assertRaises(RuntimeError):immediate.observe_start(p,cfg(),1)

    def test_no_pod_no_claim_of_start(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(host,"command",return_value=SimpleNamespace(stdout="active")):
                self.assertEqual(immediate.observe_start(Path(d),cfg(),0),"START_REQUESTED_NOT_YET_CONFIRMED")

    def test_running_inventory_is_acknowledged(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);write(p/"STARTED",{});write(p/"jobs/qrn-test-now-inventory.json",{})
            pods={"items":[{"metadata":{"name":"test-pod"},"status":{"phase":"Running"}}]}
            with patch.object(host,"command",return_value=SimpleNamespace(stdout="active")),patch.object(host,"obj",return_value=pods):
                self.assertEqual(immediate.observe_start(p,cfg(),1),"STARTED_WITH_INVENTORY_POD")
            self.assertTrue(read(p/"LAUNCH_ACK.json")["not_a_scientific_completion"])

    def test_terminal_old_run_not_reset(self):
        src=Path(immediate.__file__).read_text()
        self.assertNotIn('unlink()',src)
        self.assertNotIn('rmtree(',src)
        self.assertNotIn('reset --hard',src)

    def test_reuse_never_builds_or_downloads(self):
        tree=ast.parse(Path(immediate.__file__).read_text())
        calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertNotIn("build_image",calls)
        self.assertNotIn("urlopen",calls)

    def test_worker_reuse_checks_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);install=p/"old";repo=p/"repo"; install.mkdir();repo.mkdir()
            for name in immediate.WORKER_FILES:
                for b in (install,repo):
                    f=b/"nightops"/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text("test")
            old={"install":str(install),"image":"worker:test","image_id":"sha256:"+"1"*64}
            with patch.object(host,"INSTALL_ROOT",p),patch.object(host,"command",return_value=SimpleNamespace(stdout=json.dumps({"status":{"id":"sha256:"+"2"*64}}))):
                with self.assertRaises(RuntimeError):immediate.verify_worker_image(old,repo)

    def test_worker_file_mismatch_prevents_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);install=p/"old";repo=p/"repo"
            for name in immediate.WORKER_FILES:
                for b in (install,repo):
                    f=b/"nightops"/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text("test")
            (repo/"nightops/qrnight/worker.py").write_text("changed")
            with patch.object(host,"INSTALL_ROOT",p),patch.object(host,"command") as cmd:
                with self.assertRaises(RuntimeError):immediate.verify_worker_image({"install":str(install)},repo)
                cmd.assert_not_called()


if __name__=="__main__":unittest.main()

class StartNowIntegrationTests(unittest.TestCase):
    def test_full_host_flow_reuses_image_and_creates_fresh_run(self):
        """Simulated Kubernetes only: tests the complete host-side path."""
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); state=base/"state"; (state/"runs").mkdir(parents=True)
            old=state/"runs/qrn-old";old.mkdir()
            work=base/"old-artifacts";work.mkdir()
            repo=base/"repo";(repo/"nightops").mkdir(parents=True)
            defaults=json.loads((Path(host.__file__).parents[1]/"config.json").read_text())
            write(repo/"nightops/config.json",defaults);write(repo/"nightops/prompts.json",[])
            write(old/"PREPARED.json",{"preflight":"PASS"});write(old/"RESTORED.json",{"vllm_was_not_stopped":True})
            write(old/"config.json",{**defaults,"run_id":"qrn-old","artifacts":str(work)})
            write(work/"preflight.json",{"status":"PASS","plan_sha256":"same-plan"})
            # The production implementation intentionally requires /home for outputs.
            # Use a temporary /home path in this isolated local test only.
            output=Path(tempfile.mkdtemp(prefix="qrnight-test-"))
            args=SimpleNamespace(repo=str(repo),reuse_prepared_run="qrn-old",max_hours=12,
                                 cpu_workers=None,output_root=str(output))
            def kubobj(*args):
                if args[0]=="pvc":return {"status":{"phase":"Bound"},"spec":{"volumeName":"pv"}}
                if args[0]=="pv":return {"spec":{"hostPath":{"path":str(base/"data")}}}
                if args[0]=="deployment":return {"spec":{"template":{"spec":{}}}}
                raise AssertionError(args)
            def launch(r,c,action,end):
                self.assertEqual(action,"preflight")
                write(Path(c["artifacts"])/"preflight.json",{"status":"PASS","plan_sha256":"same-plan"})
                return c["run_id"]+"-preflight"
            old_hash=(old/"RESTORED.json").read_bytes()
            try:
                with contextlib.ExitStack() as stack:
                    for name,replacement in [
                        ("STATE_ROOT",state),("clock_ok",lambda:None),("idle_check",lambda c:{}),
                        ("health",lambda c,i:{"health":True,"model":"test-model","tiny_inference":i}),
                        ("obj",kubobj),("k",lambda *a,**kw:None),("launch",launch),
                        ("wait",lambda r,c,n,e:{n[0]:"success"})]:
                        stack.enter_context(patch.object(host,name,replacement))
                    stack.enter_context(patch.object(immediate,"verify_release",return_value={"head":"abc","manifest":{}}))
                    stack.enter_context(patch.object(immediate,"ensure_no_other_run"))
                    stack.enter_context(patch.object(immediate,"verify_worker_image",return_value={"image":"worker:old","image_id":"sha256:test","worker_files":{}}))
                    stack.enter_context(patch.object(immediate,"install_host",return_value=(base/"install","release")))
                    stack.enter_context(patch.object(immediate.os,"chown"))
                    # Confine test files to a temporary directory; mock only the /home prefix validation.
                    stack.enter_context(patch.object(immediate, "validate_output_root"))
                    stack.enter_context(patch.object(immediate.shutil,"disk_usage",return_value=SimpleNamespace(free=100*2**30)))
                    arm=stack.enter_context(patch.object(immediate,"arm_now"))
                    stack.enter_context(patch.object(immediate,"observe_start",return_value="STARTED_WITH_INVENTORY_POD"))
                    with contextlib.redirect_stdout(io.StringIO()):immediate._start_now(args)
                    arm.assert_called_once()
                    newr,newc=arm.call_args.args
                    self.assertNotEqual(newr,old)
                    self.assertEqual(newc["image"],"worker:old")
                    self.assertEqual(newc["launch_mode"],"immediate")
                    self.assertTrue(read(newr/"PREPARED.json")["vllm_checks"]["tiny_inference"])
                    self.assertEqual(read(state/"latest.json")["run_id"],newr.name)
                self.assertEqual((old/"RESTORED.json").read_bytes(),old_hash)
            finally:
                import shutil
                shutil.rmtree(output)
