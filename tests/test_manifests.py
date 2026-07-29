import json
from pathlib import Path

import yaml


def test_kubernetes_manifests_and_dashboard_are_valid():
    root = Path("deploy/k8s")
    documents = []
    for path in root.glob("*.yaml"):
        documents.extend(document for document in yaml.safe_load_all(path.read_text()) if document)

    kinds = {document["kind"] for document in documents}
    assert {
        "CronJob",
        "Deployment",
        "PrometheusRule",
        "Service",
        "ServiceMonitor",
        "StatefulSet",
    } <= kinds

    dashboard_config = next(
        document
        for document in documents
        if document["kind"] == "ConfigMap"
        and document["metadata"]["name"] == "prooftag-qr-dashboard"
    )
    dashboard = json.loads(dashboard_config["data"]["prooftag-qr.json"])
    assert dashboard["uid"] == "prooftag-qr"
    assert len(dashboard["panels"]) == 46
    assert "prooftag_qr_srpg_step_diagnostic" in dashboard_config["data"][
        "prooftag-qr.json"
    ]
