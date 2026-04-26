import json


def test_registry_can_write_and_read_project_record(tmp_path):
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()

    record = {
        "project": "liar",
        "decision": "promoted",
        "primary_metric": "macro_f1",
        "metrics": {"macro_f1": 0.3247},
    }

    record_path = registry_dir / "liar_latest.json"

    record_path.write_text(json.dumps(record), encoding="utf-8")

    loaded = json.loads(record_path.read_text(encoding="utf-8"))

    assert loaded["project"] == "liar"
    assert loaded["decision"] == "promoted"
    assert loaded["primary_metric"] == "macro_f1"
    assert loaded["metrics"]["macro_f1"] == 0.3247
