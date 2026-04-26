class FakeAdapter:
    def train(self):
        return {"status": "trained", "model_path": "fake/model.pkl"}

    def evaluate(self):
        return {
            "primary_metric": "macro_f1",
            "metrics": {"macro_f1": 0.75},
        }

    def package(self):
        return {"status": "packaged", "artifact_path": "fake/artifact"}

    def smoke(self):
        return {"status": "passed"}


def test_fake_adapter_lifecycle_smoke():
    adapter = FakeAdapter()

    train_result = adapter.train()
    eval_result = adapter.evaluate()
    package_result = adapter.package()
    smoke_result = adapter.smoke()

    assert train_result["status"] == "trained"
    assert eval_result["primary_metric"] == "macro_f1"
    assert eval_result["metrics"]["macro_f1"] == 0.75
    assert package_result["status"] == "packaged"
    assert smoke_result["status"] == "passed"
