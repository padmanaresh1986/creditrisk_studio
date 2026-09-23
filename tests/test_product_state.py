import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from core.projects import ProjectManager
from core.state import AppStore
from ml.models import MODEL_SPECS, build_model_pipeline
from ml.training import train_candidates
import inspect


def make_store():
    td = TemporaryDirectory()
    manager = ProjectManager(Path(td.name) / "projects")
    df = pd.DataFrame({"Default": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
    project = manager.create_project("Test Project", "train.csv", "dict.csv", df)
    store = AppStore(manager=manager)
    store.select_project(project.project_id)
    return td, manager, store, project


def test_registry_starts_empty():
    with TemporaryDirectory() as td:
        manager = ProjectManager(Path(td) / "projects")
        store = AppStore(manager=manager)
        assert store.has_models is False
        assert store.user_models() == []
        assert store.active_model_name == "No model released"


def test_model_specs_are_three_candidate_families():
    assert list(MODEL_SPECS.keys()) == ["Random Forest", "XGBoost", "L1 Logistic Regression"]


def test_model_pipeline_factory_builds_all_candidates():
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "cat": ["a", "b", "a", "b"]})
    y = pd.Series([0, 1, 0, 1])
    for name in MODEL_SPECS:
        pipe, numeric, categorical = build_model_pipeline(name, X, y)
        assert "preprocessing" in pipe.named_steps
        assert "model" in pipe.named_steps


def test_publish_models_marks_only_one_final_model_and_activates_project():
    td, manager, store, project = make_store()
    try:
        versions=[]
        for idx, name in enumerate(["Random Forest", "XGBoost", "L1 Logistic Regression"], 1):
            version = f"release-{idx}"
            artifact = Path(project.root_path) / "models" / "releases" / f"{version}.joblib"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"x")
            context = artifact.with_suffix(".json")
            context.write_text("{}", encoding="utf-8")
            store.register_model({
                "project_id": project.project_id,
                "model_name": name, "family": "test", "version": version,
                "cv_metrics": {"PR-AUC": 0.1 * idx}, "final_metrics": {"PR-AUC": 0.1 * idx},
                "threshold": 0.3, "artifact_path": str(artifact), "context_path": str(context),
            })
            versions.append(version)
        store.publish_models(["release-1"], "release-1")
        assert store.recommended_version == "release-1"
        assert store.models["release-1"].status == "Default"
        assert store.user_models()[0].version == "release-1"
        assert all(store.models[v].status == "Trained" for v in versions[1:])
        assert manager.active_project_id == project.project_id
    finally:
        td.cleanup()


def test_model_records_persist_and_reload_with_relative_paths():
    td, manager, store, project = make_store()
    try:
        artifact = Path(project.root_path) / "models" / "releases" / "m.joblib"
        context = artifact.with_suffix(".json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"x")
        context.write_text("{}", encoding="utf-8")
        store.register_model({
            "project_id": project.project_id, "model_name": "Random Forest", "family": "test",
            "version": "m1", "artifact_path": str(artifact), "context_path": str(context),
        })
        reloaded = AppStore(manager=manager)
        assert "m1" in reloaded.models
        assert reloaded.models["m1"].artifact_path == str(artifact.resolve())
    finally:
        td.cleanup()


def test_model_record_published_filter_property():
    from core.state import ModelRecord
    candidate = ModelRecord(model_id="c", model_name="M", family="f", version="c", status="Candidate")
    published = ModelRecord(model_id="p", model_name="M", family="f", version="p", status="Published")
    default = ModelRecord(model_id="d", model_name="M", family="f", version="d", status="Default")
    assert candidate.is_published is False
    assert published.is_published is True
    assert default.is_published is True


def test_user_models_exposes_only_final_model():
    td, manager, store, project = make_store()
    try:
        versions=[]
        for idx, name in enumerate(["Random Forest", "XGBoost", "L1 Logistic Regression"], 1):
            version=f"m{idx}"
            artifact=Path(project.root_path)/"models"/"releases"/f"{version}.joblib"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b"x")
            context=artifact.with_suffix(".json")
            context.write_text("{}", encoding="utf-8")
            store.register_model({"project_id":project.project_id,"model_name":name,"family":"test","version":version,
                                  "cv_metrics":{"PR-AUC":0.1*idx},"final_metrics":{"PR-AUC":0.1*idx},
                                  "artifact_path":str(artifact),"context_path":str(context)})
            versions.append(version)
        store.set_user_project(project.project_id, versions[1])
        visible=store.user_models()
        assert [r.version for r in visible] == [versions[1]]
        assert store.models[versions[1]].status == "Default"
        assert all(store.models[v].status == "Trained" for v in (versions[0], versions[2]))
    finally:
        td.cleanup()


def test_training_engine_accepts_selected_model_names():
    params=inspect.signature(train_candidates).parameters
    assert "model_names" in params
    assert params["model_names"].default is None
