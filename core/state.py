from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from core.projects import get_project_manager


@dataclass
class ModelRecord:
    model_id: str
    model_name: str
    family: str
    version: str
    project_id: str = ""
    project_name: str = ""
    status: str = "Trained"  # Trained | Published | Default | Archived
    cv_metrics: dict[str, float] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.50
    artifact_path: str = ""
    context_path: str = ""
    dataset_fingerprint: str = ""
    trained_at_utc: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    feature_count: int = 0
    transformed_feature_count: int = 0
    explanation_ready: bool = False
    default_probability_base_rate: float = 0.0

    @property
    def is_published(self) -> bool:
        return self.status in {"Published", "Default"}

    @property
    def is_trained(self) -> bool:
        return self.status in {"Trained", "Published", "Default"} or bool(self.final_metrics)


class AppStore:
    """Filesystem-backed project/model registry with an in-process cache."""

    def __init__(self, manager=None) -> None:
        self._lock = RLock()
        self.manager = manager or get_project_manager()
        self.models: dict[str, ModelRecord] = {}
        self.training_runs: dict[str, dict[str, Any]] = {}
        self._runs_loaded = False
        self.active_project_id: str | None = self.manager.active_project_id
        self.current_project_id: str | None = self.active_project_id
        self.recommended_version: str | None = None
        self._hydrate_from_disk()

    def _hydrate_from_disk(self) -> None:
        """Load only lightweight project metadata at startup. Model indexes are lazy."""
        with self._lock:
            self.models.clear()
            self._hydrated_project_ids: set[str] = set()
            self._model_index_mtime_ns: dict[str, int] = {}
            self.training_runs = {}
            self._runs_loaded = False
            active = self.manager.active_project()
            if active:
                self.active_project_id = active.project_id
                self.current_project_id = active.project_id
                self.recommended_version = active.recommended_version
            else:
                self.active_project_id = None
                self.current_project_id = None
                self.recommended_version = None

    def _ensure_project_models_loaded(self, project_id: str | None) -> None:
        if not project_id:
            return
        with self._lock:
            project = self.manager.get_project(project_id)
            if project is None:
                return
            index_path = self.manager._models_index_path(project_id)
            try:
                mtime_ns = index_path.stat().st_mtime_ns
            except OSError:
                mtime_ns = -1
            if project_id in self._hydrated_project_ids and self._model_index_mtime_ns.get(project_id) == mtime_ns:
                return

            # Remove any stale in-memory records for this project before reloading.
            self.models = {v: r for v, r in self.models.items() if r.project_id != project_id}
            for row in self.manager.model_index(project_id):
                try:
                    row = dict(row)
                    row.setdefault("project_id", project.project_id)
                    row.setdefault("project_name", project.name)
                    project_root = Path(project.root_path)
                    for path_key in ("artifact_path", "context_path"):
                        value = row.get(path_key, "")
                        if value and not Path(str(value)).is_absolute():
                            row[path_key] = str((project_root / str(value)).resolve())
                    rec = ModelRecord(**row)
                    if rec.status == "Candidate" and (rec.final_metrics or rec.artifact_path):
                        rec.status = "Trained"
                    self.models[rec.version] = rec
                except TypeError:
                    continue
            self._hydrated_project_ids.add(project_id)
            self._model_index_mtime_ns[project_id] = mtime_ns

    def _ensure_all_models_loaded(self) -> None:
        for project in self.manager.list_projects():
            self._ensure_project_models_loaded(project.project_id)

    def get_model(self, version: str | None) -> ModelRecord | None:
        if not version:
            return None
        with self._lock:
            rec = self.models.get(version)
            if rec is not None:
                return rec
            active_id = self.active_project_id
            current_id = self.current_project_id
        # Prediction paths normally resolve inside the active project; keep this fast.
        for project_id in dict.fromkeys([active_id, current_id]):
            if project_id:
                self._ensure_project_models_loaded(project_id)
                rec = self.models.get(version)
                if rec is not None:
                    return rec
        # Explicit lookup fallback for admin/history paths.
        for project in self.manager.list_projects():
            if project.project_id in {active_id, current_id}:
                continue
            self._ensure_project_models_loaded(project.project_id)
            rec = self.models.get(version)
            if rec is not None:
                return rec
        return None

    def refresh(self, project_id: str | None = None) -> None:
        """Refresh lightweight project metadata; optionally hydrate one project."""
        self._hydrate_from_disk()
        if project_id:
            self._ensure_project_models_loaded(project_id)

    def ensure_runs_loaded(self) -> None:
        if self._runs_loaded:
            return
        with self._lock:
            if self._runs_loaded:
                return
            self.training_runs = {str(r.get("Run ID")): r for r in self.manager.list_runs() if r.get("Run ID")}
            self._runs_loaded = True

    def has_any_projects(self) -> bool:
        return bool(self.manager.list_projects())

    @property
    def current_project(self):
        return self.manager.get_project(self.current_project_id)

    @property
    def active_project(self):
        return self.manager.get_project(self.active_project_id)

    @property
    def has_models(self) -> bool:
        pid = self.current_project_id
        self._ensure_project_models_loaded(pid)
        with self._lock:
            return any(r.project_id == pid for r in self.models.values())

    @property
    def active_model_name(self) -> str:
        self._ensure_project_models_loaded(self.active_project_id)
        with self._lock:
            rec = self.models.get(self.recommended_version or "")
            if rec and rec.project_id == self.active_project_id:
                return rec.model_name
            return "No model released"

    @property
    def active_model_version(self) -> str:
        self._ensure_project_models_loaded(self.active_project_id)
        rec = self.models.get(self.recommended_version or "")
        return rec.version if rec and rec.project_id == self.active_project_id else "—"

    @property
    def active_threshold(self) -> float:
        self._ensure_project_models_loaded(self.active_project_id)
        rec = self.models.get(self.recommended_version or "")
        return float(rec.threshold) if rec and rec.project_id == self.active_project_id else 0.50

    @property
    def active_metrics(self) -> dict[str, float]:
        self._ensure_project_models_loaded(self.active_project_id)
        rec = self.models.get(self.recommended_version or "")
        return rec.final_metrics.copy() if rec and rec.project_id == self.active_project_id else {}

    def select_project(self, project_id: str | None) -> None:
        with self._lock:
            if project_id is not None and self.manager.get_project(project_id) is None:
                raise KeyError(project_id)
            self.current_project_id = project_id

    def register_model(self, info: dict[str, Any]) -> ModelRecord:
        with self._lock:
            project_id = str(info.get("project_id", self.current_project_id or ""))
            project = self.manager.get_project(project_id)
            if not project:
                raise ValueError("A valid project must exist before registering a model.")
            rec = ModelRecord(
                model_id=info["version"], model_name=info["model_name"], family=info.get("family", ""),
                version=info["version"], project_id=project_id, project_name=project.name,
                status=info.get("status", "Candidate"), cv_metrics=dict(info.get("cv_metrics", {})),
                final_metrics=dict(info.get("final_metrics", {})), threshold=float(info.get("threshold", 0.50)),
                artifact_path=info.get("artifact_path", ""), context_path=info.get("context_path", ""),
                dataset_fingerprint=info.get("dataset_fingerprint", ""), trained_at_utc=info.get("trained_at_utc", ""),
                params=dict(info.get("params", {})), feature_count=int(info.get("feature_count", 0)),
                transformed_feature_count=int(info.get("transformed_feature_count", 0)),
                explanation_ready=bool(info.get("explanation_ready", False)),
                default_probability_base_rate=float(info.get("default_probability_base_rate", 0.0)),
            )
            self.models[rec.version] = rec
            self._persist_project_models(project_id)
            return rec

    def replace_candidate(self, version: str, **updates: Any) -> ModelRecord:
        rec = self.get_model(version)
        if rec is None:
            raise KeyError(version)
        with self._lock:
            for key, value in updates.items():
                setattr(rec, key, value)
            self._persist_project_models(rec.project_id)
            return rec

    def _persist_project_models(self, project_id: str) -> None:
        project = self.manager.get_project(project_id)
        rows = []
        for r in self.models.values():
            if r.project_id != project_id:
                continue
            row = asdict(r)
            if project:
                project_root = Path(project.root_path)
                for path_key in ("artifact_path", "context_path"):
                    value = row.get(path_key, "")
                    if value:
                        try:
                            row[path_key] = str(Path(value).resolve().relative_to(project_root.resolve()))
                        except Exception:
                            row[path_key] = str(value)
            rows.append(row)
        self.manager.save_model_index(project_id, rows)

    def clear_project_models(self, project_id: str | None) -> None:
        if not project_id:
            return
        self._ensure_project_models_loaded(project_id)
        with self._lock:
            self.models = {v: r for v, r in self.models.items() if r.project_id != project_id}
            self._hydrated_project_ids.discard(project_id)
            self.recommended_version = self.manager.get_project(project_id).recommended_version if self.manager.get_project(project_id) else None
            self.manager.save_model_index(project_id, [])

    def publish_models(self, versions: list[str], recommended_version: str) -> None:
        """Backward-compatible wrapper that publishes exactly one final model."""
        if len(versions) != 1 or versions[0] != recommended_version:
            raise ValueError("Exactly one model must be designated as the final prediction model.")
        rec = self.get_model(recommended_version)
        if rec is None:
            raise KeyError(recommended_version)
        self.set_user_project(rec.project_id, recommended_version)

    def user_models(self) -> list[ModelRecord]:
        """Return exactly one administrator-selected final model for the active project."""
        active_id = self.active_project_id
        self._ensure_project_models_loaded(active_id)
        version = self.recommended_version
        if not active_id or not version:
            return []
        with self._lock:
            rec = self.models.get(version)
            if rec and rec.project_id == active_id and rec.status == "Default":
                return [rec]
        return []

    def current_project_models(self) -> list[ModelRecord]:
        pid = self.current_project_id
        if not pid:
            return []
        self._ensure_project_models_loaded(pid)
        return sorted([r for r in self.models.values() if r.project_id == pid], key=lambda r: r.model_name)

    def trained_models_for_project(self, project_id: str | None = None) -> list[ModelRecord]:
        pid = project_id or self.current_project_id
        if not pid:
            return []
        self._ensure_project_models_loaded(pid)
        return sorted([r for r in self.models.values() if r.project_id == pid and r.is_trained], key=lambda r: (r.model_name, r.trained_at_utc or ""))

    def latest_final_models(self, project_id: str | None = None) -> list[ModelRecord]:
        pid = project_id or self.current_project_id
        self._ensure_project_models_loaded(pid)
        latest: dict[str, ModelRecord] = {}
        for rec in self.models.values():
            if rec.project_id != pid or not rec.final_metrics or not rec.artifact_path:
                continue
            current = latest.get(rec.model_name)
            if current is None or (rec.trained_at_utc or "") > (current.trained_at_utc or ""):
                latest[rec.model_name] = rec
        return sorted(latest.values(), key=lambda r: r.model_name)

    def set_user_project(self, project_id: str, recommended_version: str | None = None) -> None:
        """Expose exactly one final model from one project to end users."""
        with self._lock:
            project = self.manager.get_project(project_id)
            if project is None:
                raise KeyError(project_id)
            self._ensure_project_models_loaded(project_id)
            if self.active_project_id and self.active_project_id != project_id:
                self._ensure_project_models_loaded(self.active_project_id)
            eligible = self.latest_final_models(project_id)
            if not eligible:
                raise ValueError("This project has no completed holdout model versions ready for User release.")
            allowed = {r.version for r in eligible}
            if recommended_version is None:
                recommended_version = max(eligible, key=lambda r: float(r.cv_metrics.get("PR-AUC", -1.0))).version
            if recommended_version not in allowed:
                raise ValueError("Final prediction model must be one of the latest evaluated models in this project.")
            previous_active = self.active_project_id
            if previous_active and previous_active != project_id:
                for rec in self.models.values():
                    if rec.project_id == previous_active and rec.status in {"Published", "Default"}:
                        rec.status = "Trained"
                self._persist_project_models(previous_active)
            for rec in self.models.values():
                if rec.project_id == project_id:
                    rec.status = "Trained"
            self.models[recommended_version].status = "Default"
            self.recommended_version = recommended_version
            self.active_project_id = project_id
            self.current_project_id = project_id
            self.manager.update_project(
                project_id,
                model_versions=project.model_versions,
                recommended_version=recommended_version,
                recommended_model_name=self.models[recommended_version].model_name,
                status="Trained",
            )
            self._persist_project_models(project_id)
            self.manager.activate_project(project_id, recommended_version)
            self.current_project_id = project_id
            self.recommended_version = recommended_version
            self.active_project_id = project_id

    def record_run(self, dataset_name: str, summary: dict[str, Any], status: str = "Completed", project_id: str | None = None, logs: list[str] | None = None) -> str:
        with self._lock:
            run_id = f"run-{uuid4().hex[:8]}"
            row = {
                "Run ID": run_id,
                "Started/Recorded UTC": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "Dataset": dataset_name, "Status": status, "Project ID": project_id or self.current_project_id,
                **summary,
            }
            self.training_runs[run_id] = row
            self._runs_loaded = True
            if project_id:
                self.manager.append_run(project_id, run_id, row, logs or [])
            return run_id


_STORE: AppStore | None = None


def get_store() -> AppStore:
    global _STORE
    if _STORE is None:
        _STORE = AppStore()
    return _STORE
