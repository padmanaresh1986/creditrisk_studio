from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from threading import RLock
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = ROOT / "projects"
INDEX_PATH = PROJECTS_ROOT / "project_index.json"


@dataclass
class ProjectRecord:
    project_id: str
    name: str
    slug: str
    root_path: str
    status: str = "Draft"  # Draft | Trained | Active | Archived
    created_at_utc: str = ""
    updated_at_utc: str = ""
    dataset_filename: str = ""
    dictionary_filename: str = ""
    dataset_fingerprint: str = ""
    rows: int = 0
    columns: int = 0
    default_rate: float = 0.0
    model_versions: list[str] = field(default_factory=list)
    recommended_version: str | None = None
    recommended_model_name: str | None = None
    release_note: str = ""

    @property
    def is_active(self) -> bool:
        return self.status == "Active"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:70] or "credit-risk-project"


def _safe_filename(name: str, fallback: str) -> str:
    cleaned = Path(name).name
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", cleaned)
    return cleaned or fallback


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def dataframe_fingerprint(df) -> str:
    payload = df.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class ProjectManager:
    """Small filesystem-backed project registry for the Streamlit-only app."""

    def __init__(self, root: Path = PROJECTS_ROOT) -> None:
        self.root = root
        self.index_path = self.root / "project_index.json"
        self._lock = RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_cache: dict[str, Any] | None = None
        self._index_mtime_ns: int = -1
        self._projects_cache: list[ProjectRecord] | None = None
        self._projects_mtime_ns: int = -1
        self._model_index_cache: dict[str, list[dict[str, Any]]] = {}
        self._model_index_mtime_ns: dict[str, int] = {}
        if not self.index_path.exists():
            _atomic_json_write(self.index_path, {"active_project_id": None, "projects": []})

    def _invalidate_index_cache(self) -> None:
        self._index_cache = None
        self._index_mtime_ns = -1
        self._projects_cache = None
        self._projects_mtime_ns = -1

    def _read_index(self, force: bool = False) -> dict[str, Any]:
        try:
            mtime_ns = self.index_path.stat().st_mtime_ns
            if not force and self._index_cache is not None and mtime_ns == self._index_mtime_ns:
                return self._index_cache
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                raw.setdefault("active_project_id", None)
                raw.setdefault("projects", [])
                self._index_cache = raw
                try:
                    self._index_mtime_ns = self.index_path.stat().st_mtime_ns
                except OSError:
                    self._index_mtime_ns = -1
                return raw
        except Exception:
            pass
        raw = {"active_project_id": None, "projects": []}
        self._index_cache = raw
        self._index_mtime_ns = -1
        return raw

    def _write_index(self, index: dict[str, Any]) -> None:
        _atomic_json_write(self.index_path, index)
        self._index_cache = dict(index)
        try:
            self._index_mtime_ns = self.index_path.stat().st_mtime_ns
        except OSError:
            self._index_mtime_ns = -1
        self._projects_cache = None
        self._projects_mtime_ns = -1

    def _manifest_path(self, project_id: str) -> Path:
        return self.root / project_id / "project.json"

    def _models_index_path(self, project_id: str) -> Path:
        return self.root / project_id / "models" / "model_index.json"

    def list_projects(self) -> list[ProjectRecord]:
        with self._lock:
            try:
                mtime_ns = self.index_path.stat().st_mtime_ns
            except OSError:
                mtime_ns = -1
            if self._projects_cache is not None and mtime_ns == self._projects_mtime_ns:
                return list(self._projects_cache)
            index = self._read_index()
            projects: list[ProjectRecord] = []
            for item in index.get("projects", []):
                try:
                    projects.append(ProjectRecord(**item))
                except TypeError:
                    continue
            projects = sorted(projects, key=lambda p: p.updated_at_utc, reverse=True)
            self._projects_cache = projects
            self._projects_mtime_ns = mtime_ns
            return list(projects)

    def get_project(self, project_id: str | None) -> ProjectRecord | None:
        if not project_id:
            return None
        with self._lock:
            index = self._read_index()
            for item in index.get("projects", []):
                if item.get("project_id") == project_id:
                    try:
                        return ProjectRecord(**item)
                    except TypeError:
                        return None
        return None

    @property
    def active_project_id(self) -> str | None:
        with self._lock:
            return self._read_index().get("active_project_id")

    def active_project(self) -> ProjectRecord | None:
        return self.get_project(self.active_project_id)

    def create_project(self, name: str, dataset_filename: str, dictionary_filename: str, df) -> ProjectRecord:
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Project name is required.")
        project_id = f"prj-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}"
        slug = _slugify(cleaned_name)
        project_root = self.root / f"{slug}__{project_id}"
        (project_root / "data").mkdir(parents=True, exist_ok=False)
        (project_root / "models" / "candidates").mkdir(parents=True, exist_ok=True)
        (project_root / "models" / "releases").mkdir(parents=True, exist_ok=True)
        (project_root / "analysis").mkdir(parents=True, exist_ok=True)
        (project_root / "runs").mkdir(parents=True, exist_ok=True)
        (project_root / "exports").mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        project = ProjectRecord(
            project_id=project_id,
            name=cleaned_name,
            slug=slug,
            root_path=str(project_root.resolve()),
            created_at_utc=now,
            updated_at_utc=now,
            dataset_filename=_safe_filename(dataset_filename, "training.csv"),
            dictionary_filename=_safe_filename(dictionary_filename, "data_dictionary.csv"),
            rows=int(len(df)),
            columns=int(len(df.columns)),
            default_rate=float(df["Default"].mean()) if "Default" in df.columns else 0.0,
            dataset_fingerprint=dataframe_fingerprint(df),
        )
        _atomic_json_write(project_root / "project.json", asdict(project))
        self._persist_project(project)
        return project

    def _persist_project(self, project: ProjectRecord) -> None:
        with self._lock:
            index = self._read_index()
            items = [p for p in index.get("projects", []) if p.get("project_id") != project.project_id]
            items.append(asdict(project))
            index["projects"] = items
            self._write_index(index)
            _atomic_json_write(Path(project.root_path) / "project.json", asdict(project))

    def update_project(self, project_id: str, **updates: Any) -> ProjectRecord:
        with self._lock:
            project = self.get_project(project_id)
            if project is None:
                raise KeyError(project_id)
            for key, value in updates.items():
                if hasattr(project, key):
                    setattr(project, key, value)
            project.updated_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self._persist_project(project)
            return project

    def copy_uploads(self, project_id: str, training_bytes: bytes, training_filename: str, dictionary_bytes: bytes, dictionary_filename: str) -> tuple[str, str]:
        project = self.get_project(project_id)
        if project is None:
            raise KeyError(project_id)
        data_dir = Path(project.root_path) / "data"
        train_name = _safe_filename(training_filename, "training.csv")
        dict_name = _safe_filename(dictionary_filename, "data_dictionary.csv")
        train_path = data_dir / train_name
        dict_path = data_dir / dict_name
        train_path.write_bytes(training_bytes)
        dict_path.write_bytes(dictionary_bytes)
        self.update_project(project_id, dataset_filename=train_name, dictionary_filename=dict_name)
        return str(train_path), str(dict_path)

    def model_index(self, project_id: str) -> list[dict[str, Any]]:
        """Read one project's model index with an mtime-aware in-process cache."""
        path = self._models_index_path(project_id)
        if not path.exists():
            self._model_index_cache[project_id] = []
            self._model_index_mtime_ns[project_id] = -1
            return []
        try:
            mtime_ns = path.stat().st_mtime_ns
            cached = self._model_index_cache.get(project_id)
            if cached is not None and self._model_index_mtime_ns.get(project_id) == mtime_ns:
                return cached
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else []
            self._model_index_cache[project_id] = rows
            self._model_index_mtime_ns[project_id] = mtime_ns
            return rows
        except Exception:
            return self._model_index_cache.get(project_id, [])

    def save_model_index(self, project_id: str, rows: list[dict[str, Any]]) -> None:
        _atomic_json_write(self._models_index_path(project_id), rows)
        self._model_index_cache[project_id] = list(rows)
        try:
            self._model_index_mtime_ns[project_id] = self._models_index_path(project_id).stat().st_mtime_ns
        except OSError:
            self._model_index_mtime_ns[project_id] = -1
        project = self.get_project(project_id)
        if project:
            project.model_versions = [str(r.get("version")) for r in rows if r.get("version")]
            self._persist_project(project)

    def append_run(self, project_id: str, run_id: str, summary: dict[str, Any], logs: list[str]) -> None:
        project = self.get_project(project_id)
        if project is None:
            raise KeyError(project_id)
        run_dir = Path(project.root_path) / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(run_dir / "run.json", summary)
        (run_dir / "console.log").write_text("\n".join(logs), encoding="utf-8")
        self.update_project(project_id)

    def save_upload_analysis(self, project_id: str, name: str, payload: Any) -> None:
        project = self.get_project(project_id)
        if project is None:
            raise KeyError(project_id)
        _atomic_json_write(Path(project.root_path) / "analysis" / f"{name}.json", payload)

    def activate_project(self, project_id: str, recommended_version: str) -> ProjectRecord:
        with self._lock:
            project = self.get_project(project_id)
            if project is None:
                raise KeyError(project_id)
            versions = {str(v) for v in project.model_versions}
            if recommended_version not in versions:
                raise ValueError("Recommended model is not part of the selected project.")
            index = self._read_index()
            for item in index.get("projects", []):
                if item.get("project_id") == project_id:
                    item["status"] = "Active"
                    item["recommended_version"] = recommended_version
                    item["release_note"] = "Active user-facing project"
                elif item.get("status") == "Active":
                    item["status"] = "Archived"
            index["active_project_id"] = project_id
            self._write_index(index)
            updated = self.get_project(project_id)
            assert updated is not None
            return updated

    def archive_project(self, project_id: str) -> None:
        project = self.get_project(project_id)
        if project:
            self.update_project(project_id, status="Archived")

    def list_runs(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for project in self.list_projects():
            runs_dir = Path(project.root_path) / "runs"
            if not runs_dir.exists():
                continue
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                run_file = run_dir / "run.json"
                if run_file.exists():
                    try:
                        row = json.loads(run_file.read_text(encoding="utf-8"))
                        if isinstance(row, dict):
                            rows.append(row)
                    except Exception:
                        continue
        return sorted(rows, key=lambda r: str(r.get("Started/Recorded UTC", "")), reverse=True)

    def project_summary(self) -> list[dict[str, Any]]:
        active_id = self.active_project_id
        rows=[]
        for p in self.list_projects():
            rows.append({
                "Project": p.name,
                "Status": p.status,
                "Models": len(p.model_versions),
                "Rows": p.rows,
                "Default rate": p.default_rate,
                "Created": p.created_at_utc,
                "Active for Users": "Yes" if p.project_id == active_id else "No",
                "Project ID": p.project_id,
            })
        return rows


_MANAGER: ProjectManager | None = None


def get_project_manager() -> ProjectManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ProjectManager()
    return _MANAGER
