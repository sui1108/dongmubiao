"""Run directory and metadata lifecycle management."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


class RunManager:
    """Create and finalize a run folder with config/session metadata."""

    def __init__(self, run_root: str, input_path: str = "", note: str = "", bridge_info: Optional[Dict[str, object]] = None):
        self.run_root = Path(os.path.expanduser(run_root))
        self.input_path = input_path
        self.note = note
        self.bridge_info = bridge_info or {}
        self.run_dir: Optional[Path] = None
        self.run_id = ""
        self.started_at = datetime.now(timezone.utc)

    def start(self, config: Dict[str, object]) -> Path:
        self.run_root.mkdir(parents=True, exist_ok=True)
        suffix = ""
        if self.input_path:
            stem = Path(self.input_path).stem[:20]
            suffix = f"_{stem}"
        base = self.started_at.strftime("%Y-%m-%d_%H-%M-%S") + suffix
        candidate = self.run_root / base
        idx = 1
        while candidate.exists():
            candidate = self.run_root / f"{base}_{idx:02d}"
            idx += 1
        candidate.mkdir(parents=True, exist_ok=False)
        (candidate / "snapshots").mkdir(exist_ok=True)

        self.run_dir = candidate
        self.run_id = candidate.name

        self._write_json(candidate / "config.json", config)
        session_info = {
            "run_id": self.run_id,
            "started_at_utc": self.started_at.isoformat(),
            "input_path": self.input_path,
            "note": self.note,
            "bridge": self.bridge_info,
            "git_commit": self._git_commit(),
        }
        self._write_json(candidate / "session_info.json", session_info)
        return candidate

    def finish(self, summary: Dict[str, object]) -> None:
        if self.run_dir is None:
            return
        ended_at = datetime.now(timezone.utc)
        session_path = self.run_dir / "session_info.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["ended_at_utc"] = ended_at.isoformat()
        session["duration_sec"] = (ended_at - self.started_at).total_seconds()
        self._write_json(session_path, session)
        self._write_json(self.run_dir / "summary.json", summary)

    def _git_commit(self) -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
            return out
        except Exception:
            return "unknown"

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, object]) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
