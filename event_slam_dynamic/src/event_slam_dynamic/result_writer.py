"""JSONL result stream writer and structured runtime logger."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict


class ResultWriter:
    """Write streaming outputs to JSONL files."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._files = {
            "static": open(run_dir / "static_tracks.jsonl", "a", encoding="utf-8"),
            "dynamic": open(run_dir / "dynamic_tracks.jsonl", "a", encoding="utf-8"),
            "debug": open(run_dir / "debug_stream.jsonl", "a", encoding="utf-8"),
            "clusters": open(run_dir / "clusters.jsonl", "a", encoding="utf-8"),
        }
        self.logger = self._build_logger(run_dir / "runtime.log")

    def _build_logger(self, log_path: Path) -> logging.Logger:
        logger = logging.getLogger(f"result_writer_{id(self)}")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    def append(self, key: str, payload: Dict[str, object]) -> None:
        if key not in self._files:
            raise KeyError(f"Unsupported stream key: {key}")
        self._files[key].write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._files[key].flush()

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        for h in list(self.logger.handlers):
            h.close()
            self.logger.removeHandler(h)
