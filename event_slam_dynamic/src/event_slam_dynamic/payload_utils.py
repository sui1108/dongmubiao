"""Payload normalization utilities for backwards compatibility."""

from __future__ import annotations

import json
from typing import Dict


def parse_json_msg(raw: str) -> Dict[str, object]:
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def normalize_track_payload(payload: Dict[str, object], stream_name: str) -> Dict[str, object]:
    # New schema passthrough
    if "tracks" in payload:
        payload.setdefault("schema_version", "1.0")
        payload.setdefault("stream", stream_name)
        return payload

    # Legacy schema compatibility
    tracks = payload.get(stream_name, payload.get("data", payload.get("tracks", [])))
    ts = payload.get("timestamp", payload.get("ts", 0.0))
    return {
        "schema_version": "1.0",
        "stream": stream_name,
        "timestamp": ts,
        "tracks": tracks if isinstance(tracks, list) else [],
    }
