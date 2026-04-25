from __future__ import annotations

from pathlib import Path
import requests
from tqdm import tqdm


def download_raw(raw_url: str, target_path: Path, force_download: bool = False) -> Path:
    target_path = target_path.expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size > 0 and not force_download:
        print(f"[download] RAW already exists, skip: {target_path}")
        return target_path

    print(f"[download] downloading RAW from: {raw_url}")
    with requests.get(raw_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(target_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="RAW") as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    if not target_path.exists():
        raise FileNotFoundError(f"Downloaded RAW not found: {target_path}")
    if target_path.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded RAW size is 0: {target_path}")

    print(f"[download] RAW absolute path: {target_path}")
    return target_path
