"""Download the four Kaggle datasets used by ClimateSense AI and record provenance.

Usage:  python scripts/download_datasets.py
Requires KAGGLE_API_TOKEN in the environment or in a local .env file.
Idempotent: kagglehub caches downloads; files are mirrored into data/raw/<name>/.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATASETS = {
    "weather": "nelgiriyewithana/global-weather-repository",
    "drought_us": "cdminix/us-drought-meteorological-data",
    "world_cities": "juanmah/world-cities",
    "historical_temperature": "berkeleyearth/climate-change-earth-surface-temperature-data",
}


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main() -> None:
    load_env()
    import kagglehub

    manifest = {"downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "kagglehub_version": kagglehub.__version__, "datasets": {}}
    for name, handle in DATASETS.items():
        dest = ROOT / "data" / "raw" / name
        entry = {"handle": handle, "url": f"https://www.kaggle.com/datasets/{handle}"}
        for attempt in range(3):
            try:
                cache = Path(kagglehub.dataset_download(handle))
                # Some daily-updated Kaggle versions are published empty; fall back
                # to the most recent earlier version that actually contains files.
                fallback = 0
                while not any(f.suffix == ".csv" for f in cache.rglob("*")) and fallback < 10:
                    fallback += 1
                    version = int(cache.name) - 1 if cache.name.isdigit() else None
                    if version is None:
                        break
                    entry.setdefault("skipped_empty_versions", []).append(int(cache.name))
                    cache = Path(kagglehub.dataset_download(f"{handle}/versions/{version}"))
                break
            except Exception as exc:  # network / auth errors
                entry["error"] = repr(exc)
                time.sleep(5 * (attempt + 1))
        else:
            entry["status"] = "failed"
            manifest["datasets"][name] = entry
            print(f"[FAIL] {name}: {entry['error']}")
            continue
        entry.pop("error", None)
        dest.mkdir(parents=True, exist_ok=True)
        files = []
        for f in sorted(cache.rglob("*")):
            if f.is_file():
                rel = f.relative_to(cache)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size != f.stat().st_size:
                    shutil.copy2(f, target)
                files.append({"file": str(rel).replace("\\", "/"), "bytes": f.stat().st_size})
        entry.update(status="ok", cache_path=str(cache), version=cache.name, files=files)
        manifest["datasets"][name] = entry
        print(f"[OK] {name}: {len(files)} files -> {dest}")

    out = ROOT / "data" / "manifests" / "download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print("manifest:", out)


if __name__ == "__main__":
    main()
