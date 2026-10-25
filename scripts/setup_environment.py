"""Record the hardware/software environment used for the experiments.

Writes docs/environment_info.json.  (Create the venv first:
    python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt
and install torch from the cu126 index, see requirements.txt.)
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    info = {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "machine": platform.machine()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, gpu_vram_gb=round(p.total_memory / 1e9, 2),
                        gpu_capability=f"{p.major}.{p.minor}", cuda_version=torch.version.cuda,
                        cudnn=torch.backends.cudnn.version())
    except ImportError:
        info["torch"] = None
    try:
        import os
        import ctypes

        class MEM(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEM(); m.dwLength = ctypes.sizeof(MEM)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        info["ram_gb"] = round(m.ullTotalPhys / 1024 ** 3, 1)
        info["logical_cpus"] = os.cpu_count()
    except Exception:
        pass
    info["disk_free_gb"] = round(shutil.disk_usage(ROOT).free / 1e9, 1)
    info["git"] = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
    pkgs = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "optuna", "kagglehub", "pyarrow",
            "python-docx", "nbclient", "pytest"]
    info["packages"] = {}
    for p in pkgs:
        try:
            info["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info["packages"][p] = None
    out = ROOT / "docs/environment_info.json"
    out.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
