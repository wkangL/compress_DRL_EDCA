"""Create a verifiable recovery bundle without modifying historical runs."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/archive/optimization_recovery_20261020"

SOURCE_DIRS = ("configs", "experiments", "models", "ns3_ext", "reporting", "rl", "scripts", "sim", "tests")
SOURCE_FILES = ("README_CN.md", "project_config.py", "requirements.txt", "run.py", "run_full.ps1", "run_quick.ps1")
FORMAL_RUNS = (
    "results/raw/baseline_absolute_full_20261019",
    "results/raw/full_absolute_monitored_20261019_153000",
    "results/raw/full_absolute_resume98_20261020",
    "results/summary/full_absolute_resume98_20261020",
    "results/figures/full_absolute_resume98_20261020",
    "results/figures/baseline_absolute_full_20261019",
)
CHECKPOINTS = (
    "results/raw/full_absolute_monitored_20261019_153000/train/checkpoint.pt",
    "results/raw/full_absolute_resume98_20261020/train/checkpoint.pt",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_files() -> list[Path]:
    paths = [ROOT / p for p in SOURCE_FILES]
    for folder in SOURCE_DIRS:
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return sorted(set(paths))


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "root": str(ROOT),
        "git_available": (ROOT / ".git").exists(),
        "python": sys.version,
        "platform": platform.platform(),
        "formal_runs": [],
        "source_files": {},
        "checkpoints": {},
    }
    with zipfile.ZipFile(DEST / "source_and_reports.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in source_files():
            rel = path.relative_to(ROOT).as_posix()
            archive.write(path, rel)
            manifest["source_files"][rel] = {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted((ROOT / "reports").glob("*")):
            if path.is_file():
                rel = path.relative_to(ROOT).as_posix()
                archive.write(path, rel)
                manifest["source_files"][rel] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    (DEST / "environment_packages.txt").write_text(run([sys.executable, "-m", "pip", "freeze"]) + "\n", encoding="utf-8")
    for rel in FORMAL_RUNS:
        folder = ROOT / rel
        files = [p for p in folder.rglob("*") if p.is_file()]
        manifest["formal_runs"].append({
            "path": rel,
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
        })
    checkpoint_dir = DEST / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for rel in CHECKPOINTS:
        source = ROOT / rel
        target = checkpoint_dir / (source.parents[1].name + "__checkpoint.pt")
        shutil.copy2(source, target)
        manifest["checkpoints"][rel] = {
            "backup": target.relative_to(ROOT).as_posix(),
            "bytes": source.stat().st_size,
            "sha256": digest(source),
            "backup_sha256": digest(target),
        }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (DEST / "README.md").write_text(
        "# 2026-10-20 优化前恢复归档\n\n"
        "`source_and_reports.zip` 保存代码、配置、测试和报告；`checkpoints/` 保存中断前与最终续跑检查点。"
        "`manifest.json` 记录 SHA256、环境和正式结果目录规模。原始事件、CSV、日志和图片的独立压缩包由同目录的 "
        "`formal_experiment_data.tar.gz` 保存；历史源目录保持不变。\n",
        encoding="utf-8",
    )
    print(DEST)


if __name__ == "__main__":
    main()
