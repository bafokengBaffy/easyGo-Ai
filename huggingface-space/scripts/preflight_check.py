from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_config, model_file_map  # noqa: E402
from core.contracts import CONTRACTS  # noqa: E402
from core.model_store import ModelStore  # noqa: E402


def _check_contract_coverage() -> list[str]:
    issues: list[str] = []
    model_names = set(model_file_map().keys())
    contract_names = set(CONTRACTS.keys())
    missing_contracts = sorted(model_names - contract_names)
    if missing_contracts:
        issues.append(f"Missing contracts for models: {missing_contracts}")
    return issues


def _check_model_files(base_dir: Path) -> list[str]:
    issues: list[str] = []
    cfg = load_config(base_dir)
    for _, filename in model_file_map().items():
        path = cfg.model_dir / filename
        if not path.exists():
            issues.append(f"Missing model file: {path}")
        elif path.stat().st_size <= 0:
            issues.append(f"Empty model file: {path}")
    return issues


def _check_loadability(base_dir: Path) -> list[str]:
    issues: list[str] = []
    try:
        cfg = load_config(base_dir)
        store = ModelStore(cfg)
        store.load_all()
    except Exception as exc:  # pragma: no cover
        issues.append(f"Model load failed: {exc}")
    return issues


def run() -> int:
    base_dir = Path(__file__).resolve().parents[1]
    issues: list[str] = []
    issues.extend(_check_contract_coverage())
    issues.extend(_check_model_files(base_dir))
    issues.extend(_check_loadability(base_dir))

    if issues:
        print(json.dumps({"status": "fail", "issues": issues}, indent=2))
        return 1

    print(json.dumps({"status": "ok"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

