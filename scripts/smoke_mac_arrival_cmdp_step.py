"""One persistent-STEP smoke test for direct-MAC demand/deadline ledgers."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from sim.backend.base_backend import ControlAction
from sim.backend.tgax_ns3_backend import TgaxNs3Backend


OUTPUT = ROOT / "results/raw/mac_arrival40_gaming160_cmdp_step_smoke_20261029"


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    config = load_config(ROOT / "configs/mac_arrival40_gaming160_cmdp.yaml")
    backend = TgaxNs3Backend(config, 41, OUTPUT, 1.0)
    try:
        events = backend.step(ControlAction(0, False, 1, 2.0, 2.0, 7), 1.0)
    finally:
        backend.close()
    groups: dict[str, list[tuple[float, ...]]] = {}
    for event in events:
        groups.setdefault(event.kind, []).append(event.values)
    checks = {
        "one_second_completed": abs(backend.time - 1.0) < 1e-9,
        "eight_direction_ac_ledgers": len(groups.get("M", [])) == 8,
        "four_ac_demand_ledgers": len(groups.get("U", [])) == 4,
        "forty_sta_demand_ledgers": len(groups.get("V", [])) == 40,
        "generated_demand_positive": sum(row[1] for row in groups.get("U", [])) > 0,
        "acked_payload_nonnegative": all(row[2] >= 0 for row in groups.get("U", [])),
        "deadline_ledger_causal": all(row[13] <= row[12] for row in groups.get("M", [])),
    }
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "event_counts": {key: len(value) for key, value in sorted(groups.items())},
              "ac_ledgers": groups.get("U", []), "mac_ledgers": groups.get("M", [])}
    (OUTPUT / "smoke_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
