"""Strict packet-lifecycle audit for the current absolute-load scenarios.

The script builds a trace-only observer copy of the production ns-3 program.
It does not replace the production executable or alter MAC decisions.  The
observer adds destination STA, enqueue/dequeue and per-packet DATA-attempt
fields needed for STA x AC x direction conservation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, backend_source_hash, load_config
from sim.backend.base_backend import ControlAction
from sim.backend.ns3_backend import Ns3Backend, find_binary, runtime_env
from sim.scenarios.resolver import resolve_scenario

ACS = ("vo", "vi", "be", "bk")
DROP_NAMES = {0: "queue_rejected", 1: "expired", 2: "retry_limit", 3: "qos_old"}
SCHEMA = "packet-audit-v2-schema3"


def snapshot_sources(root: Path) -> None:
    files = [ROOT / name for name in ("run.py", "project_config.py", "requirements.txt", "README_CN.md")]
    for name in ("configs", "experiments", "models", "ns3_ext", "reporting", "rl", "scripts", "sim", "tests"):
        files.extend(p for p in (ROOT / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    files = sorted(set(files))
    with zipfile.ZipFile(root / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, path.relative_to(ROOT))
    (root / "source_sha256.json").write_text(json.dumps({
        path.relative_to(ROOT).as_posix(): sha256(path) for path in files
    }, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def observer_source(source: str) -> str:
    replacements = {
        'trace << "time_s,node,event,ac,value,uid\\n"':
            'trace << "time_s,node,event,ac,value,uid,sta\\n"',
        'void Trace(uint32_t node, const std::string& event, int ac, double value, uint64_t uid=0)':
            'void Trace(uint32_t node, const std::string& event, int ac, double value, uint64_t uid=0, int32_t sta=-1)',
        "<< value << ',' << uid << '\\n'": "<< value << ',' << uid << ',' << sta << '\\n'",
        'Trace(node,"ACKED",tag.ac,delay,mpdu->GetPacket()->GetUid());':
            'Trace(node,"ACKED",tag.ac,delay,mpdu->GetPacket()->GetUid(),tag.sta);',
        'Trace(node,"DROP",tag.ac,reason,mpdu->GetPacket()->GetUid());':
            'Trace(node,"DROP",tag.ac,reason,mpdu->GetPacket()->GetUid(),tag.sta);',
        'Trace(node,"TIMEOUT",tag.ac,reason,mpdu->GetPacket()->GetUid());':
            'Trace(node,"TIMEOUT",tag.ac,reason,mpdu->GetPacket()->GetUid(),tag.sta);',
        'Trace(node,"OFFERED",ac,bytes,packet->GetUid());':
            'Trace(node,"OFFERED",ac,bytes,packet->GetUid(),sta);',
        'if(started && mpdu->GetPacket()->PeekPacketTag(tag)) ++queuedData[node][ac];':
            'if(started && mpdu->GetPacket()->PeekPacketTag(tag)) { ++queuedData[node][ac]; Trace(node,"ENQUEUED",tag.ac,tag.bytes,mpdu->GetPacket()->GetUid(),tag.sta); }',
        'if(!started || !mpdu->GetPacket()->PeekPacketTag(tag)) return;\n        NS_ABORT_MSG_IF(queuedData[node][ac]==0,"Unmatched tagged packet dequeue");':
            'if(!started || !mpdu->GetPacket()->PeekPacketTag(tag)) return;\n        Trace(node,"DEQUEUED",tag.ac,tag.bytes,mpdu->GetPacket()->GetUid(),tag.sta);\n        NS_ABORT_MSG_IF(queuedData[node][ac]==0,"Unmatched tagged packet dequeue");',
        'auto& c=counters[node][tag.ac];\n        ++c.attempts;':
            'auto& c=counters[node][tag.ac];\n        Trace(node,"DATA_ATTEMPT",tag.ac,hdr.IsRetry(),p->GetUid(),tag.sta);\n        ++c.attempts;',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError(f"Observer anchor count {source.count(old)}: {old[:90]}")
        source = source.replace(old, new)
    return source


def build_observer(root: Path) -> Path:
    source = observer_source((ROOT / "ns3_ext/wifi-control.cc").read_text(encoding="utf-8"))
    scratch = ROOT / "vendor/ns-3.44/scratch/wifi-baseline-audit.cc"
    scratch.write_text(source, encoding="utf-8")
    (root / "wifi-statistics-audit.cc").write_text(source, encoding="utf-8")
    cmake = shutil.which("cmake", path=runtime_env()["PATH"])
    with (root / "build.log").open("w", encoding="utf-8") as out:
        subprocess.run(
            [cmake, "--build", str(ROOT / "vendor/ns-3.44/cmake-cache"), "--parallel", "4",
             "--target", "scratch_wifi-baseline-audit"],
            env=runtime_env(), stdout=out, stderr=subprocess.STDOUT, check=True,
        )
    binaries = list((ROOT / "vendor/ns-3.44/build/scratch").glob("ns3.44-wifi-baseline-audit-*.exe"))
    if len(binaries) != 1:
        raise FileNotFoundError(binaries)
    return binaries[0]


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def analyze(trace_path: Path, start: float, end: float, output: Path) -> dict:
    packets: dict[int, dict] = {}
    queue_samples: list[dict] = []
    phy = Counter()
    rts = Counter(); internal = Counter()
    with trace_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            t = float(row["time_s"]); event = row["event"]; node = int(row["node"])
            uid = int(row["uid"]); ac = int(row["ac"]); sta = int(row["sta"])
            if event == "OFFERED":
                if uid in packets:
                    raise AssertionError(f"duplicate offered UID {uid}")
                packets[uid] = dict(uid=uid, created_s=t, node=node, sta=sta, ac=ac,
                                    bytes=int(float(row["value"])), enqueued=False,
                                    dequeues=0, attempts=0, retry_attempts=0, timeouts=0,
                                    attempt_times=[], retry_attempt_times=[], timeout_times=[],
                                    terminal="", terminal_s=None, delay_ms=None)
            elif event in {"ENQUEUED", "DEQUEUED", "DATA_ATTEMPT", "TIMEOUT", "ACKED", "DROP"}:
                if uid not in packets:
                    raise AssertionError(f"unmatched {event} UID {uid}")
                p = packets[uid]
                if (p["node"], p["sta"], p["ac"]) != (node, sta, ac):
                    raise AssertionError(f"identity mismatch UID {uid}")
                if event == "ENQUEUED":
                    if p["enqueued"]:
                        raise AssertionError(f"duplicate enqueue UID {uid}")
                    p["enqueued"] = True
                elif event == "DEQUEUED":
                    p["dequeues"] += 1
                elif event == "DATA_ATTEMPT":
                    retry = int(float(row["value"]))
                    p["attempts"] += 1; p["retry_attempts"] += retry
                    p["attempt_times"].append(t)
                    if retry: p["retry_attempt_times"].append(t)
                elif event == "TIMEOUT":
                    p["timeouts"] += 1; p["timeout_times"].append(t)
                else:
                    if p["terminal"]:
                        raise AssertionError(f"multiple terminal outcomes UID {uid}")
                    p["terminal_s"] = t
                    if event == "ACKED":
                        p["terminal"] = "acked"; p["delay_ms"] = float(row["value"])
                        if abs(p["delay_ms"] - (t - p["created_s"]) * 1000) > 0.002:
                            raise AssertionError(f"delay mismatch UID {uid}")
                    else:
                        p["terminal"] = DROP_NAMES.get(int(float(row["value"])), "other_drop")
            elif event == "QUEUE_PACKETS" and start <= t <= end:
                queue_samples.append(dict(time_s=t, node=node, ac=ac, packets=int(float(row["value"]))))
            elif event in {"COLLISION_RX", "OTHER_RX_FAIL"} and start < t <= end:
                p = packets.get(uid)
                key = (event, "mapped" if p else "unmapped")
                phy[key] += 1
                if p:
                    phy[(event, p["ac"], "DL" if p["node"] == 0 else "UL")] += 1
            elif event == "RTS" and start < t <= end:
                rts[("DL" if node == 0 else "UL", int(float(row["value"])))] += 1
            elif event == "INTERNAL_COLLISION" and start < t <= end:
                internal[(ac, "DL" if node == 0 else "UL")] += 1

    groups: dict[tuple, dict] = defaultdict(lambda: dict(
        active_start=0, generated=0, accepted=0, queue_rejected=0, acked=0,
        expired=0, retry_limit=0, qos_old=0, other_drop=0, unfinished_end=0,
        attempts=0, retry_attempts=0, ack_timeouts=0, delivered_bytes=0,
        generated_bytes=0, delays=[], ontime_eligible=0, ontime=0,
        late_success=0, final_failure=0, deadline_unfinished=0,
    ))
    lifecycle_rows = []
    for p in packets.values():
        created = p["created_s"]
        terminal_s = p["terminal_s"]
        alive_start = created < start and (terminal_s is None or terminal_s > start)
        generated = start <= created < end
        terminal_window = terminal_s is not None and start < terminal_s <= end
        alive_end = created < end and (terminal_s is None or terminal_s > end)
        if not (alive_start or generated or terminal_window or alive_end):
            continue
        direction = "DL" if p["node"] == 0 else "UL"
        key = (p["sta"], p["ac"], direction)
        g = groups[key]
        g["active_start"] += int(alive_start)
        if generated:
            g["generated"] += 1; g["generated_bytes"] += p["bytes"]
            g["accepted"] += int(p["enqueued"])
        g["attempts"] += sum(start < t <= end for t in p["attempt_times"])
        g["retry_attempts"] += sum(start < t <= end for t in p["retry_attempt_times"])
        g["ack_timeouts"] += sum(start < t <= end for t in p["timeout_times"])
        if terminal_window:
            g[p["terminal"]] += 1
            if p["terminal"] == "acked":
                g["delivered_bytes"] += p["bytes"]; g["delays"].append(p["delay_ms"])
        g["unfinished_end"] += int(alive_end)
        deadline_ms = 20 if p["ac"] == 0 else 50 if p["ac"] == 1 else None
        if generated and deadline_ms is not None and created <= end - deadline_ms / 1000:
            g["ontime_eligible"] += 1
            acked = p["terminal"] == "acked"
            ontime = acked and p["delay_ms"] <= deadline_ms
            g["ontime"] += int(ontime)
            g["late_success"] += int(acked and not ontime and terminal_s <= end)
            final_failure = bool(p["terminal"] and p["terminal"] != "acked" and terminal_s <= end)
            g["final_failure"] += int(final_failure)
            g["deadline_unfinished"] += int(not ontime and not (acked and terminal_s <= end) and not final_failure)
        lifecycle_rows.append({k: v for k, v in p.items()
                               if k not in ("attempt_times", "retry_attempt_times", "timeout_times")})

    rows = []
    for (sta, ac, direction), g in sorted(groups.items()):
        terminal = sum(g[k] for k in ("acked", "queue_rejected", "expired", "retry_limit", "qos_old", "other_drop"))
        if g["active_start"] + g["generated"] != terminal + g["unfinished_end"]:
            raise AssertionError(f"window conservation failed {(sta, ac, direction)}: {g}")
        if g["accepted"] + g["queue_rejected"] != g["generated"]:
            raise AssertionError(f"enqueue decision failed {(sta, ac, direction)}: {g}")
        delays = g.pop("delays")
        row = dict(sta=sta, ac=ACS[ac], direction=direction, **g,
                   terminal_total=terminal,
                   conservation_lhs=g["active_start"] + g["generated"],
                   conservation_rhs=terminal + g["unfinished_end"],
                   offered_mbps=g["generated_bytes"] * 8 / (end - start) / 1e6,
                   goodput_mbps=g["delivered_bytes"] * 8 / (end - start) / 1e6,
                   mean_delay_ms=float(np.mean(delays)) if delays else None,
                   p95_delay_ms=percentile(delays, 95), p99_delay_ms=percentile(delays, 99),
                   max_delay_ms=percentile(delays, 100),
                   ontime_fraction=g["ontime"] / g["ontime_eligible"] if g["ontime_eligible"] else None)
        rows.append(row)

    queue_grouped = defaultdict(list)
    for q in queue_samples:
        queue_grouped[(q["node"], q["ac"])].append(q["packets"])
    queue_rows = []
    for (node, ac), values in sorted(queue_grouped.items()):
        queue_rows.append(dict(node=node, direction="DL" if node == 0 else "UL", ac=ACS[ac],
                               mean_packets=float(np.mean(values)), p95_packets=percentile(values, 95),
                               max_packets=max(values), nonempty_fraction=float(np.mean(np.asarray(values) > 0)),
                               samples=len(values)))
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("accounting_by_sta_ac_direction.csv", rows), ("queue_statistics.csv", queue_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]) if data else [])
            if data: writer.writeheader(); writer.writerows(data)
    fields = list(next(iter(lifecycle_rows), {}).keys())
    with gzip.open(output / "packet_lifecycle.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields: writer.writeheader(); writer.writerows(lifecycle_rows)
    report = dict(schema=SCHEMA, measurement_start_s=start, measurement_end_s=end,
                  packets_seen=len(packets), groups=len(rows), conservation_passed=True,
                  enqueue_decision_passed=True, multiple_terminal_outcomes=0,
                  phy_failures={"|".join(map(str, k)): v for k, v in sorted(phy.items(), key=lambda x: str(x[0]))},
                  rts={"|".join(map(str, k)): v for k, v in sorted(rts.items())},
                  internal_collisions={f"{ac}|{direction}": value for (ac,direction),value in sorted(internal.items())},
                  current_internal_collision_observable=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_case(args: tuple) -> dict:
    config, scenario_name, scenario, seed, root = args
    folder = Path(root) / f"{scenario_name}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    start_time = time.monotonic()
    with Ns3Backend(config, scenario, seed, folder, trace=True) as backend:
        backend.step(ControlAction(), 10.0)
        for _ in range(60):
            backend.step(ControlAction(), 1.0)
        if abs(backend.time - 70.0) > 1e-9:
            raise AssertionError(backend.time)
    report = analyze(folder / "mac_trace.csv", 10.0, 70.0, folder / "analysis")
    metadata = dict(schema=SCHEMA, scenario=scenario_name, seed=seed, traffic_warmup_s=10,
                    measurement_s=60, runtime_s=time.monotonic() - start_time,
                    observer_binary_sha256=sha256(find_binary(config)),
                    production_source_hash=backend_source_hash(), config=config, scenario_definition=scenario)
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"case": folder.name, "runtime_s": metadata["runtime_s"], **report}), flush=True)
    return metadata | report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", default=["S2_realtime_stress", "S4_overload"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    snapshot_sources(root)
    config = load_config(ROOT / "configs/full.yaml")
    production = find_binary(config); before = sha256(production)
    observer = build_observer(root)
    audit_config = copy.deepcopy(config); audit_config["backend"]["binary"] = str(observer)
    scenarios = load_config(ROOT / "configs/scenarios.yaml")
    capacity = json.loads((ROOT / config["traffic"]["capacity_file"]).read_text(encoding="utf-8"))["C_ref_mbps"]
    scenarios = {name: resolve_scenario(definition, config["network"]["num_sta"], capacity)
                 for name, definition in scenarios.items()}
    # A trace-only observer must preserve production stdout exactly.
    check_scenario = scenarios[args.scenarios[0]]
    for name, cfg in (("production", config), ("observer", audit_config)):
        with Ns3Backend(cfg, check_scenario, args.seeds[0], root / f"equivalence_{name}", trace=True) as backend:
            backend.step(ControlAction(), 2.0)
    prod = (root / "equivalence_production/events.log").read_bytes()
    obs = (root / "equivalence_observer/events.log").read_bytes()
    if prod != obs:
        raise AssertionError("observer changed simulation stdout")
    jobs = [(audit_config, name, scenarios[name], seed, root)
            for name in args.scenarios for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_case, jobs))
    if sha256(production) != before:
        raise AssertionError("production executable changed")
    complete = dict(schema=SCHEMA, passed=True, observer_stdout_identical=True,
                    production_binary_unchanged=True, production_binary_sha256=before,
                    cases=results)
    (root / "complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "cases": len(results)}))


if __name__ == "__main__":
    main()
