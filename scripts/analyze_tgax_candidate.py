"""Summarize and validate the sole subset7 FTP-to-BK candidate run."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ACS = ("VO", "VI", "BE", "BK")
DIRECTIONS = ("DL", "UL")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: str | None, integer: bool = False):
    if value in (None, ""):
        return 0 if integer else 0.0
    return int(value) if integer else float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="candidate input directory containing flows.csv")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    flows = [r for r in read(args.input / "flows.csv") if r["bss"] == "0"]
    input_manifest = json.loads((args.input / "manifest.json").read_text(encoding="utf-8"))
    topology = [r for r in read(args.input.parent / "topology.csv")
                if r["role"] == "STA" and r["bss"] == "0"]
    associations = {r["node"]: r for r in read(args.run / "associations.csv") if r["bss"] == "0"}
    assignments: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"DL": set(), "UL": set(), "AC": set()})
    for flow in flows:
        sta = flow["sink"] if flow["direction"] == "DL" else flow["source"]
        assignments[sta][flow["direction"]].add(f"{flow['service']}({flow['model']})")
        assignments[sta]["AC"].add(flow["ac"])
    active_nodes = set(assignments)

    sta_rows = []
    for row in sorted(topology, key=lambda r: int(r["sta"])):
        node = row["node_label"]
        info = assignments.get(node, {"DL": set(), "UL": set(), "AC": set()})
        sta_rows.append({
            "sta_index": int(row["sta"]), "node": node,
            "associated": int(associations.get(node, {}).get("associated", "0")),
            "active_business_sta": int(bool(info["DL"] or info["UL"])),
            "downlink_services": "; ".join(sorted(info["DL"])),
            "uplink_services": "; ".join(sorted(info["UL"])),
            "application_acs": "; ".join(sorted(info["AC"])),
        })
    write_csv(args.output / "bss0_sta_business_assignment.csv", sta_rows)
    table = ["# BSS 0 的40个关联STA与业务分配", "",
             "`关联=是`只表示STA已关联AP 0；`活跃=是`表示该STA在本候选配置中承载应用业务。", "",
             "| STA | 关联 | 活跃 | 下行业务 | 上行业务 | 应用AC |",
             "|---:|:---:|:---:|---|---|---|"]
    for row in sta_rows:
        table.append(f"| {row['sta_index']} | {'是' if row['associated'] else '否'} | "
                     f"{'是' if row['active_business_sta'] else '否'} | "
                     f"{row['downlink_services'] or '—'} | {row['uplink_services'] or '—'} | "
                     f"{row['application_acs'] or '—'} |")
    (args.output / "bss0_sta_business_assignment.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    manifest = json.loads((args.run / "run_manifest.json").read_text(encoding="utf-8"))
    duration = float(manifest["measurement_s"])
    outcomes = {(r["direction"], r["ac"]): r for r in read(args.run / "mac_qos_outcomes.csv")
                if r["scope"] == "TARGET"}
    delays = {(r["direction"], r["ac"]): r for r in read(args.run / "mac_service_delay_summary.csv")
              if r["scope"] == "TARGET"}
    metric_rows = []
    for ac in ACS:
        for direction in DIRECTIONS:
            outcome = outcomes.get((direction, ac), {})
            delay = delays.get((direction, ac), {})
            arrived = number(outcome.get("arrived"), True)
            attempts = number(outcome.get("tx_attempts"), True)
            retransmissions = number(outcome.get("retransmissions"), True)
            metric_rows.append({
                "ac": ac, "direction": direction,
                "p95_final_mac_service_delay_ms": (delay.get("p95_ms", "")
                    if number(delay.get("successful_packets"), True) else ""),
                "successful_mac_throughput_mbps": number(outcome.get("acked_bytes"), True) * 8 / duration / 1e6,
                "final_mac_drop_rate": number(outcome.get("failed"), True) / arrived if arrived else 0,
                "retransmission_rate": retransmissions / attempts if attempts else 0,
                "arrived_mpdus": arrived,
                "acked_mpdus": number(outcome.get("acked"), True),
                "failed_mpdus": number(outcome.get("failed"), True),
                "unfinished_mpdus": number(outcome.get("unfinished"), True),
                "tx_attempts": attempts, "retransmissions": retransmissions,
            })
    write_csv(args.output / "bss0_ac_direction_metrics.csv", metric_rows)

    configured = {int(r["flow_id"]): r for r in flows}
    runtime = {int(r["flow_id"]): r for r in read(args.run / "flow_summary.csv") if r["bss"] == "0"}
    per_sta = defaultdict(lambda: {"offered": 0, "received": 0, "flows": []})
    for flow_id, flow in configured.items():
        sta = flow["sink"] if flow["direction"] == "DL" else flow["source"]
        measured = runtime.get(flow_id, {})
        per_sta[sta]["offered"] += number(measured.get("offered_bytes"), True)
        per_sta[sta]["received"] += number(measured.get("received_bytes"), True)
        per_sta[sta]["flows"].append(flow["service"])
    fairness_rows = []
    satisfactions = []
    for sta in sorted(active_nodes, key=lambda name: int(name.split("sta")[-1])):
        demand = per_sta[sta]
        satisfaction = min(demand["received"] / demand["offered"], 1.0) if demand["offered"] else 0.0
        satisfactions.append(satisfaction)
        fairness_rows.append({"node": sta, "services": ";".join(sorted(set(demand["flows"]))),
                              "offered_bytes": demand["offered"], "received_bytes": demand["received"],
                              "demand_satisfaction": satisfaction})
    write_csv(args.output / "bss0_sta_demand_satisfaction.csv", fairness_rows)
    jain = ((sum(satisfactions) ** 2) / (len(satisfactions) * sum(v * v for v in satisfactions))
            if satisfactions and any(satisfactions) else 0.0)

    tcp_ack_rows = read(args.run / "tcp_ack_ac_summary.csv")
    target_tcp_ack = [r for r in tcp_ack_rows if r["bss"] == "0"]
    activity = {(r["direction"], r["ac"]): r for r in metric_rows}
    checks = {
        "all_40_bss0_stas_associated": len(sta_rows) == 40 and all(r["associated"] for r in sta_rows),
        "candidate_component_count_matches_manifest":
            len(flows) == int(input_manifest["one_way_components_per_bss"]),
        "service_counts_match_manifest": all(
            sum(r["service"] == service for r in flows) ==
            count * (2 if service == "D6" else 1)
            for service, count in input_manifest["service_counts_per_bss"].items()),
        "all_four_application_acs_configured": {r["ac"] for r in flows} == set(ACS),
        "vo_uplink_entered_mac": activity[("UL", "VO")]["arrived_mpdus"] > 0,
        "bk_downlink_success": activity[("DL", "BK")]["acked_mpdus"] > 0,
        "bk_uplink_success": activity[("UL", "BK")]["acked_mpdus"] > 0,
        "all_expected_application_ac_directions_active": all(
            activity[key]["arrived_mpdus"] > 0 for key in
            (("DL", "VO"), ("UL", "VO"), ("DL", "VI"), ("UL", "VI"),
             ("DL", "BE"), ("DL", "BK"), ("UL", "BK"))),
        "tcp_reverse_ack_actual_ac_observed": bool(target_tcp_ack),
        "mac_packet_conservation": all(r["arrived_mpdus"] == r["acked_mpdus"] +
                                       r["failed_mpdus"] + r["unfinished_mpdus"] for r in metric_rows),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "run": str(args.run.resolve()), "measurement_s": duration,
        "active_business_sta_count": len(active_nodes),
        "associated_sta_count": len(sta_rows),
        "jain_demand_satisfaction": jain,
        "collision_metric": "retransmission_rate",
        "collision_metric_reason": "ns-3 trace does not unambiguously attribute each failed attempt to a collision",
        "tcp_ack_actual_ac": target_tcp_ack,
        "checks": checks,
    }
    (args.output / "candidate_validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# 候选配置验证：{result['status']}", "",
             f"- 测量时长：{duration:g} s", f"- BSS 0关联STA：{len(sta_rows)}",
             f"- 实际承载应用业务STA：{len(active_nodes)}",
             f"- Jain需求满足公平性：{jain:.6f}",
             f"- TCP纯ACK实际AC：`{target_tcp_ack}`", "", "## 自动检查", ""]
    lines += [f"- [{'x' if ok else ' '}] {name}" for name, ok in checks.items()]
    (args.output / "candidate_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
