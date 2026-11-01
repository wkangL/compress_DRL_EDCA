"""Validate TGax Scenario 3 ns-3 outputs without third-party packages."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    required = ["associations.csv", "application_events.csv", "backoff_events.csv",
                "bss_summary.csv", "edca_runtime.csv", "flow_summary.csv",
                "flowmon.xml", "ip_flow_summary.csv", "mac_timeline.csv",
                "mac_access_delay.csv", "mac_packet_events.csv",
                "phy_states.csv", "queue_samples.csv"]
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise SystemExit(f"missing outputs: {missing}")

    topology = rows(args.topology)
    expected_stas = [r for r in topology if r["role"] == "STA"]
    flows = rows(args.flows)
    expected_runtime_flows = len(flows) + sum(r["model"] == "HTTP" for r in flows)
    associations = rows(run / "associations.csv")
    mac = rows(run / "mac_timeline.csv")
    flow_summary = rows(run / "flow_summary.csv")
    ip_flows = rows(run / "ip_flow_summary.csv")
    bss = rows(run / "bss_summary.csv")
    edca = rows(run / "edca_runtime.csv")
    app = rows(run / "application_events.csv")
    access_delay = rows(run / "mac_access_delay.csv")
    packet_events = rows(run / "mac_packet_events.csv")
    frame_counts = Counter(r["frame_type"] for r in mac)
    ac_counts = Counter(r["ac"] for r in mac if r["frame_type"] == "QOS_DATA")
    configured_acs = {r["ac"] for r in flows}
    tcp_ack_ac = Counter(r["ac"] for r in mac if r["tcp_ack_only"] == "1")
    checks = {
        "all_expected_stas_reported": len(associations) == len(expected_stas),
        "all_stas_associated": bool(associations) and all(r["associated"] == "1" for r in associations),
        "all_bssid_matches": bool(associations) and all(r["match"] == "1" for r in associations),
        "all_application_flows_reported": len(flow_summary) == expected_runtime_flows,
        "ip_forward_flows_observed": len(ip_flows) > 0,
        "real_qos_data_observed": frame_counts["QOS_DATA"] > 0,
        "mac_ack_observed": frame_counts["MAC_ACK"] > 0,
        "block_ack_observed": frame_counts["BLOCK_ACK"] > 0,
        "ampdu_observed": any(int(r["mpdus"]) > 1 and r["frame_type"] == "QOS_DATA" for r in mac),
        "pure_tcp_ack_observed": bool(tcp_ack_ac),
        "rts_cts_disabled_in_trace": frame_counts["RTS"] == 0 and frame_counts["CTS"] == 0,
        "edca_runtime_exported": bool(edca),
        "all_configured_access_categories_used": all(ac_counts[name] > 0 for name in configured_acs),
        "delivery_fraction_bounded": all(0 <= float(r["delivery_fraction"]) <= 1.000001 for r in flow_summary),
        "busy_fraction_bounded": all(0 <= float(r["busy_fraction"]) <= 1.000001 for r in bss),
        "application_events_observed": bool(app),
        "tgax_mac_access_delay_observed": bool(access_delay),
        "mac_arrival_and_ack_events_observed": (
            any(r["event"] == "ARRIVAL" for r in packet_events)
            and any(r["event"] == "ACKED" for r in packet_events)),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "nodes": len(topology), "stas": len(expected_stas), "configured_flows": len(flows),
            "runtime_flows_including_http_requests": expected_runtime_flows,
            "mac_psdu": len(mac), "application_events": len(app),
            "mac_access_delay_samples": len(access_delay),
            "frame_types": frame_counts, "qos_data_by_ac": ac_counts,
            "tcp_ack_psdu_by_ac": tcp_ack_ac,
        },
        "bss_summary": bss,
        "edca_runtime": edca,
        "scope_note": "This validates one supplied run only; smoke inputs are not performance evidence.",
    }
    (run / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report = args.report or (run / "validation.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# TGax ns-3运行验证：{result['status']}", "",
             f"- 节点：{len(topology)}（STA {len(expected_stas)}）",
             f"- 输入业务流：{len(flows)}",
             f"- PHY发射PSDU：{len(mac)}",
             f"- 帧类型：`{dict(frame_counts)}`",
             f"- QoS DATA按AC：`{dict(ac_counts)}`",
             f"- 含纯TCP ACK的PSDU按AC：`{dict(tcp_ack_ac)}`", "",
             "## 自动检查", ""]
    lines.extend(f"- [{'x' if ok else ' '}] {name}" for name, ok in checks.items())
    lines += ["", "> 本报告验证的是所给运行的协议与统计链路；若输入为smoke流量，不能据此宣称完整TGax业务性能。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
