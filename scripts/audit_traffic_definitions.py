"""Expand scenario traffic into per-STA/per-direction rates and a transparent airtime lower bound."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_config import ROOT, load_config
from sim.scenarios.resolver import resolve_scenario
from sim.traffic.payload import mean_payload


ACS = ("VO", "VI", "BE", "BK")


def ofdm_time_us(bytes_: float, ndbps: int) -> float:
    return 20.0 + 4.0 * math.ceil((16.0 + 8.0 * bytes_ + 6.0) / ndbps)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT/"configs/default.yaml")
    parser.add_argument("--output", type=Path, default=ROOT/"results/summary/traffic_definition_audit_20261020")
    args = parser.parse_args(); config = load_config(args.config); args.output.mkdir(parents=True, exist_ok=True)
    scenarios = load_config(ROOT/config["traffic"]["scenarios"])
    capacity = json.loads((ROOT/config["traffic"]["capacity_file"]).read_text(encoding="utf-8"))["C_ref_mbps"]
    profiles = json.loads((ROOT/config["action"]["profiles"]).read_text(encoding="utf-8"))
    profile = profiles[config["action"]["default_profile"]]
    rows=[]; summary=[]
    for scenario_name, definition in scenarios.items():
        resolved=resolve_scenario(definition,config["network"]["num_sta"],capacity)
        rates=resolved.get("per_sta_mbps_by_ac",[resolved["offered_mbps"]*x/config["network"]["num_sta"] for x in resolved["ac_mix"]])
        lower=0.0
        for index,(ac,per_sta_mbps,payload_model) in enumerate(zip(ACS,rates,config["traffic"]["payload"])):
            payload=mean_payload(payload_model); pps=per_sta_mbps*1e6/(8*payload); ul=resolved["ul_fraction"]
            # Payload + LLC/SNAP + QoS MAC header + FCS. This is an analytical lower bound,
            # not an ns-3 replacement and excludes collisions, retries and queueing.
            data_us=ofdm_time_us(payload+8+26+4,216); ack_us=ofdm_time_us(14,24)
            p=profile[ac]; access_us=16+p["AIFSN"]*9+p["CWmin"]*9/2
            exchange_us=data_us+16+ack_us+access_us
            demand=pps*config["network"]["num_sta"]*exchange_us/1e6
            lower += demand
            rows.append(dict(scenario=scenario_name,ac=ac,mean_payload_bytes=payload,
                per_sta_total_mbps=per_sta_mbps,per_sta_total_pps=pps,
                per_sta_ul_mbps=per_sta_mbps*ul,per_sta_ul_pps=pps*ul,
                per_sta_dl_mbps=per_sta_mbps*(1-ul),per_sta_dl_pps=pps*(1-ul),
                network_mbps=per_sta_mbps*config["network"]["num_sta"],
                network_pps=pps*config["network"]["num_sta"],
                ideal_exchange_us=exchange_us,ideal_airtime_fraction=demand))
        summary.append(dict(scenario=scenario_name,load_model=resolved["load_model"],arrival=resolved.get("arrival"),
            ul_fraction=resolved["ul_fraction"],per_sta_total_mbps=sum(rates),
            network_total_mbps=resolved["offered_mbps"],payload_only_fraction_of_54mbps=resolved["offered_mbps"]/54,
            ideal_serial_airtime_lower_bound=lower,
            note="lower bound sums one unaggregated DATA/SIFS/ACK/AIFS/mean-initial-backoff exchange per generated packet; excludes collisions/retries/queueing"))
    write_csv(args.output/"per_sta_ac_direction.csv",rows); write_csv(args.output/"scenario_summary.csv",summary)
    (args.output/"provenance.json").write_text(json.dumps({"config":str(args.config.resolve()),
        "scenario_file":config["traffic"]["scenarios"],"capacity_mbps":capacity,
        "phy_assumption":"802.11a 54 Mbps data, 6 Mbps ACK, no aggregation",
        "warning":"airtime is an analytical lower bound only; measured ns-3 load scan is authoritative"},indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
