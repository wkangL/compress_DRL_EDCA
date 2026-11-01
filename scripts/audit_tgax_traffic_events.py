"""Audit application events emitted by the TGax ns-3 traffic generators."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,required=True); a=p.parse_args()
    events=[r for r in read(a.run/"application_events.csv") if r["event"]=="GENERATE"]
    groups=defaultdict(list)
    for r in events: groups[(r["model"],r["direction"])].append(r)
    exact={
        "voip_payload_sizes_exact": all(int(r["bytes"]) in {7,33} for r in events if r["model"]=="VOIP"),
        "http_request_is_real_350B_payload": all(int(r["bytes"])==350 for r in events if r["model"]=="HTTP_REQ") and any(r["model"]=="HTTP_REQ" for r in events),
        "ftp_files_positive_and_at_most_5MiB": all(0<int(r["bytes"])<=5*1024*1024 for r in events if r["model"]=="FTP"),
        "video_frames_positive": all(int(r["bytes"])>0 for r in events if r["model"] in {"BV3","BV6"}),
        "gaming_packets_positive": all(int(r["bytes"])>0 for r in events if r["model"]=="GMG"),
        "all_official_models_observed": {"VOIP","BV3","BV6","FTP","HTTP","HTTP_REQ","GMG"}.issubset({r["model"] for r in events}),
    }
    stats=[]
    gamma=.5772156649015329
    targets={
        ("BV3","DL"):20850*math.gamma(1+1/.8099), ("BV3","UL"):20850*math.gamma(1+1/.8099),
        ("BV6","DL"):54210*math.gamma(1+1/.8099),
        ("GMG","DL"):390+gamma*89, ("GMG","UL"):158+gamma*26.2,
    }
    for key, rows in sorted(groups.items()):
        values=[int(r["bytes"]) for r in rows]; target=targets.get(key)
        stats.append({"model":key[0],"direction":key[1],"samples":len(values),
                      "mean_bytes":fmean(values),"min_bytes":min(values),"max_bytes":max(values),
                      "reference_mean_bytes":target,
                      "relative_mean_error":((fmean(values)-target)/target if target else None)})
    result={"status":"PASS" if all(exact.values()) else "FAIL","exact_checks":exact,
            "sample_statistics":stats,
            "interpretation":"Exact constraints are pass/fail. Distribution means are evidence with finite-sample error, not pass criteria."}
    (a.run/"traffic_model_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[f"# TGax业务生成审计：{result['status']}",""]+[f"- [{'x' if ok else ' '}] {k}" for k,ok in exact.items()]
    lines += ["","## 事件样本统计","","| 模型 | 方向 | n | 均值B | 最小B | 最大B | 理论均值B | 相对误差 |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in stats:
        target="—" if r["reference_mean_bytes"] is None else f'{r["reference_mean_bytes"]:.2f}'
        error="—" if r["relative_mean_error"] is None else f'{100*r["relative_mean_error"]:.2f}%'
        lines.append(f'| {r["model"]} | {r["direction"]} | {r["samples"]} | {r["mean_bytes"]:.2f} | {r["min_bytes"]} | {r["max_bytes"]} | {target} | {error} |')
    lines += ["","> 精确约束用于PASS/FAIL；分布均值只报告有限样本误差，不用宽松阈值把短运行包装成统计收敛。",""]
    (a.run/"traffic_model_audit.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"status":result["status"],"exact_checks":exact},ensure_ascii=False))
    if result["status"]!="PASS": raise SystemExit(1)


if __name__=="__main__": main()
