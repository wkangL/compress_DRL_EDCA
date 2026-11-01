"""Stage 0: upstream protocol tests and measured scenario checks, never AI training."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import csv, json, subprocess, sys, time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project_config import ROOT,load_config,merge,digest,backend_source_hash
from sim.backend.ns3_backend import Ns3Backend,runtime_env,find_binary
from sim.backend.base_backend import ControlAction
from sim.statistics.metrics import summarize


def run_case(job: tuple) -> dict:
    name,c,scenario,seed,action,root=job
    folder=Path(root)/f"{name}_seed{seed}"
    folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    (folder/"config.json").write_text(json.dumps(c,indent=2),encoding="utf-8")
    meta={"seed":seed,"scenario":scenario,"action":asdict(action),"started":datetime.now().astimezone().isoformat(),
          "config_hash":digest(c),"ns3_commit":"43dce6710b8df69685e3479c1d33a571dda714ea"}
    try:
        with Ns3Backend(c,scenario,seed,folder,trace=True) as backend:
            events=backend.step(action,c["validation"]["duration_s"])
        result=summarize(events,c["network"]["num_sta"],c["validation"]["duration_s"])
        frames={}; max_txop=[0.]*4; sequence=[]; exchanges=0
        with (folder/"mac_trace.csv").open() as f:
            for row in csv.DictReader(f):
                event=row["event"]; frames[event]=frames.get(event,0)+1
                if event in ("RTS","CTS","DATA","ACK"):
                    sequence=(sequence+[event])[-4:]
                    if sequence==["RTS","CTS","DATA","ACK"]: exchanges+=1
                if event=="TXOP":
                    ac=int(row["ac"]); max_txop[ac]=max(max_txop[ac],float(row["value"]))
        result.update({f"frames_{k}":v for k,v in frames.items()})
        result.update({f"max_txop_us_{i}":v for i,v in enumerate(max_txop)})
        result["rts_cts_data_ack_sequences"]=exchanges
        result.update(case=name,seed=seed,status="measured")
    except Exception as e:
        result=dict(case=name,seed=seed,status="error",error=repr(e))
    meta["runtime_s"]=time.monotonic()-started
    (folder/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    (folder/"metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def main() -> None:
    c=load_config(ROOT/"configs/quick.yaml")
    stamp=datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    root=ROOT/"results/raw"/f"edca_validation_{stamp}"; root.mkdir(parents=True)
    suites={}
    runner=next((ROOT/"vendor/ns-3.44/build/utils").glob("*test-runner*.exe"))
    for suite in ("wifi-devices-dcf","wifi-devices-edca","wifi-channel-access-manager","wifi-mac-queue","wifi-txop-nonht"):
        try:
            p=subprocess.run([str(runner),f"--suite={suite}","--verbose"],env=runtime_env(),capture_output=True,text=True,timeout=120,cwd=ROOT/"vendor/ns-3.44")
            suites[suite]={"passed":p.returncode==0 and f"PASS {suite}" in p.stdout,"returncode":p.returncode}
            (root/f"{suite}.log").write_text(p.stdout+p.stderr,encoding="utf-8")
        except subprocess.TimeoutExpired:
            suites[suite]={"passed":False,"error":"timeout"}
    def case(name: str, count: int, load: float, mix: list[float], rts: bool=False,hidden: bool=False,action: ControlAction=ControlAction()) -> list[tuple]:
        cfg=merge(c,{"network":{"num_sta":count,"rts_cts":rts,"hidden_nodes":hidden}})
        scenario={"offered_mbps":load,"ac_mix":mix,"ul_fraction":1.}
        return [(name,cfg,scenario,seed,action,str(root)) for seed in c["validation"]["seeds"]]
    jobs=[]
    for v in load_config(ROOT/"configs/validation.yaml")["cases"]:
        jobs+=case(v["name"],v["num_sta"],v["offered_mbps"],v["ac_mix"],v.get("rts_cts",False),v.get("hidden_nodes",False),ControlAction(**v.get("action",{})))
    rows=[]
    with ProcessPoolExecutor(max_workers=c["run"]["workers"]) as pool:
        for result in pool.map(run_case,jobs):
            rows.append(result); print(json.dumps(result),flush=True)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (root/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    def mean(name: str,key: str) -> float:
        values=[r[key] for r in rows if r["case"]==name and r["status"]=="measured"]
        return float(np.mean(values)) if len(values)==len(c["validation"]["seeds"]) else float("nan")
    checks=[]
    def check(name: str,expected: str,passed: bool,actual: str) -> None:
        checks.append(dict(name=name,expected=expected,passed=bool(passed),actual=actual))
    ratio="collision_rate"
    check("单STA低负载","实际重叠接收失败接近0",mean("single_light",ratio)<c["validation"]["single_collision_max"],str(mean("single_light",ratio)))
    a=[mean(f"contention_{i}",ratio) for i in (2,5,10)]
    check("STA数量增加","固定每STA offered load，竞争失败总体增加",a[-1]>a[0],str(a))
    delays=[mean("ac_priority",f"{ac}_delay_mean_ms") for ac in ("vo","vi","be","bk")]
    check("AC优先级","VO/VI平均时延低于BE/BK",max(delays[:2])<min(delays[2:]),str(delays))
    small,large=mean("cw_small",ratio),mean("cw_large",ratio)
    ds,dl=mean("cw_small","be_delay_mean_ms"),mean("cw_large","be_delay_mean_ms")
    check("CWmin增加","失败比例下降且平均等待增加",large<small and dl>ds,f"timeout {small}/{large}; delay {ds}/{dl} ms")
    sequence=mean("rts_visible","rts_cts_data_ack_sequences")
    check("RTS/CTS帧","trace实际连续产生RTS→CTS→DATA→ACK",sequence>0,f"平均完整序列数 {sequence}")
    normal,hidden,recovered=[mean(name,"goodput_mbps") for name in ("hidden_off","hidden_on","hidden_rts")]
    check("隐藏节点恢复","隐藏节点降低吞吐，RTS/CTS部分恢复",hidden<normal and recovered>hidden,f"{normal}/{hidden}/{recovered} Mbps")
    base,custom=mean("contention_10","goodput_mbps"),mean("beb_custom","goodput_mbps")
    check("BEB退化","相同种子和负载，custom beta=2 gamma=7近似标准BEB",abs(custom-base)/max(base,1e-9)<=c["validation"]["beb_relative_tolerance"],f"{base}/{custom} Mbps")
    check("碰撞独立归因","预期接收端真实RxDrop且有重叠可达信号；timeout单列",mean("contention_10","collision_count")>0 and mean("single_light","other_rx_failures")==0,
          f"碰撞 {mean('contention_10','collision_count')}; timeout {mean('contention_10','timeouts')}; 低负载其他失败 {mean('single_light','other_rx_failures')}")
    passed=all(v["passed"] for v in suites.values()) and all(v["passed"] for v in checks)
    fingerprint=backend_source_hash()
    result={"passed":passed,"suites":suites,"checks":checks,"source_hash":fingerprint,"config_hash":digest(c),"raw_directory":str(root)}
    (root/"validation.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    (ROOT/"results/summary").mkdir(parents=True,exist_ok=True)
    (ROOT/"results/summary/edca_validation.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# EDCA仿真正确性检查","",f"运行目录：`{root}`","",f"训练门禁：{'通过' if passed else '未通过，禁止正式训练'}。","", "## 上游协议测试",""]
    for name,v in suites.items(): lines.append(f"- {name}: {'PASS' if v['passed'] else 'FAIL'}，{v}")
    lines += ["","## 场景检查","","| 检查 | 理论预期 | 实测 | 状态 |","|---|---|---|---|"]
    for v in checks: lines.append(f"|{v['name']}|{v['expected']}|{v['actual']}|{'PASS' if v['passed'] else 'FAIL'}|")
    lines += ["","碰撞定义为预期接收端DATA/RTS实际接收失败，且同一接收时间有另一个可达发送信号。分母为DATA/RTS尝试数。ACK/CTS timeout单列。该归因依赖本场景固定矩阵链路，尚不支持任意功率捕获归因。", "", "完整上游 wifi-txop 在 HE 子用例发生越界（见首轮记录），未宣称修复。wifi-txop-nonht 使用原测试实现的三个802.11a用例，未修改断言；仅适用于当前无聚合802.11a场景，不代表HE通过。"]
    (ROOT/"reports/EDCA仿真正确性检查.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("STAGE0", "PASS" if passed else "FAIL",root)
    raise SystemExit(0 if passed else 2)


if __name__=="__main__": main()
