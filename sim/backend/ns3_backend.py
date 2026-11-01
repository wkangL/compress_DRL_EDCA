"""Persistent line protocol: one crossing per control interval, no per-MAC inference."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
from pathlib import Path

from project_config import ROOT
from sim.backend.base_backend import BaseBackend, ControlAction, Event
from sim.edca.profile_table import ACS, load_profiles
from sim.traffic.payload import mean_payload


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(ROOT / ".tools/mingw64/bin"),
                                 str(ROOT / "vendor/ns-3.44/build/lib"), env.get("PATH", "")])
    return env


def find_binary(config: dict) -> Path:
    requested = ROOT / config["backend"]["binary"]
    if requested.is_file():
        return requested
    matches = list((ROOT / config["backend"]["source"] / "build/scratch").glob("ns3.44-wifi-control-*.exe"))
    if len(matches) != 1:
        raise FileNotFoundError("Build ns-3 first: python scripts/build_ns3.py")
    return matches[0]


class Ns3Backend(BaseBackend):
    def __init__(self, config: dict, scenario: dict, seed: int, directory: Path, trace: bool = False):
        self.config = config
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.timeout = config["backend"]["timeout_s"]
        self.time = 0.0
        self.events: queue.Queue[str | None] = queue.Queue()
        profiles = load_profiles(ROOT / config["action"]["profiles"])
        profile_path = directory / "profiles.txt"
        profile_path.write_text("\n".join(" ".join(str(p[ac][k]) for ac in ACS
                                   for k in ("AIFSN", "CWmin", "CWmax", "TXOP_ms")) for p in profiles), encoding="ascii")
        net, t, traffic = config["network"], config["timing"], config["traffic"]
        if net["phy_error_probability"] != 0:
            raise NotImplementedError("Random PHY error extension is not validated yet")
        if not net["ack"] or net["ampdu"] or net["block_ack"] or net["standard"] != "802.11a" or net["channel_width_mhz"] != 20:
            raise ValueError("This validated backend supports QoS 802.11a 20MHz DATA/ACK only")
        if not net["legacy_applies_profile"]:
            raise NotImplementedError("Legacy fixed-profile alternative requires a separate backend scenario")
        params = dict(num_sta=net["num_sta"], seed=seed, profiles=profile_path.resolve().as_posix(),
                      trace=(directory / "mac_trace.csv").resolve().as_posix() if trace else "none",
                      enhanced_fraction=net["enhanced_fraction"], warmup_s=t["association_warmup_s"],
                      sample_s=t["sta_sample_s"], report_min_s=t["report_min_s"], hidden=int(net["hidden_nodes"]),
                      rts=int(net["rts_cts"]), tx_power_dbm=net["tx_power_dbm"], visible_loss_db=net["visible_loss_db"],
                      hidden_loss_db=net["hidden_loss_db"], data_mode=net["data_mode"], control_mode=net["control_mode"],
                      retry_limit=net["retry_limit"], queue_delay_s=config["queues"]["max_delay_s"],
                      offered_mbps=scenario["offered_mbps"], ul_fraction=scenario["ul_fraction"],
                      video_variable=int(traffic["payload"][1]["type"] != "fixed"),
                      video_sigma=traffic["payload"][1].get("sigma", .35),
                      video_min=traffic["payload"][1].get("min_bytes", 1280),
                      video_max=traffic["payload"][1].get("max_bytes", 1280))
        deadlines=config["reward"]["qos_threshold_ms"]
        params.update({f"deadline{ac}_ms":deadlines[ac] if ac<len(deadlines) else 0 for ac in range(4)})
        pattern=scenario.get("load_pattern",{"type":"stationary"})
        dynamic=pattern.get("type","stationary")=="two_state"
        params.update(dynamic=int(dynamic),dynamic_period_s=pattern.get("period_s",1.0),
                      dynamic_low_scale=pattern.get("low_scale",1.0),
                      dynamic_high_scale=pattern.get("high_scale",1.0),
                      dynamic_stagger=int(bool(pattern.get("stagger_by_sta",False))))
        for ac in range(4):
            p = traffic["payload"][ac]
            params.update({f"payload{ac}":p.get("bytes",p.get("mean_bytes")),
                           f"mean{ac}":mean_payload(p),f"queue{ac}":config["queues"]["limits_packets"][ac],
                           f"mix{ac}":scenario["ac_mix"][ac]})
        config_path = directory / "backend.txt"
        config_path.write_text("\n".join(f"{k}={v}" for k,v in params.items())+"\n", encoding="utf-8")
        self.error_file = (directory / "backend.stderr.log").open("w",encoding="utf-8")
        self.raw_file = (directory / "events.log").open("w",encoding="utf-8")
        self.process = subprocess.Popen([str(find_binary(config)), "--config="+str(config_path.resolve())],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.error_file,
                                        text=True, encoding="utf-8", env=runtime_env(), bufsize=1)
        def reader() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.events.put(line)
            self.events.put(None)
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        try:
            ready = self._read()
            if ready.strip() not in ("READY 1", "READY 2", "READY 3", "READY 4"):
                raise RuntimeError(f"Unexpected handshake {ready!r}")
            self.accounting_version=int(ready.split()[1])
        except BaseException:
            self.close()
            raise

    def _read(self) -> str:
        try:
            line = self.events.get(timeout=self.timeout)
        except queue.Empty as e:
            raise TimeoutError("ns-3 process timed out") from e
        if line is None:
            self.error_file.flush()
            error=(self.directory / "backend.stderr.log").read_text(encoding="utf-8")
            raise RuntimeError(f"ns-3 exited unexpectedly: {error[-4000:]}")
        self.raw_file.write(line)
        return line

    def step(self, action: ControlAction, duration_s: float) -> list[Event]:
        if duration_s<=0 or action.beta2<action.beta1:
            raise ValueError("Invalid step or action")
        command=f"STEP {duration_s} {action.profile} {int(action.custom_growth)} {action.alpha} {round(action.beta1*10)} {round(action.beta2*10)} {action.gamma}\n"
        assert self.process.stdin is not None
        self.process.stdin.write(command); self.process.stdin.flush()
        result=[]
        while True:
            line=self._read().strip().split()
            if line[0]=="END":
                if self.accounting_version>=2:
                    records=[e for e in result if e.kind=="P"]
                    expected={(node,ac) for node in range(self.config["network"]["num_sta"]+1) for ac in range(4)}
                    if len(records)!=len(expected) or {(int(e.values[1]),int(e.values[2])) for e in records}!=expected:
                        raise RuntimeError("Incomplete STEP packet accounting")
                self.time=float(line[1]); self.raw_file.flush()
                return result
            result.append(Event(line[0],tuple(map(float,line[1:]))))

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write("CLOSE\n"); self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill(); self.process.wait()
        self.reader.join(timeout=2)
        for f in (self.process.stdin,self.process.stdout,self.error_file,self.raw_file):
            if f is not None: f.close()
