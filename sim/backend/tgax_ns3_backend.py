"""Persistent ns-3 TGax control protocol; one process spans all one-second decisions."""
from __future__ import annotations

import os
import queue
import subprocess
import threading
from pathlib import Path

from project_config import ROOT
from sim.backend.base_backend import ControlAction, Event
from sim.edca.profile_table import ACS, load_profiles


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(ROOT / ".tools/mingw64/bin"),
                                  str(ROOT / "vendor/ns-3.44/build/lib"), env.get("PATH", "")])
    if os.name != "nt":
        library_paths = [str(ROOT / "vendor/ns-3.44/build/lib")]
        if env.get("LD_LIBRARY_PATH"):
            library_paths.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    return env


def find_binary(config: dict) -> Path:
    """Resolve the configured TGax executable on Windows or Linux."""
    requested = ROOT / config["backend"]["binary"]
    if requested.is_file():
        return requested
    build = ROOT / "vendor/ns-3.44/build/scratch"
    candidates = [path for path in build.glob("ns3.44-tgax-control*")
                  if path.is_file() and (os.name == "nt" or path.suffix.lower() != ".exe")]
    if os.name != "nt":
        candidates = [path for path in candidates if os.access(path, os.X_OK)]
    if not candidates:
        raise FileNotFoundError(
            f"TGax backend not found (configured as {requested}). "
            "Build it first with: python3 scripts/build_ns3.py"
        )
    preferred = ("ns3.44-tgax-control", "ns3.44-tgax-control-default",
                 "ns3.44-tgax-control-relwithdebinfo")
    candidates.sort(key=lambda path: (preferred.index(path.name) if path.name in preferred else len(preferred),
                                      path.name))
    return candidates[0]


class TgaxNs3Backend:
    def __init__(self, config: dict, seed: int, directory: Path, duration_s: float):
        self.config=config; self.directory=directory; self.time=0.0; self.duration_s=float(duration_s)
        self.control_warmup=float(config["training"].get("traffic_warmup_s",0.0))
        self.offset=0.0; self.warmup_events=[]
        self.warmup=float(config["timing"]["association_warmup_s"])
        directory.mkdir(parents=True,exist_ok=False)
        profiles=load_profiles(ROOT/config["action"]["profiles"])
        flat=directory/"profiles.txt"
        flat.write_text("\n".join(" ".join(str(p[ac][k]) for ac in ACS
            for k in ("AIFSN","CWmin","CWmax","TXOP_ms")) for p in profiles),encoding="ascii")
        b=find_binary(config)
        topology=ROOT/config["backend"]["topology"]
        flows=ROOT/config["backend"]["flows"]
        self.timeout=config["backend"].get("timeout_s",900)
        self.stderr=(directory/"backend.stderr.log").open("w",encoding="utf-8")
        self.raw=(directory/"events.log").open("w",encoding="utf-8")
        phy=config["backend"].get("phy_backend","yans")
        traffic=config["backend"].get("traffic_mode","applications")
        command=[str(b),f"--topology={topology}",f"--flows={flows}",f"--output={directory}",
                 f"--warmup={config['timing']['association_warmup_s']}",f"--duration={duration_s+self.control_warmup}",
                 f"--drain={config['timing'].get('drain_s',1.0)}",f"--seed={seed}",
                 "--traceLevel=summary",f"--phyBackend={phy}","--targetProfile=default",
                 "--interactive=1",f"--profiles={flat}"]
        if traffic!="applications":
            command += [f"--trafficMode={traffic}",
                        f"--directHighWatermark={config['backend'].get('direct_high_watermark',0)}",
                        f"--directAdmissionPollUs={config['backend'].get('direct_admission_poll_us',500)}"]
        self.process=subprocess.Popen(command,cwd=ROOT,env=runtime_env(),stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding="utf-8",bufsize=1)
        self.lines: queue.Queue[str|None]=queue.Queue()
        def reader():
            assert self.process.stdout is not None
            for line in self.process.stdout: self.lines.put(line)
            self.lines.put(None)
        self.reader=threading.Thread(target=reader,daemon=True); self.reader.start()
        ready=self._read().strip()
        if ready!="READY TGAX1":
            self.close(); raise RuntimeError(f"Unexpected TGax handshake: {ready!r}")
        if self.control_warmup>0:
            raw=self._exchange(ControlAction(0,False,1,2.0,2.0,7),self.control_warmup)
            self.offset=self.control_warmup; self.time=0.0
            self.warmup_events=self._normalize(raw)

    def _read(self) -> str:
        try: line=self.lines.get(timeout=self.timeout)
        except queue.Empty as error: raise TimeoutError("TGax backend timeout") from error
        if line is None:
            self.stderr.flush()
            error=(self.directory/"backend.stderr.log").read_text(encoding="utf-8")
            raise RuntimeError(f"TGax backend exited: {error[-5000:]}")
        self.raw.write(line)
        return line

    def _exchange(self,action:ControlAction,duration_s:float)->list[Event]:
        if duration_s<=0 or action.beta2<action.beta1: raise ValueError("Invalid TGax action")
        assert self.process.stdin is not None
        if action.sta_profile is None:
            request=(f"STEP {duration_s} {action.profile} {int(action.custom_growth)} "
                     f"{action.alpha} {round(action.beta1*10)} {round(action.beta2*10)} {action.gamma}\n")
        else:
            sta_custom=(action.custom_growth if action.sta_custom_growth is None
                        else action.sta_custom_growth)
            request=(f"STEP2 {duration_s} {action.profile} {int(action.custom_growth)} "
                     f"{action.sta_profile} {int(sta_custom)} {action.alpha} "
                     f"{round(action.beta1*10)} {round(action.beta2*10)} {action.gamma}\n")
        self.process.stdin.write(request)
        self.process.stdin.flush(); result=[]
        while True:
            fields=self._read().strip().split()
            if fields[0]=="END":
                self.time=float(fields[1])-self.offset
                self.raw.flush(); return result
            result.append(Event(fields[0],tuple(map(float,fields[1:]))))

    def _normalize(self,events:list[Event])->list[Event]:
        result=[]
        for event in events:
            values=list(event.values)
            if event.kind in {"S","G","R","X","Y","B"}: values[0]-=self.offset
            if event.kind=="R": values[2]-=self.offset
            result.append(Event(event.kind,tuple(values)))
        return result

    def step(self, action: ControlAction, duration_s: float) -> list[Event]:
        return self._normalize(self._exchange(action,duration_s))

    def close(self) -> None:
        if hasattr(self,"process") and self.process.poll() is None:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write("CLOSE\n"); self.process.stdin.flush()
                self.process.wait(timeout=self.timeout)
            except (OSError,subprocess.TimeoutExpired):
                self.process.kill(); self.process.wait()
        if hasattr(self,"reader"): self.reader.join(timeout=2)
        for stream in (getattr(self.process,"stdin",None),getattr(self.process,"stdout",None),
                       getattr(self,"stderr",None),getattr(self,"raw",None)):
            if stream is not None:
                try: stream.close()
                except OSError: pass
