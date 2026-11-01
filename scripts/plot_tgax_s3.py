"""Generate report-ready figures directly from TGax inputs and ns-3 traces."""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.patches import RegularPolygon
import numpy as np

from build_tgax_s3_inputs import link_shadow_db, pathloss_db

COLORS = {"VO": "#d62728", "VI": "#ff7f0e", "BE": "#1f77b4", "BK": "#9467bd"}
CHANNEL_COLORS = {42: "#1f77b4", 58: "#ff7f0e", 106: "#2ca02c"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, stem: str, png_dir: Path, pdf_dir: Path) -> None:
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def hexagon(ax, x, y, color, alpha=.08, lw=.7) -> None:
    ax.add_patch(RegularPolygon((x, y), 6, radius=10, orientation=math.pi / 2,
                                facecolor=color, edgecolor=color, alpha=alpha, lw=lw))


def topology_plot(topology, active, refs, png_dir, pdf_dir):
    fig, ax = plt.subplots(figsize=(10.2, 8.0))
    active_ids = {int(r["bss"]) for r in active}
    for r in refs:
        hexagon(ax, float(r["x"]), float(r["y"]), "#9aa0a6", .025, .35)
    for r in active:
        bss = int(r["bss"])
        hexagon(ax, float(r["x"]), float(r["y"]), "#d62728" if bss == 0 else "#1f77b4", .09, .8)
    by_bss = defaultdict(list)
    aps = {}
    for r in topology:
        if r["role"] == "AP": aps[int(r["bss"])] = r
        else: by_bss[int(r["bss"])].append(r)
    for bss in active_ids:
        ap = aps[bss]; axx, ayy = float(ap["x"]), float(ap["y"])
        stas = by_bss[bss]
        lines = [[(axx, ayy), (float(s["x"]), float(s["y"]))] for s in stas]
        ax.add_collection(LineCollection(lines, colors="#aeb6bf", linewidths=.25, alpha=.28, zorder=1))
        ax.scatter([float(s["x"]) for s in stas], [float(s["y"]) for s in stas],
                   s=7, color="#555", alpha=.62, zorder=2)
        ax.scatter([axx], [ayy], marker="^", s=115 if bss == 0 else 65,
                   color="#d62728" if bss == 0 else "#1f77b4", edgecolor="white", zorder=4)
        ax.text(axx, ayy + 2.0, f"BSS {bss}", ha="center", fontsize=7,
                weight="bold" if bss == 0 else "normal")
    xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
    sx, sy = xmin + 3, ymin + 4
    ax.plot([sx, sx + 20], [sy, sy], color="black", lw=2)
    ax.text(sx + 10, sy + 2, "20 m", ha="center", fontsize=8)
    ax.set(title=f"TGax-parameter-based simplified multi-BSS scene: actual ns-3 coordinates\n({len(active_ids)} APs, {sum(len(v) for v in by_bss.values())} STAs)",
           xlabel="x / m", ylabel="y / m", aspect="equal")
    ax.grid(alpha=.15)
    fig.text(.5, .015, "Triangles: APs; dots: STAs; red: target BSS; thin lines: configured association", ha="center", fontsize=9)
    save(fig, "01_topology_actual", png_dir, pdf_dir)


def channel_plot(active, refs, png_dir, pdf_dir):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    channels = [42, 58, 106]
    for r in refs:
        ch = channels[(int(r["base_q"]) - int(r["base_r"])) % 3]
        hexagon(ax0, float(r["x"]), float(r["y"]), CHANNEL_COLORS[ch], .24, .55)
    for ch, color in CHANNEL_COLORS.items(): ax0.scatter([], [], marker="h", s=90, color=color, alpha=.45, label=f"80 MHz ch {ch}")
    ax0.set(title="Official reuse-3 reference plan (61 cells)", xlabel="x / m", ylabel="y / m", aspect="equal")
    ax0.legend(frameon=False, fontsize=8); ax0.grid(alpha=.12)
    ys = list(range(len(active)))
    for y, r in zip(ys, active):
        ax1.barh(y, 80, left=5170, height=.65, color=CHANNEL_COLORS[42], alpha=.75)
    ax1.set(yticks=ys, yticklabels=[f"BSS {r['bss']}" for r in active], xlabel="frequency / MHz",
            title=f"Actual {len(active)} active co-channel BSSs: ch 42, 80 MHz\n(primary 20 MHz: ch 36)")
    ax1.axvline(5180, color="black", ls="--", lw=1, label="primary 20 MHz center")
    ax1.legend(frameon=False, fontsize=8); ax1.grid(axis="x", alpha=.2)
    save(fig, "02_channel_assignment", png_dir, pdf_dir)


def interference_plot(active, seed, png_dir, pdf_dir):
    aps = [(int(r["bss"]), float(r["x"]), float(r["y"]), float(r["z"])) for r in active]
    n = len(aps); matrix = np.full((n, n), np.nan)
    edges = []
    for i, (_, x1, y1, z1) in enumerate(aps):
        for j, (_, x2, y2, z2) in enumerate(aps):
            if i == j: continue
            d = math.dist((x1, y1, z1), (x2, y2, z2))
            matrix[i, j] = 23.0102999566 - pathloss_db(d) - link_shadow_db((x1,y1,z1),(x2,y2,z2),seed)
            if i < j and max(matrix[i,j], matrix[j,i]) >= -82: edges.append((i,j))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    im = ax0.imshow(matrix, cmap="viridis", vmin=-100, vmax=-45)
    ax0.set(xticks=range(n), yticks=range(n), xticklabels=[a[0] for a in aps], yticklabels=[a[0] for a in aps],
            xlabel="receiver BSS", ylabel="transmitter BSS", title="AP-to-AP received power / dBm")
    fig.colorbar(im, ax=ax0, shrink=.8)
    for i,j in edges: ax1.plot([aps[i][1],aps[j][1]],[aps[i][2],aps[j][2]], color="#d62728", alpha=.5, lw=1)
    ax1.scatter([a[1] for a in aps],[a[2] for a in aps], c=["#d62728" if a[0]==0 else "#1f77b4" for a in aps], s=75, marker="^")
    for b,x,y,_ in aps: ax1.text(x,y+1.5,str(b),ha="center",fontsize=8)
    ax1.set(title=f"CCA-sense graph at −82 dBm ({len(edges)} undirected links)", xlabel="x / m", ylabel="y / m", aspect="equal")
    ax1.grid(alpha=.15)
    save(fig, "03_propagation_interference", png_dir, pdf_dir)


def traffic_plot(flows, png_dir, pdf_dir):
    bsses = sorted({int(r["bss"]) for r in flows}); services = ["BV6","BV3","FTP","HTTP","GMG","VOIP"]
    colors = {s: c for s,c in zip(services, ["#6a3d9a","#cab2d6","#1f78b4","#a6cee3","#33a02c","#e31a1c"])}
    counts = defaultdict(Counter)
    for r in flows: counts[int(r["bss"])][r["model"]] += 1
    fig, (ax0,ax1) = plt.subplots(1,2,figsize=(13,5.3))
    bottom=np.zeros(len(bsses))
    for s in services:
        vals=np.array([counts[b][s] for b in bsses]); ax0.bar(bsses,vals,bottom=bottom,label=s,color=colors[s]); bottom+=vals
    ax0.set(title="Configured one-way application components per BSS", xlabel="BSS", ylabel="flow components", xticks=bsses); ax0.legend(ncol=2,frameon=False)
    matrix=np.zeros((4,3),dtype=int); acs=["VO","VI","BE","BK"]; dirs=["DL","UL","Total"]
    for r in flows:
        matrix[acs.index(r["ac"]), 0 if r["direction"]=="DL" else 1]+=1
    matrix[:,2]=matrix[:,0]+matrix[:,1]
    im=ax1.imshow(matrix,cmap="Blues")
    ax1.set(xticks=range(3),xticklabels=dirs,yticks=range(4),yticklabels=acs,title="Traffic direction and Wi-Fi access category")
    for i in range(4):
        for j in range(3): ax1.text(j,i,str(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]>matrix.max()/2 else "black")
    fig.colorbar(im,ax=ax1,shrink=.8)
    save(fig,"04_traffic_configuration",png_dir,pdf_dir)


def timelines(run, png_dir, pdf_dir):
    app=read(run/"application_events.csv"); mac=read(run/"mac_timeline.csv")
    fig, axes=plt.subplots(3,1,figsize=(13,8),sharex=False)
    selected=[]
    for model in ("VOIP","BV6","FTP"):
        ids=[r["flow_id"] for r in app if r["model"]==model]
        if ids: selected.append((model,ids[0]))
    for row,(model,fid) in enumerate(selected[:3]):
        subset=[r for r in app if r["flow_id"]==fid]
        axes[row].vlines([float(r["time_s"]) for r in subset],0,[max(1,int(r["bytes"])) for r in subset],color=COLORS.get(subset[0]["ac"],"#555"),lw=.8)
        axes[row].set(ylabel="bytes",title=f"{model} application generation timeline (flow {fid})")
        axes[row].grid(alpha=.15)
    axes[-1].set_xlabel("simulation time / s")
    fig.tight_layout(); save(fig,"05_application_timeline",png_dir,pdf_dir)

    data=[r for r in mac if r["frame_type"] in {"QOS_DATA","MAC_ACK","BLOCK_ACK","BLOCK_ACK_REQ"}]
    if data:
        # Association-time ACKs occur before application traffic.  Anchor the
        # view on the first QoS DATA so the report shows DATA/A-MPDU and its
        # actual ACK/Block-ACK exchange rather than only management ACKs.
        t0=min(float(r["time_s"]) for r in data if r["frame_type"] == "QOS_DATA")
        data=[r for r in data if t0 <= float(r["time_s"]) <= t0+.015]
    nodes=[]
    for r in data:
        if r["node"] not in nodes: nodes.append(r["node"])
    nodes=nodes[:12]; pos={n:i for i,n in enumerate(nodes)}
    fig,ax=plt.subplots(figsize=(13,5.8))
    frame_colors={"QOS_DATA":"#1f77b4","MAC_ACK":"#888","BLOCK_ACK":"#2ca02c","BLOCK_ACK_REQ":"#ff7f0e"}
    for r in data:
        if r["node"] not in pos: continue
        x=float(r["time_s"])*1000; w=max(float(r["duration_s"])*1000,.006)
        ax.broken_barh([(x,w)],(pos[r["node"]]-.35,.7),facecolors=frame_colors[r["frame_type"]])
        if int(r["mpdus"])>1: ax.text(x+w/2,pos[r["node"]],r["mpdus"],fontsize=5,ha="center",va="center",color="white")
    handles=[Patch(facecolor=c,label=typ) for typ,c in frame_colors.items()]
    ax.set(yticks=range(len(nodes)),yticklabels=nodes,xlabel="simulation time / ms",title="Actual MAC exchange timeline (numbers are MPDUs in A-MPDU)")
    ax.legend(handles=handles,ncol=4,frameon=False); ax.grid(axis="x",alpha=.2)
    save(fig,"06_mac_exchange_timeline",png_dir,pdf_dir)


def metrics_plot(run,png_dir,pdf_dir):
    bss=read(run/"bss_summary.csv"); ids=[int(r["bss"]) for r in bss]
    fig,axes=plt.subplots(2,2,figsize=(11,7.6))
    specs=[("throughput_mbps","Throughput / Mbit/s"),("delivery_fraction","Application byte delivery fraction"),("busy_fraction","AP PHY non-idle fraction"),("mean_queue_packets","Mean MAC queue / packets")]
    for ax,(key,title) in zip(axes.flat,specs):
        vals=[float(r[key]) for r in bss]; ax.bar(ids,vals,color=["#d62728" if i==0 else "#1f77b4" for i in ids]); ax.set(title=title,xlabel="BSS",xticks=ids); ax.grid(axis="y",alpha=.2)
        if key == "mean_queue_packets":
            ax.scatter(ids, [float(r["max_queue_packets"]) for r in bss], marker="D",
                       s=24, color="#ff7f0e", label="maximum sampled queue")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Default EDCA per-BSS results from the selected ns-3 run\n(check run scope before performance interpretation)")
    fig.tight_layout(); save(fig,"07_default_edca_bss_metrics",png_dir,pdf_dir)


def delay_ccdf_plot(run, png_dir, pdf_dir):
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in read(run / "mac_access_delay.csv"):
        if row["ac"] in COLORS:
            grouped[row["ac"]].append(float(row["access_delay_ms"]))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for ac in ("VO", "VI", "BE", "BK"):
        samples = sorted(grouped.get(ac, []))
        if not samples:
            continue
        xs = np.asarray(samples); ys = (len(xs) - np.arange(len(xs))) / len(xs)
        ax.step(xs, ys, where="post", label=f"{ac} (n={len(xs)})", color=COLORS[ac])
    ax.set(xlabel="MAC access delay / ms", ylabel="CCDF  P(delay ≥ x)",
           title="Default EDCA MAC access-delay CCDF")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1.05); ax.grid(True, which="both", alpha=.2); ax.legend(frameon=False)
    fig.text(.5,.01,"IEEE TGax definition: time from MAC arrival to first PHY transmission start; retransmissions are not new samples.",ha="center",fontsize=8)
    save(fig,"08_delay_ccdf",png_dir,pdf_dir)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--run",type=Path,required=True); p.add_argument("--png-dir",type=Path,required=True); p.add_argument("--pdf-dir",type=Path,required=True); p.add_argument("--seed",type=int,default=20261021); a=p.parse_args()
    topology=read(a.input/"topology.csv"); active=read(a.input/"active_bss.csv"); refs=read(a.input/"reference_61_cells.csv"); flows=read(a.input/"flows.csv")
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Microsoft YaHei","DejaVu Sans"],"font.size":9,"axes.titlesize":11,"axes.labelsize":9,"legend.fontsize":8,"figure.dpi":130,"pdf.fonttype":42})
    topology_plot(topology,active,refs,a.png_dir,a.pdf_dir); channel_plot(active,refs,a.png_dir,a.pdf_dir); interference_plot(active,a.seed,a.png_dir,a.pdf_dir); traffic_plot(flows,a.png_dir,a.pdf_dir); timelines(a.run,a.png_dir,a.pdf_dir); metrics_plot(a.run,a.png_dir,a.pdf_dir); delay_ccdf_plot(a.run,a.png_dir,a.pdf_dir)


if __name__=="__main__": main()
