"""Create a self-contained interactive SVG browser from actual TGax CSV inputs."""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def read(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    nodes=read(a.input/"topology.csv"); active=read(a.input/"active_bss.csv")
    xs=[float(r["x"]) for r in nodes]; ys=[float(r["y"]) for r in nodes]
    xmin,xmax=min(xs)-12,max(xs)+12; ymin,ymax=min(ys)-12,max(ys)+12; width=1000; height=760
    sx=lambda x:(x-xmin)/(xmax-xmin)*width; sy=lambda y:height-(y-ymin)/(ymax-ymin)*height
    aps={int(r["bss"]):r for r in nodes if r["role"]=="AP"}
    parts=[]
    for r in active:
        b=int(r["bss"]); x=sx(float(r["x"])); y=sy(float(r["y"])); color="#d62728" if b==0 else "#1f77b4"
        parts.append(f'<polygon class="cell b{b}" points="{x},{y-70} {x+61},{y-35} {x+61},{y+35} {x},{y+70} {x-61},{y+35} {x-61},{y-35}" fill="{color}"/>')
    for r in nodes:
        if r["role"]!="STA": continue
        b=int(r["bss"]); ap=aps[b]
        parts.append(f'<line class="assoc b{b}" x1="{sx(float(ap["x"])):.2f}" y1="{sy(float(ap["y"])):.2f}" x2="{sx(float(r["x"])):.2f}" y2="{sy(float(r["y"])):.2f}"/>')
    for r in nodes:
        b=int(r["bss"]); x=sx(float(r["x"])); y=sy(float(r["y"])); label=html.escape(r["node_label"])
        tip=(f"{label} | BSS {b} | ({float(r['x']):.2f}, {float(r['y']):.2f}, {float(r['z']):.1f}) m"
             + (f" | strongest margin {float(r['strongest_margin_db']):.2f} dB" if r["role"]=="STA" else " | AP"))
        if r["role"]=="AP": parts.append(f'<path class="node ap b{b}" data-tip="{tip}" d="M{x},{y-9} L{x+9},{y+8} L{x-9},{y+8} Z"/>')
        else: parts.append(f'<circle class="node sta b{b}" data-tip="{tip}" cx="{x:.2f}" cy="{y:.2f}" r="3"/>')
    options=''.join(f'<option value="{b}">BSS {b}</option>' for b in sorted(aps))
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TGax参数驱动的简化3-BSS拓扑</title>
<style>body{{font:14px system-ui;margin:18px;background:#fafafa;color:#222}} .toolbar{{display:flex;gap:14px;align-items:center;margin-bottom:10px}} svg{{background:white;border:1px solid #ccc;max-width:100%;height:auto}} .cell{{opacity:.07;stroke-width:2}} .assoc{{stroke:#777;stroke-width:.45;opacity:.2}} .node{{cursor:pointer}} .ap{{fill:#d62728;stroke:white;stroke-width:2}} .sta{{fill:#333;opacity:.75}} #tip{{min-height:22px;font-weight:600}} .hidden{{display:none}}</style>
<h1>参考TGax参数的简化3-BSS场景：实际ns-3坐标</h1><div class="toolbar"><label>显示 <select id="filter"><option value="all">全部BSS</option>{options}</select></label><label><input id="links" type="checkbox" checked> 关联线</label><span>红色三角：目标AP；悬停节点查看原始数值</span></div><div id="tip">请选择或悬停节点</div>
<svg viewBox="0 0 {width} {height}" role="img">{''.join(parts)}</svg>
<script>const f=document.querySelector('#filter'),tip=document.querySelector('#tip'),links=document.querySelector('#links'); function apply(){{const v=f.value; document.querySelectorAll('.cell,.assoc,.node').forEach(e=>e.classList.toggle('hidden',v!=='all'&&!e.classList.contains('b'+v))); document.querySelectorAll('.assoc').forEach(e=>e.style.visibility=links.checked?'visible':'hidden')}} f.onchange=apply;links.onchange=apply;document.querySelectorAll('.node').forEach(e=>{{e.onmouseenter=()=>tip.textContent=e.dataset.tip;e.onclick=()=>tip.textContent=e.dataset.tip}});</script></html>'''
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(doc,encoding="utf-8"); print(a.output.resolve())


if __name__=="__main__": main()
