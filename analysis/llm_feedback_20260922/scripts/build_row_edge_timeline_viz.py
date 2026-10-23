#!/usr/bin/env python3
"""Build the compact interactive row/edge trajectory explainer from replay logs."""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


def edge_columns(description: str) -> list[str]:
    tokens = re.findall(r"\b[A-Z][A-Z0-9_]*\b", description)
    return list(dict.fromkeys(x for x in tokens if x not in {"IF", "AND", "THEN"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--guided-trace", type=Path, required=True)
    parser.add_argument("--deployed-graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.timeline.read_text())
    guided = json.loads(args.guided_trace.read_text())
    deployed = {edge["candidate_id"] for edge in json.loads(args.deployed_graph.read_text())}
    keep_labels = {"S1", "S2", "S3", "S4", "S5", "S7"}
    graph = [{**edge, "columns": edge_columns(edge["description"]),
              "deployed": edge["edge_id"] in deployed} for edge in source["graph"]]
    examples = []
    for ex in source["examples"]:
        if ex["label"] not in keep_labels:
            continue
        examples.append({
            "label": ex["label"],
            "row": ex["sample_index"],
            "ready": ex["ready_states"],
            "transitions": ex["state_transitions"],
            "activeEdges": ex["active_edges"],
            "finalValues": ex["final_values"],
            "values": [[step["values"].get(col) for col in source["involved_columns"]]
                       for step in ex["timeline"]],
            "codes": [[source["state_names"].index(edge["state"]) for edge in step["edges"]]
                      for step in ex["timeline"]],
            "raw": ["".join("1" if edge["raw_active"] else "0" for edge in step["edges"])
                    for step in ex["timeline"]],
            "stable": ["".join("1" if edge["stable_active"] else "0" for edge in step["edges"])
                       for step in ex["timeline"]],
            "streak": [[edge["streak"] for edge in step["edges"]] for step in ex["timeline"]],
        })
    payload = {
        "dataset": source["dataset"],
        "seed": source["seed"],
        "samples": source["samples"],
        "steps": source["steps"],
        "states": source["state_names"],
        "columns": source["involved_columns"],
        "graph": graph,
        "t": [row["t"] for row in source["aggregate"]],
        "required": [row["required_streak"] for row in source["aggregate"]],
        "aggregate": [[row[name] for name in source["state_names"]] for row in source["aggregate"]],
        "examples": examples,
        "guided": {
            "seed": 20261447,
            "t": [round(row["t"], 4) for row in guided["trace"]],
            "actions": [row["actions"] for row in guided["trace"]],
            "active": [row["active_row_edges"] for row in guided["trace"]],
            "flips": [row["topology_flips"] for row in guided["trace"]],
            "actionEvents": guided["action_events"],
            "topologyAdditions": guided["topology_additions"],
            "topologyRemovals": guided["topology_removals"],
        },
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    fragment = TEMPLATE.replace("__PAYLOAD__", encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment)
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size,
                      "examples": len(examples), "edges": len(graph)}, indent=2))


TEMPLATE = r'''<section id="default-edge-trajectory-viz" aria-labelledby="detv-title">
  <style>
    #default-edge-trajectory-viz { color: var(--foreground); width: 100%; }
    #default-edge-trajectory-viz .detv-head { padding-right: 96px; }
    #default-edge-trajectory-viz .detv-sub { color: var(--muted-foreground); margin: 0.25rem 0 0; }
    #default-edge-trajectory-viz .detv-controls { margin: 1rem 0 0.75rem; }
    #default-edge-trajectory-viz .detv-control { min-width: 150px; flex: 1 1 150px; }
    #default-edge-trajectory-viz .detv-slider { flex: 4 1 320px; }
    #default-edge-trajectory-viz .detv-section { margin-top: 1rem; }
    #default-edge-trajectory-viz .detv-section h2 { margin-bottom: 0.35rem; }
    #default-edge-trajectory-viz .detv-split { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(260px, 0.75fr); gap: 1rem; align-items: start; }
    #default-edge-trajectory-viz .detv-summary { padding: 0.85rem; }
    #default-edge-trajectory-viz .detv-summary-line { display: flex; gap: 0.7rem; flex-wrap: wrap; align-items: baseline; }
    #default-edge-trajectory-viz .detv-value { font-weight: 500; }
    #default-edge-trajectory-viz .detv-legend { display: flex; flex-wrap: wrap; gap: 0.7rem; margin: 0.4rem 0; color: var(--muted-foreground); }
    #default-edge-trajectory-viz .detv-key { display: inline-flex; align-items: center; gap: 0.3rem; }
    #default-edge-trajectory-viz .detv-swatch { width: 0.72rem; height: 0.72rem; display: inline-block; }
    #default-edge-trajectory-viz .detv-plot { min-width: 0; }
    #default-edge-trajectory-viz .detv-plot svg { display: block; width: 100%; height: auto; }
    #default-edge-trajectory-viz svg text { fill: var(--foreground); font-size: 12px; font-weight: 400; }
    #default-edge-trajectory-viz svg .detv-muted { fill: var(--muted-foreground); }
    #default-edge-trajectory-viz .detv-network path { fill: none; stroke-width: 2; }
    #default-edge-trajectory-viz .detv-node { fill: color-mix(in srgb, var(--muted) 82%, transparent); stroke: var(--border); stroke-width: 1; }
    #default-edge-trajectory-viz .detv-node-selected { fill: color-mix(in srgb, var(--accent) 70%, transparent); stroke: var(--ring); }
    #default-edge-trajectory-viz .detv-edge-label { font-size: 11px; }
    #default-edge-trajectory-viz .detv-frame { fill: transparent; stroke: var(--border); stroke-width: 1; }
    #default-edge-trajectory-viz .detv-current { stroke: var(--foreground); stroke-width: 1.5; fill: none; pointer-events: none; }
    #default-edge-trajectory-viz .detv-hit { fill: transparent; }
    #default-edge-trajectory-viz .detv-state { font-weight: 500; }
    #default-edge-trajectory-viz .detv-table-state { white-space: nowrap; font-weight: 500; }
    #default-edge-trajectory-viz .detv-dot { display: inline-block; width: 0.62rem; height: 0.62rem; margin-right: 0.3rem; vertical-align: -0.04rem; }
    #default-edge-trajectory-viz .detv-note { margin: 0.45rem 0 0; color: var(--muted-foreground); }
    #default-edge-trajectory-viz .detv-axis-title { fill: var(--muted-foreground); }
    @media (max-width: 760px) {
      #default-edge-trajectory-viz .detv-split { grid-template-columns: 1fr; }
      #default-edge-trajectory-viz .detv-head { padding-right: 70px; }
    }
  </style>

  <header class="detv-head">
    <h1 id="detv-title">Default: dynamic constraint graph theo row và reverse time</h1>
    <p class="detv-sub">Replay 4.096 synthetic rows · 50 denoising steps · 16 verified catalog edges, trong đó 7 edges được deploy · seed 20261440</p>
  </header>

  <div class="viz-controls detv-controls" aria-label="Điều khiển trajectory">
    <label class="form-label detv-control">Sample
      <select class="form-select" id="detv-sample"></select>
    </label>
    <label class="form-label detv-control">Edge
      <select class="form-select" id="detv-edge"></select>
    </label>
    <label class="form-label detv-slider">Reverse step: <span id="detv-step-label" class="tabular-nums"></span>
      <input class="form-range" id="detv-step" type="range" min="0" max="49" value="30" step="1">
    </label>
  </div>

  <div class="detv-split">
    <section class="detv-plot" aria-labelledby="detv-graph-title">
      <h2 id="detv-graph-title">Active graph tại state đang chọn</h2>
      <svg class="detv-network" id="detv-network" viewBox="0 0 620 410" role="img" aria-label="Graph giữa các cột, màu cạnh biểu diễn trạng thái"></svg>
    </section>
    <aside class="card detv-summary" aria-live="polite">
      <div class="detv-summary-line"><span class="text-muted">Edge</span><span id="detv-selected-id" class="detv-value"></span><span id="detv-selected-class" class="viz-badge"></span></div>
      <p id="detv-selected-description"></p>
      <div class="detv-summary-line"><span class="text-muted">Trạng thái</span><span id="detv-selected-state" class="detv-state"></span></div>
      <div class="detv-summary-line"><span class="text-muted">Applicability</span><span id="detv-applicability" class="tabular-nums"></span></div>
      <div class="detv-summary-line"><span class="text-muted">Evidence streak</span><span id="detv-streak" class="tabular-nums"></span></div>
      <hr>
      <div id="detv-values"></div>
    </aside>
  </div>

  <section class="detv-section" aria-labelledby="detv-timeline-title">
    <h2 id="detv-timeline-title">State timeline của từng edge</h2>
    <div class="detv-legend" id="detv-legend"></div>
    <div class="detv-plot"><svg id="detv-heatmap" viewBox="0 0 1000 420" role="img" aria-label="Heatmap trạng thái của 16 edges qua 50 reverse steps"></svg></div>
  </section>

  <section class="detv-section" aria-labelledby="detv-pop-title">
    <h2 id="detv-pop-title">Toàn bộ 4.096 rows: graph trở nên đọc được và có thể hành động khi nào?</h2>
    <div class="detv-plot"><svg id="detv-aggregate" viewBox="0 0 1000 220" role="img" aria-label="Tỷ lệ row-edge theo trạng thái qua reverse time"></svg></div>
    <p class="detv-note text-small">Replay unguided dùng để quan sát state. Một replay guided độc lập ghi nhận 238 topology flips và 76 action events; vì khác seed, các action đó không được gán ngược vào các sample phía trên.</p>
  </section>

  <section class="detv-section" aria-labelledby="detv-table-title">
    <h2 id="detv-table-title">Snapshot của tất cả edges tại step đang chọn</h2>
    <div class="table-responsive"><table class="table table-sm"><thead><tr><th>Edge</th><th>Constraint</th><th>State</th><th class="text-end">Streak / K(t)</th></tr></thead><tbody id="detv-edge-table"></tbody></table></div>
  </section>

  <script type="application/json" id="detv-data">__PAYLOAD__</script>
  <script>
  (() => {
    const root = document.getElementById('default-edge-trajectory-viz');
    const data = JSON.parse(root.querySelector('#detv-data').textContent);
    const stateLabels = ['chưa đọc được', 'không áp dụng', 'đã thỏa', 'đang vi phạm', 'đủ bằng chứng'];
    const stateColors = ['var(--muted)', 'var(--border)', 'var(--green)', 'var(--orange)', 'var(--purple)'];
    const sampleSelect = root.querySelector('#detv-sample');
    const edgeSelect = root.querySelector('#detv-edge');
    const stepInput = root.querySelector('#detv-step');
    const saved = window.openai && window.openai.widgetState && window.openai.widgetState.privateContent;
    const ui = { sample: Math.min(saved?.sample ?? 0, data.examples.length - 1), edge: Math.min(saved?.edge ?? 0, data.graph.length - 1), step: Math.min(saved?.step ?? 30, data.steps - 1) };
    data.examples.forEach((ex, i) => sampleSelect.add(new Option(`${ex.label} · row ${ex.row} · ${ex.transitions} transitions`, i)));
    data.graph.forEach((edge, i) => edgeSelect.add(new Option(`${edge.edge_id} · ${edge.description}`, i)));
    sampleSelect.value = ui.sample; edgeSelect.value = ui.edge; stepInput.value = ui.step;

    const legend = root.querySelector('#detv-legend');
    stateLabels.forEach((label, i) => {
      const item = document.createElement('span'); item.className = 'detv-key text-small';
      item.innerHTML = `<span class="detv-swatch" style="background:${stateColors[i]}"></span>${label}`;
      legend.appendChild(item);
    });

    const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const fmt = value => typeof value === 'number' ? new Intl.NumberFormat('en-US', {maximumFractionDigits: 2}).format(value) : value;
    function persist() {
      if (window.openai?.setWidgetState) window.openai.setWidgetState({modelContent:{dataset:data.dataset,sample:data.examples[ui.sample].label,row:data.examples[ui.sample].row,step:ui.step,t:data.t[ui.step],edge:data.graph[ui.edge].edge_id,state:stateLabels[data.examples[ui.sample].codes[ui.step][ui.edge]]},privateContent:ui}).catch(() => {});
    }
    function svgEl(tag, attrs={}, text='') {
      const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs).forEach(([k,v]) => node.setAttribute(k,v)); if (text) node.textContent=text; return node;
    }
    function edgeColor(code) { return stateColors[code]; }
    function drawNetwork(ex) {
      const svg = root.querySelector('#detv-network'); svg.replaceChildren();
      svg.appendChild(svgEl('rect',{x:0.5,y:0.5,width:619,height:409,class:'detv-frame'}));
      const columns = [...new Set(data.graph.flatMap(e => e.columns))];
      const center = {x:310,y:205}, rx=238, ry=150;
      const pos = {};
      columns.forEach((col,i) => { const a=-Math.PI/2 + i*2*Math.PI/columns.length; pos[col]={x:center.x+rx*Math.cos(a),y:center.y+ry*Math.sin(a)}; });
      data.graph.forEach((edge,i) => {
        const code=ex.codes[ui.step][i], from=pos[edge.columns[0]], to=pos[edge.columns[edge.columns.length-1]];
        if (!from || !to || from===to) return;
        const bend=((i%5)-2)*12, mx=(from.x+to.x)/2+bend, my=(from.y+to.y)/2-bend;
        const path=svgEl('path',{d:`M${from.x},${from.y} Q${mx},${my} ${to.x},${to.y}`,stroke:edgeColor(code),opacity:edge.deployed?(code===0?0.28:(code===1?0.45:0.9)):0.22,'stroke-dasharray':edge.deployed?'':'4 4'});
        path.appendChild(svgEl('title',{},`${edge.edge_id}: ${stateLabels[code]} — ${edge.description}`));
        svg.appendChild(path);
      });
      columns.forEach(col => {
        const p=pos[col], selected=data.graph[ui.edge].columns.includes(col);
        svg.appendChild(svgEl('rect',{x:p.x-48,y:p.y-15,width:96,height:30,rx:6,class:selected?'detv-node detv-node-selected':'detv-node'}));
        svg.appendChild(svgEl('text',{x:p.x,y:p.y+4,'text-anchor':'middle'},col));
      });
      svg.appendChild(svgEl('text',{x:12,y:24,class:'detv-muted'},`Sample ${ex.label} · step ${ui.step} · t=${data.t[ui.step].toFixed(4)}`));
    }
    function drawHeatmap(ex) {
      const svg=root.querySelector('#detv-heatmap'); svg.replaceChildren();
      const W=1000,H=420,left=268,right=16,top=32,bottom=42;
      const cw=(W-left-right)/data.steps, ch=(H-top-bottom)/data.graph.length;
      svg.appendChild(svgEl('rect',{x:left,y:top,width:W-left-right,height:H-top-bottom,class:'detv-frame'}));
      data.graph.forEach((edge,e) => {
        const y=top+e*ch;
        const label=svgEl('text',{x:left-8,y:y+ch*0.68,'text-anchor':'end',class:e===ui.edge?'detv-state':'detv-muted'},edge.edge_id);
        label.appendChild(svgEl('title',{},edge.description)); svg.appendChild(label);
        ex.codes.forEach((row,s) => {
          const code=row[e];
          const r=svgEl('rect',{x:left+s*cw+0.25,y:y+0.4,width:Math.max(cw-0.5,1),height:Math.max(ch-0.8,1),fill:edgeColor(code),opacity:e===ui.edge?1:0.78});
          r.appendChild(svgEl('title',{},`${edge.edge_id} · step ${s} · t=${data.t[s]} · ${stateLabels[code]}`)); svg.appendChild(r);
        });
      });
      [0,10,20,30,40,49].forEach(s => { const x=left+(s+0.5)*cw; svg.appendChild(svgEl('text',{x,y:H-20,'text-anchor':'middle',class:'detv-muted'},data.t[s].toFixed(2))); });
      svg.appendChild(svgEl('text',{x:left+(W-left-right)/2,y:H-3,'text-anchor':'middle',class:'detv-axis-title'},'normalized reverse time t: noisy  →  clean'));
      svg.appendChild(svgEl('rect',{x:left+ui.step*cw,y:top,width:cw,height:H-top-bottom,class:'detv-current'}));
      const hit=svgEl('rect',{x:left,y:top,width:W-left-right,height:H-top-bottom,class:'detv-hit'});
      hit.addEventListener('pointerdown',evt => { const box=svg.getBoundingClientRect(); const px=(evt.clientX-box.left)*W/box.width; const py=(evt.clientY-box.top)*H/box.height; ui.step=Math.max(0,Math.min(49,Math.floor((px-left)/cw))); ui.edge=Math.max(0,Math.min(data.graph.length-1,Math.floor((py-top)/ch))); stepInput.value=ui.step; edgeSelect.value=ui.edge; render(); persist(); });
      svg.appendChild(hit);
    }
    function drawAggregate() {
      const svg=root.querySelector('#detv-aggregate'); svg.replaceChildren();
      const W=1000,H=220,left=64,right=18,top=18,bottom=44,plotW=W-left-right,plotH=H-top-bottom;
      svg.appendChild(svgEl('rect',{x:left,y:top,width:plotW,height:plotH,class:'detv-frame'}));
      for(let s=0;s<data.steps;s++) {
        const total=data.aggregate[s].reduce((a,b)=>a+b,0), x=left+s*plotW/data.steps, w=plotW/data.steps+0.2;
        let y=top+plotH;
        data.aggregate[s].forEach((count,code) => { const h=count/total*plotH; y-=h; svg.appendChild(svgEl('rect',{x,y,width:w,height:h,fill:edgeColor(code),opacity:0.82})); });
      }
      [0,20,30,40,49].forEach(s => { const x=left+(s+0.5)*plotW/data.steps; svg.appendChild(svgEl('text',{x,y:H-20,'text-anchor':'middle',class:'detv-muted'},data.t[s].toFixed(2))); });
      svg.appendChild(svgEl('text',{x:left+plotW/2,y:H-3,'text-anchor':'middle',class:'detv-axis-title'},'normalized reverse time t'));
      svg.appendChild(svgEl('text',{x:16,y:top+plotH/2,transform:`rotate(-90 16 ${top+plotH/2})`,'text-anchor':'middle',class:'detv-axis-title'},'share of row-edge states'));
      svg.appendChild(svgEl('line',{x1:left+(ui.step+0.5)*plotW/data.steps,x2:left+(ui.step+0.5)*plotW/data.steps,y1:top,y2:top+plotH,stroke:'var(--foreground)','stroke-width':1.5}));
    }
    function renderDetails(ex) {
      const edge=data.graph[ui.edge], code=ex.codes[ui.step][ui.edge], raw=ex.raw[ui.step][ui.edge]==='1', stable=ex.stable[ui.step][ui.edge]==='1', streak=ex.streak[ui.step][ui.edge];
      root.querySelector('#detv-step-label').textContent=`${ui.step}/49 · t=${data.t[ui.step].toFixed(4)} · K(t)=${data.required[ui.step]}`;
      root.querySelector('#detv-selected-id').textContent=edge.edge_id;
      root.querySelector('#detv-selected-class').textContent=`${edge.classification} · ${edge.deployed?'deployed':'catalog-only'}`;
      root.querySelector('#detv-selected-description').textContent=edge.description;
      const state=root.querySelector('#detv-selected-state'); state.textContent=stateLabels[code]; state.style.color=edgeColor(code);
      root.querySelector('#detv-applicability').textContent=`raw=${raw?'on':'off'} · stabilized=${stable?'on':'off'} · readable cutoff=${edge.cutoff}`;
      root.querySelector('#detv-streak').textContent=`${streak} / ${data.required[ui.step]}`;
      const values=edge.columns.map(col => { const ix=data.columns.indexOf(col); return `<tr><th>${esc(col)}</th><td class="text-end">${esc(fmt(ex.values[ui.step][ix]))}</td></tr>`; }).join('');
      root.querySelector('#detv-values').innerHTML=`<table class="table table-sm"><thead><tr><th>Column liên quan</th><th class="text-end">x̂₀(t)</th></tr></thead><tbody>${values}</tbody></table>`;
      const rows=data.graph.map((item,i) => { const c=ex.codes[ui.step][i]; return `<tr><td><button class="btn btn-ghost" type="button" data-edge="${i}">${item.edge_id}</button></td><td>${esc(item.description)}<br><span class="text-small text-muted">${item.deployed?'deployed':'catalog-only'}</span></td><td class="detv-table-state" style="color:${edgeColor(c)}"><span class="detv-dot" style="background:${edgeColor(c)}"></span>${stateLabels[c]}</td><td class="text-end">${ex.streak[ui.step][i]} / ${data.required[ui.step]}</td></tr>`; }).join('');
      const body=root.querySelector('#detv-edge-table'); body.innerHTML=rows;
      body.querySelectorAll('[data-edge]').forEach(btn => btn.addEventListener('click',()=>{ui.edge=Number(btn.dataset.edge);edgeSelect.value=ui.edge;render();persist();}));
    }
    function render() { const ex=data.examples[ui.sample]; drawNetwork(ex); drawHeatmap(ex); drawAggregate(); renderDetails(ex); }
    sampleSelect.addEventListener('change',()=>{ui.sample=Number(sampleSelect.value);render();persist();});
    edgeSelect.addEventListener('change',()=>{ui.edge=Number(edgeSelect.value);render();persist();});
    stepInput.addEventListener('input',()=>{ui.step=Number(stepInput.value);render();persist();});
    window.addEventListener('openai:set_globals',event=>{const s=event.detail?.globals?.widgetState?.privateContent;if(!s)return;ui.sample=Math.min(s.sample??ui.sample,data.examples.length-1);ui.edge=Math.min(s.edge??ui.edge,data.graph.length-1);ui.step=Math.min(s.step??ui.step,data.steps-1);sampleSelect.value=ui.sample;edgeSelect.value=ui.edge;stepInput.value=ui.step;render();});
    render();
  })();
  </script>
</section>
'''


if __name__ == "__main__":
    main()
