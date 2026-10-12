"""Self-contained HTML report: benchmark summary + the prioritised review queue."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from PIL import Image

TYPE_LABEL = {
    "missing_object": "Missing label",
    "wrong_class": "Wrong class",
    "loose_box": "Loose box",
    "spurious_box": "Label on nothing",
    "duplicate_box": "Duplicate",
}
TYPE_SLOT = {"missing_object": 1, "wrong_class": 2, "loose_box": 3, "spurious_box": 4, "duplicate_box": 5}
SERIES = ["ReviewLens ranking", "Random order (status quo)", "Most labels first"]

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--border:#e2e0da;--text-primary:#0b0b0b;
--text-secondary:#52514e;--text-muted:#7a7873;--grid:#e8e6e1;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;
--series-4:#eda100;--series-5:#e87ba4}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--surface-2:#242422;--border:#3a3936;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--text-muted:#9a988f;--grid:#33322f;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;
--series-5:#d55181}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;--border:#3a3936;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#9a988f;--grid:#33322f;--series-1:#3987e5;
--series-2:#d95926;--series-3:#199e70;--series-4:#c98500;--series-5:#d55181}
*{box-sizing:border-box}
body{margin:0}
.viz-root{background:var(--surface-1);color:var(--text-primary);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
min-height:100vh;padding:32px 20px}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:40px 0 10px}
p.lede{color:var(--text-secondary);margin:0 0 24px;max-width:760px}
.muted{color:var(--text-muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.tile{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:16px}
.tile .v{font-size:32px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.tile .k{color:var(--text-secondary);font-size:13px}
.card{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:16px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--border);font-size:14px}
th{color:var(--text-secondary);font-weight:600}
td.n,th.n{text-align:right}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--text-secondary);margin:4px 0 8px}
.sw{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.chip{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:middle;margin-right:6px}
.chart{position:relative}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:8px 10px;font-size:13px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;white-space:nowrap}
.queue{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.q{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.q .img{position:relative;line-height:0}
.q .img img{width:100%;display:block}
.q .img svg{position:absolute;inset:0;width:100%;height:100%}
.q .meta{padding:10px 12px;font-size:13px}
.q .meta b{font-variant-numeric:tabular-nums}
.q ul{margin:6px 0 0;padding-left:18px;color:var(--text-secondary)}
.tag{font-size:11px;padding:1px 6px;border-radius:4px;border:1px solid var(--border);color:var(--text-secondary);margin-left:6px}
details summary{cursor:pointer;color:var(--text-secondary)}
"""


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _line_chart(bench: dict) -> str:
    W, H, L, R, T, B = 760, 320, 52, 20, 16, 44
    xs = bench["budgets"]
    def px(x): return L + (W - L - R) * x
    def py(y): return T + (H - T - B) * (1 - y)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Share of injected errors caught vs share of files reviewed">']
    for g in (0, .25, .5, .75, 1):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{py(g)}" y2="{py(g)}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{L - 8}" y="{py(g) + 4}" text-anchor="end" font-size="12" fill="var(--text-muted)">{_pct(g)}</text>')
    for g in (0, .2, .4, .6, .8, 1):
        parts.append(f'<text x="{px(g)}" y="{H - B + 18}" text-anchor="middle" font-size="12" fill="var(--text-muted)">{_pct(g)}</text>')
    parts.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle" font-size="12" fill="var(--text-secondary)">share of files reviewed, in queue order</text>')
    for k, name in enumerate(SERIES):
        ys = [0.0] + bench["curves"][name]
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip([0.0] + xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--series-{k + 1})" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in zip(xs, ys[1:]):
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="var(--series-{k + 1})" '
                         f'stroke="var(--surface-1)" stroke-width="2"/>')
    # direct label on the headline series at 20%
    i20 = xs.index(0.2)
    y20 = bench["curves"][SERIES[0]][i20]
    parts.append(f'<text x="{px(.2) - 10}" y="{py(y20) - 12}" text-anchor="end" font-size="12" fill="var(--text-primary)">'
                 f'{_pct(y20)} caught at 20%</text>')
    parts.append(f'<line class="xh" x1="0" x2="0" y1="{T}" y2="{H - B}" stroke="var(--text-muted)" stroke-width="1" '
                 f'stroke-dasharray="3 3" visibility="hidden"/>')
    parts.append(f'<rect class="hit" x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="transparent"/>')
    parts.append("</svg>")
    return "".join(parts)


def _thumb(path: Path, max_w: int = 640) -> tuple[str, int, int]:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w > max_w:
        im = im.resize((max_w, round(h * max_w / w)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode(), w, h


def _queue_card(rank: int, row: dict, ds, images_dir: Path, truth: list[dict] | None) -> str:
    img = ds.images[row["image_id"]]
    p = images_dir / img["file_name"]
    if not p.exists():
        return ""
    b64, w, h = _thumb(p)
    boxes, items = [], []
    for it in row["issues"][:6]:
        x, y, bw, bh = it.bbox
        slot = TYPE_SLOT[it.type]
        boxes.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" fill="none" stroke="var(--series-{slot})" '
                     f'stroke-width="{max(2, w / 220):.1f}" vector-effect="non-scaling-stroke"/>')
        items.append(f'<li><span class="chip" style="background:var(--series-{slot})"></span>'
                     f'{TYPE_LABEL[it.type]} <span class="muted">· {it.score:.2f}</span> — {html.escape(it.detail)}</li>')
    truth_note = ""
    if truth is not None:
        t = [e for e in truth if e["image_id"] == row["image_id"]]
        truth_note = (f'<span class="tag">answer key: {len(t)} injected ({", ".join(sorted({TYPE_LABEL[e["type"]] for e in t}))})</span>'
                      if t else '<span class="tag">answer key: none injected</span>')
    return (f'<div class="q"><div class="img"><img loading="lazy" src="data:image/jpeg;base64,{b64}" alt="{html.escape(img["file_name"])}">'
            f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none">{"".join(boxes)}</svg></div>'
            f'<div class="meta">#{rank} · <b>{html.escape(img["file_name"])}</b> · ~<b>{row["expected_errors"]:.1f}</b> likely errors '
            f'· {row["n_labels"]} labels {truth_note}<ul>{"".join(items)}</ul></div></div>')


def render(out_path: str | Path, ds, rows: list[dict], images_dir: str | Path, bench: dict | None = None,
           truth: list[dict] | None = None, title: str = "ReviewLens — review the right files first",
           top_n: int = 36, notes_html: str = "") -> Path:
    images_dir = Path(images_dir)
    s = [(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
         f'<title>{html.escape(title)}</title><style>{CSS}</style></head><body><div class="viz-root"><div class="wrap">'),
         f'<h1>{html.escape(title)}</h1>',
         ('<p class="lede">ReviewLens compares the human labels waiting for review against a detector\'s predictions, '
         'flags likely mistakes (missing labels, wrong class, loose boxes, labels on nothing, duplicates), '
         'and orders the review queue so reviewers open the riskiest files first.</p>')]
    if bench:
        c, e = bench["curves"], bench["effort_for_80pct"]
        i20 = bench["budgets"].index(0.2)
        s.append('<div class="tiles">')
        s.append(f'<div class="tile"><div class="v">{_pct(c[SERIES[0]][i20])}</div><div class="k">of injected errors sit in the top 20% of the ReviewLens queue '
                 f'(random order: {_pct(c[SERIES[1]][i20])})</div></div>')
        s.append(f'<div class="tile"><div class="v">{_pct(e[SERIES[0]])}</div><div class="k">of files reviewed to catch 80% of errors '
                 f'(random order: {_pct(e[SERIES[1]])})</div></div>')
        s.append(f'<div class="tile"><div class="v">{bench["n_injected"]}</div><div class="k">errors injected into {bench["n_labels"]:,} labels on '
                 f'{bench["n_images"]} images ({bench["n_images_with_errors"]} files affected)</div></div>')
        pa = bench["issue_precision_at"]
        if 100 in pa:
            s.append(f'<div class="tile"><div class="v">{_pct(pa[100])}</div><div class="k">of the top 100 individual flags point at an injected error '
                     f'(many of the rest are real mistakes in the original data, see notes)</div></div>')
        s.append('</div>')

        s.append('<h2>Errors caught vs. review effort</h2><div class="legend">')
        for k, name in enumerate(SERIES):
            s.append(f'<span><span class="sw" style="background:var(--series-{k + 1})"></span>{name}</span>')
        s.append(f'</div><div class="card chart" id="c1">{_line_chart(bench)}<div class="tip"></div></div>')
        s.append('<details><summary>Table view</summary><table><tr><th>Files reviewed</th>'
                 + "".join(f'<th class="n">{n}</th>' for n in SERIES) + "</tr>")
        for i, b in enumerate(bench["budgets"]):
            s.append(f'<tr><td>{_pct(b)}</td>' + "".join(f'<td class="n">{_pct(c[n][i])}</td>' for n in SERIES) + "</tr>")
        s.append("</table></details>")

        s.append('<h2>Which mistakes does it find?</h2><div class="card"><table><tr><th>Mistake type</th><th class="n">Injected</th>'
                 '<th class="n">Pointed at (any flag on the box)</th><th class="n">…with the right diagnosis</th></tr>')
        for t, v in bench["per_type"].items():
            if v["injected"]:
                s.append(f'<tr><td><span class="chip" style="background:var(--series-{TYPE_SLOT[t]})"></span>{TYPE_LABEL[t]}</td>'
                         f'<td class="n">{v["injected"]}</td><td class="n">{_pct(v["located"] / v["injected"])}</td>'
                         f'<td class="n">{_pct(v["typed"] / v["injected"])}</td></tr>')
        s.append("</table></div>")

        s.append('<h2>Which annotator needs attention?</h2><p class="muted">Annotator 1 was simulated as careless (3× error rate). '
                 'The flag score is computed without the answer key.</p><div class="card"><table><tr><th>Annotator</th>'
                 '<th class="n">Images</th><th class="n">True errors / 100 images</th><th class="n">ReviewLens flag score / 100 images</th></tr>')
        for a in bench["annotators"]:
            s.append(f'<tr><td>{a["annotator"]}</td><td class="n">{a["images"]}</td><td class="n">{a["true_errors_per_100"]:.0f}</td>'
                     f'<td class="n">{a["flag_score_per_100"]:.0f}</td></tr>')
        s.append("</table></div>")
    if notes_html:
        s.append(notes_html)
    s.append(f'<h2>Review queue — top {top_n} files</h2><div class="legend">')
    for t, slot in TYPE_SLOT.items():
        s.append(f'<span><span class="chip" style="background:var(--series-{slot})"></span>{TYPE_LABEL[t]}</span>')
    s.append('</div><div class="queue">')
    for k, row in enumerate(rows[:top_n]):
        s.append(_queue_card(k + 1, row, ds, images_dir, truth))
    s.append("</div>")
    if bench:
        s.append(_tooltip_js(bench))
    s.append("</div></div></body></html>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(s), encoding="utf-8")
    return out


def _tooltip_js(bench: dict) -> str:
    import json
    data = json.dumps({"x": bench["budgets"], "series": SERIES, "y": [bench["curves"][n] for n in SERIES]})
    return """<script>(function(){const d=__DATA__;const box=document.getElementById('c1');if(!box)return;
const svg=box.querySelector('svg'),tip=box.querySelector('.tip'),xh=svg.querySelector('.xh'),hit=svg.querySelector('.hit');
const L=52,R=20,W=760;hit.addEventListener('mousemove',e=>{const pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;
const p=pt.matrixTransform(svg.getScreenCTM().inverse());const fx=(p.x-L)/(W-L-R);let i=0,b=9;
d.x.forEach((x,k)=>{if(Math.abs(x-fx)<b){b=Math.abs(x-fx);i=k}});const X=L+(W-L-R)*d.x[i];xh.setAttribute('x1',X);xh.setAttribute('x2',X);
xh.setAttribute('visibility','visible');tip.innerHTML='<b>'+Math.round(d.x[i]*100)+'% of files reviewed</b><br>'+d.series.map((s,k)=>
'<span class="sw" style="background:var(--series-'+(k+1)+')"></span>'+s+': <b>'+Math.round(d.y[k][i]*100)+'%</b>').join('<br>');
tip.style.display='block';const r=box.getBoundingClientRect();let left=e.clientX-r.left+14;if(left>r.width-260)left-=280;
tip.style.left=left+'px';tip.style.top=(e.clientY-r.top-10)+'px'});
hit.addEventListener('mouseleave',()=>{tip.style.display='none';xh.setAttribute('visibility','hidden')});})();</script>""".replace("__DATA__", data)
