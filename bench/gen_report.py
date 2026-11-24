#!/usr/bin/env python3
"""Render the interactive bench report (self-contained HTML, no external assets)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "bench175.json")))
A = D["ctx200k-1stream"]
cats = A["categories"]
order = ["code", "reasoning", "json", "prose", "chat", "edit", "summary"]
labels = {"code": "Code", "reasoning": "Reasoning / math", "json": "JSON", "prose": "Prose",
          "chat": "Chat / explain", "edit": "Edit / rewrite", "summary": "Summarize"}
vals = [(labels[k], cats[k]["decode"], cats[k]["gen"]) for k in order]

TMAX_C, SCALE_W = 200.0, 540.0
rows_n = len(vals)
chart_h = rows_n * 30 + 30
bar_rows, grid = [], []
for tick in range(0, 201, 50):
    gx = 170 + SCALE_W * tick / TMAX_C
    grid.append(f'<line x1="{gx:.1f}" y1="0" x2="{gx:.1f}" y2="{rows_n * 30 + 4}" stroke="#232830"/>')
    grid.append(f'<text x="{gx:.1f}" y="{rows_n * 30 + 20}" class="lbl" text-anchor="middle">{tick}</text>')
y = 0
for name, v, gen in vals:
    w = SCALE_W * v / TMAX_C
    bar_rows.append(f'<g transform="translate(0,{y})"><text x="0" y="14" class="lbl">{name}</text>'
                    f'<rect x="170" y="2" width="{w:.1f}" height="18" rx="3" fill="url(#g1)"/>'
                    f'<text x="{170 + w + 8:.1f}" y="16" class="val">{v:.1f} t/s</text></g>')
    y += 30
cat_svg = (f'<svg viewBox="0 0 780 {chart_h}" class="chart"><defs><linearGradient id="g1" x1="0" x2="1">'
           '<stop offset="0" stop-color="#2f81f7"/><stop offset="1" stop-color="#7ee787"/></linearGradient></defs>'
           + "".join(grid) + "".join(bar_rows) + f'<line x1="170" y1="0" x2="170" y2="{rows_n * 30 + 4}" stroke="#484f58"/></svg>')

conf = [("ctx200k-1stream", "200K · 1 slot", "#2f81f7"),
        ("ctx65k-2stream", "65K · 2 slots", "#7ee787"),
        ("ctx32k-4stream", "32K · 4 slots", "#d2a8ff")]
series = []
for key, label, color in conf:
    pts = {int(k): v["aggregate"] for k, v in D.get(key, {}).get("levels", {}).items()}
    if not pts:
        pts = {1: cats["chat"]["decode"]}
    series.append((label, color, pts))

TMAX, X0, X1, Y0, Y1 = 300.0, 70.0, 760.0, 22.0, 210.0
levels = [1, 2, 4]
group_w, bar_w = (X1 - X0) / len(levels), 30.0
parts = []
for tick in range(0, 301, 50):
    yy = Y1 - (Y1 - Y0) * tick / TMAX
    parts.append(f'<line x1="{X0}" y1="{yy:.1f}" x2="{X1}" y2="{yy:.1f}" stroke="#232830"/>')
    parts.append(f'<text x="{X0 - 10:.1f}" y="{yy + 4:.1f}" class="lbl" text-anchor="end">{tick}</text>')
for gi, lv in enumerate(levels):
    gx = X0 + gi * group_w
    present = [(lab, col, pts[lv]) for lab, col, pts in series if lv in pts]
    span = len(present) * (bar_w + 8)
    bx = gx + (group_w - span) / 2
    for lab, col, v in present:
        h = (Y1 - Y0) * v / TMAX
        parts.append(f'<rect x="{bx:.1f}" y="{Y1 - h:.1f}" width="{bar_w}" height="{h:.1f}" rx="3" fill="{col}"/>')
        parts.append(f'<text x="{bx + bar_w / 2:.1f}" y="{Y1 - h - 6:.1f}" class="val" text-anchor="middle">{v:.0f}</text>')
        bx += bar_w + 8
    parts.append(f'<text x="{gx + group_w / 2:.1f}" y="{Y1 + 20:.1f}" class="lbl" text-anchor="middle">{lv} stream{"s" if lv > 1 else ""}</text>')
parts.append(f'<line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="#484f58"/>')
lx = X0 + 6
for lab, col, _ in series:
    parts.append(f'<rect x="{lx:.1f}" y="4" width="10" height="10" rx="2" fill="{col}"/>')
    parts.append(f'<text x="{lx + 15:.1f}" y="13" class="lbl">{lab}</text>')
    lx += 24 + len(lab) * 6.2
tp_svg = '<svg viewBox="0 0 780 245" class="chart">' + "".join(parts) + "</svg>"

depth_rows = "".join(
    f"<tr><td>{lab}</td><td>{e['prompt_tokens']:,}</td><td>{e['ttft']:.2f} s</td>"
    f"<td>{e['prefill']:,.0f}</td><td>{e.get('decode_at_depth','—')}</td></tr>" for lab, e in A["depth"].items())
q = A["quality"]
qrows = "".join(
    f"<tr><td>{it['q']}</td><td class='mono'>{it['answer'][:110].replace(chr(10),' ')}</td>"
    f"<td class='{'ok' if it['ok'] else 'bad'}'>{'✓' if it['ok'] else '✗'}</td></tr>" for it in q["items"])
t = A["tools"]
peak = max(v for _, v, _ in vals)

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>TensorFold XPU · Hermes Qwen3.8-27B — bench</title><style>
:root{{--bg:#0b0d10;--card:#12151a;--line:#232830;--fg:#e6edf3;--dim:#8b949e;--acc:#7ee787}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Inter,sans-serif;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:15px;margin:28px 0 10px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}}
.sub{{color:var(--dim);font-size:13px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:18px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}}
.kpi .n{{font-size:26px;font-weight:650;letter-spacing:-.02em}} .kpi .t{{color:var(--dim);font-size:12px;margin-top:2px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:12px}}
.chart{{width:100%;height:auto;display:block}} .lbl{{fill:var(--dim);font-size:12px}} .val{{fill:var(--fg);font-size:12px;font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line)}}
th{{color:var(--dim);font-weight:500;font-size:12px}} .mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;color:#c9d1d9}}
.ok{{color:var(--acc)}} .bad{{color:#ff7b72}} .note{{color:var(--dim);font-size:12px;margin-top:8px}}
</style></head><body>
<h1>Qwen3.8-27B-Hermes-adapted GPTQ-Int4 on TensorFold XPU</h1>
<div class="sub">Arc Pro B70 · 1 card · DFlash2 drafter · int8 KV · Hermes-adapted fine-tune of Qwen3.8-27B (GPTQ g128, dense F16 head) · 2026-11-24</div>
<div class="grid">
<div class="kpi"><div class="n">{peak:.0f}</div><div class="t">peak single-stream decode (t/s)</div></div>
<div class="kpi"><div class="n">262.9</div><div class="t">aggregate, 4 streams (t/s)</div></div>
<div class="kpi"><div class="n">1,869</div><div class="t">prefill at 6K prompt (t/s)</div></div>
<div class="kpi"><div class="n">200K</div><div class="t">context · needle recall ✓</div></div>
<div class="kpi"><div class="n">{q['score']}/{q['total']}</div><div class="t">quality checks (greedy)</div></div>
<div class="kpi"><div class="n">{t['wall']:.2f}s</div><div class="t">required tool call, args correct</div></div>
</div>
<h2>Decode by category — single stream, 200K context</h2>
<div class="card">{cat_svg}<div class="note">300 generated tokens per category except JSON (36), edit (35) and summary (164). TTFT from streaming; decode = tokens ÷ decode window.</div></div>
<h2>Throughput vs concurrency</h2>
<div class="card">{tp_svg}<div class="note">Same chat prompt, 250 tokens each. Context traded for slots: 200K → 1, 65K → 2, 32K → 4. Aggregate rises because the GPU idles most of the time at one stream.</div></div>
<h2>Depth — prefill and decode with a long document in context</h2>
<div class="card"><table><tr><th>depth</th><th>prompt tokens</th><th>TTFT</th><th>prefill t/s</th><th>decode t/s at depth</th></tr>
{depth_rows}</table><div class="note">Needle recall verified separately at 108,823 and 141,603 prompt tokens; both returned the buried code exactly.</div></div>
<h2>Tool calling</h2>
<div class="card"><table><tr><th>case</th><th>result</th></tr>
<tr><td>required, "Buy 250 MSFT @ 415.25 on NASDAQ, day"</td><td class="mono">{t['name']}({t['args']}) <span class="ok">✓</span></td></tr>
<tr><td>auto, "what's AAPL's price"</td><td class="mono">get_price({{"ticker":"AAPL"}}) <span class="ok">✓</span></td></tr>
<tr><td>auto, "say hello, no tools"</td><td class="mono">no tool call <span class="ok">✓</span></td></tr>
<tr><td>latency (required call)</td><td>{t['wall']:.2f} s wall</td></tr></table>
<div class="note">Structured output and required tool calls run through libtfgrammar (xgrammar v0.2.8), built from source and installed beside the server.</div></div>
<h2>Quality — 10 checkable questions, greedy</h2>
<div class="card"><table><tr><th>question</th><th>answer (truncated)</th><th></th></tr>
{qrows}</table></div>
<h2>Hardware during the run</h2>
<div class="card"><table><tr><th>signal</th><th>value</th></tr>
<tr><td>board power (energy counter delta, card under load)</td><td>≈ 190 W</td></tr>
<tr><td>board power, idle</td><td>≈ 6 W</td></tr>
<tr><td>GPU temperature under decode</td><td>36–44 °C</td></tr>
<tr><td>idle card temperature</td><td>24–26 °C</td></tr></table>
<div class="note">The <code>xe</code> driver on this box does not export <code>power1_input</code>; power is derived from <code>energy1_input</code> deltas.</div></div>
<h2>Method</h2>
<div class="card"><div class="note">
Engine: TensorFold <code>tensorfold-native serve</code>, <code>--context 200000 --parallel 1 --prompt-cache-gib 0.25 --drafter DFlash2 --no-thinking</code>, <code>XPU_KV_INT8=1</code>, one Arc Pro B70.<br>
Model: Qwen3.8-27B + Hermes-adapted LoRA (alpha 64) merged to BF16, then GPTQ int4 g128, dense tensors F16, vision tower dropped, 17 GB.<br>
Client: streaming SSE; TTFT = first content chunk; decode = (tokens−1) ÷ (last chunk − first chunk). Greedy sampling.<br>
Caveat: single-card single-box run, not the project's easy-mode weighted suite.
</div></div></body></html>"""

out = os.path.join(HERE, "bench175.html")
open(out, "w").write(html)
print("wrote", out, len(html))
