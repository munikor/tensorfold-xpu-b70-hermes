#!/usr/bin/env python3
"""Bench harness for a TensorFold XPU lane serving Qwen3.8-27B (GPTQ int4).

Usage:  python3 bench175.py single ctx200k-1stream
        python3 bench175.py quick  ctx200k-1stream     # tools + quality only
        python3 bench175.py conc   ctx32k-4stream 1,2,4
Set BASE to your endpoint."""
import json, sys, time, threading, urllib.request, os

BASE = "http://127.0.0.1:8202"
OUT = "bench175.json"

PROMPTS = {
    "code": ("Write a Python function that merges two sorted lists into one sorted list. "
             "Include type hints, a docstring, and handle duplicates. About 40 lines.", 300),
    "reasoning": ("Compute step by step: 17*23 + 145/5 - 12^2. Then verify the result by a second "
                  "method and explain each step. About 250 words.", 300),
    "prose": ("Write a 250-word explanation of how a limit order book matches buyers and sellers, "
              "with a worked example.", 300),
    "json": ("Return only a JSON object with keys: ticker, side, qty, limit_price, venue, tif. "
             "Values: NVDA, buy, 100, 182.50, ARCA, DAY. No prose, no markdown fences.", 120),
    "edit": ("Rewrite the following paragraph in a formal register, keeping all facts: "
             "'so we looked at the numbers and honestly they were kind of a mess, the fills "
             "slipped like crazy and the desk lost money on three out of five names.'", 150),
    "summary": ("Summarize this in 150 words: A market maker quotes both sides of a two-sided market. "
                "It earns the spread but takes inventory risk. When order flow becomes one-sided, the "
                "maker widens quotes or hedges in a correlated instrument. Adverse selection arises when "
                "informed traders pick off stale quotes, so makers monitor fill toxicity and adjust.", 200),
    "chat": ("Explain how a limit order differs from a market order, then give two concrete examples "
             "from equity trading.", 250),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "place_order",
        "description": "Place a limit order",
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string"}, "side": {"type": "string", "enum": ["buy", "sell"]},
            "qty": {"type": "integer"}, "limit_price": {"type": "number"},
            "venue": {"type": "string"}, "tif": {"type": "string"}},
            "required": ["ticker", "side", "qty", "limit_price"]}}},
]
TOOL_PROMPT = "Buy 250 shares of MSFT with a limit price of 415.25 on NASDAQ, day order."

QUALITY = [
    ("What is 17*23?", ["391"]),
    ("A trader buys 100 shares at 50 and sells at 55. Profit in dollars?", ["500"]),
    ("If a limit buy is at 99 and the market is at 100, does it fill immediately?", ["no", "No", "NO"]),
    ("What does TIF stand for in equities?", ["time in force", "Time In Force", "Time in force"]),
    ("Compute 12^2.", ["144"]),
    ("Which is riskier for a market maker: one-sided flow or balanced flow?", ["one-sided", "One-sided", "one sided"]),
    ("What is 145/5?", ["29"]),
    ("Name the order type that guarantees execution but not price.", ["market", "Market"]),
    ("If spread is 0.02 and you trade 1,000 shares as a maker, gross edge in dollars?", ["20"]),
    ("What is 391 - 144?", ["247"]),
]

def stream_chat(prompt, n, tools=None, choice=None, timeout=1800):
    body = {"model": "m", "temperature": 0.0, "max_tokens": n, "stream": True,
            "messages": [{"role": "user", "content": prompt}]}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = choice or "auto"
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None
    content = []; tool_calls = {}; finish = None; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            txt = delta.get("content")
            if txt:
                if t_first is None:
                    t_first = time.time()
                content.append(txt); t_last = time.time()
            for tc in (delta.get("tool_calls") or []):
                i = tc.get("index", 0)
                e = tool_calls.setdefault(i, {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    e["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    e["name"] += fn["name"]
                if fn.get("arguments"):
                    e["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    t_end = time.time()
    tokens = (usage or {}).get("completion_tokens")
    if tokens is None:
        tokens = len(content)
    ttft = (t_first - t0) if t_first else None
    dec = None
    if t_first and tokens and tokens > 1:
        dec = (tokens - 1) / max(t_last - t_first, 1e-6)
    merged = [{"id": tool_calls[k]["id"], "type": "function",
               "function": {"name": tool_calls[k]["name"], "arguments": tool_calls[k]["arguments"]}}
              for k in sorted(tool_calls)]
    return {"wall": t_end - t0, "ttft": ttft, "tokens": tokens, "decode": dec,
            "content": "".join(content), "tool_calls": merged, "finish": finish,
            "usage": usage or {}}

def filler_doc(target_chars):
    line = ("Line {i}: the quarterly report records routine warehouse movements, pallet counts "
            "and dock scheduling for the western distribution center.\n")
    chunk = "".join(line.format(i=i) for i in range(60))
    parts = []; total = 0
    while total < target_chars:
        parts.append(chunk); total += len(chunk)
    return "".join(parts)

def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}

def save(d):
    json.dump(d, open(OUT, "w"), indent=1)

def phase_single(cfg):
    res = {"config": cfg, "categories": {}, "depth": {}, "quality": {}, "tools": {}}
    for name, (p, n) in PROMPTS.items():
        r = stream_chat(p, n)
        res["categories"][name] = {"prompt_tokens": r["usage"].get("prompt_tokens"),
                                   "gen": r["tokens"], "ttft": round(r["ttft"] or 0, 3),
                                   "wall": round(r["wall"], 2),
                                   "decode": round(r["decode"] or 0, 1),
                                   "sample": r["content"][:160]}
        print(f"  {name:10s} prompt={res['categories'][name]['prompt_tokens']} gen={r['tokens']} "
              f"ttft={r['ttft']:.2f}s decode={r['decode']:.1f} t/s", flush=True)
    r = stream_chat(TOOL_PROMPT, 120, tools=TOOLS, choice="required")
    args = ""
    name = None
    if r["tool_calls"]:
        name = r["tool_calls"][0].get("function", {}).get("name")
        args = r["tool_calls"][0].get("function", {}).get("arguments", "")
    ok = ('MSFT' in args and '415.25' in args and '250' in args)
    res["tools"] = {"wall": round(r["wall"], 2), "ttft": round(r["ttft"] or 0, 2),
                    "finish": r["finish"], "name": name, "args": args, "correct": ok}
    print(f"  toolcall   wall={r['wall']:.2f}s correct={ok} args={args}", flush=True)
    hits = []
    for q, exp in QUALITY:
        r = stream_chat(q, 250)
        hit = any(e in r["content"] for e in exp)
        hits.append({"q": q, "answer": r["content"][:120], "ok": hit})
        print(f"  Q {q[:46]:46s} -> {'OK' if hit else 'MISS'}", flush=True)
    res["quality"] = {"score": sum(h["ok"] for h in hits), "total": len(hits), "items": hits}
    for label, chars, dec_tokens in (("8K", 32000, 0), ("32K", 128000, 200), ("109K", 440000, 200)):
        doc = filler_doc(chars)
        p = doc + "\n\nQuestion: how many lines mention pallet counts? Answer in one sentence."
        if dec_tokens:
            r = stream_chat(p, dec_tokens)
            entry = {"prompt_tokens": r["usage"].get("prompt_tokens"),
                     "ttft": round(r["ttft"] or 0, 2),
                     "prefill": round(r["usage"].get("prompt_tokens", 0) / max(r["ttft"] or 1e-6, 1e-6), 0),
                     "decode_at_depth": round(r["decode"] or 0, 1), "wall": round(r["wall"], 1)}
        else:
            r = stream_chat(p, 1)
            entry = {"prompt_tokens": r["usage"].get("prompt_tokens"),
                     "ttft": round(r["ttft"] or r["wall"], 2),
                     "prefill": round(r["usage"].get("prompt_tokens", 0) / max((r["ttft"] or r["wall"]), 1e-6), 0)}
        res["depth"][label] = entry
        print(f"  depth {label:5s} prompt={entry['prompt_tokens']} ttft={entry['ttft']}s "
              f"prefill={entry['prefill']:.0f} t/s decode={entry.get('decode_at_depth','-')}", flush=True)
    return res

def phase_concurrency(cfg, levels):
    res = {"config": cfg, "levels": {}}
    p, n = PROMPTS["chat"]
    for k in levels:
        out = []
        def run(i):
            out.append(stream_chat(p, n))
        t0 = time.time()
        ts = [threading.Thread(target=run, args=(i,)) for i in range(k)]
        for t in ts: t.start()
        for t in ts: t.join()
        wall = time.time() - t0
        toks = sum(r["tokens"] for r in out)
        res["levels"][str(k)] = {"streams": k, "wall": round(wall, 2), "tokens": toks,
                                 "aggregate": round(toks / max(wall, 1e-6), 1),
                                 "per_stream": [round(r["tokens"] / r["wall"], 1) for r in out]}
        print(f"  streams={k} wall={wall:.2f}s aggregate={toks/wall:.1f} t/s", flush=True)
    return res

if __name__ == "__main__":
    d = load()
    mode, cfg = sys.argv[1], sys.argv[2]
    if mode == "single":
        d[cfg] = phase_single(cfg)
    elif mode == "quick":
        res = d.get(cfg) or {"config": cfg, "categories": {}, "depth": {}}
        r = stream_chat(TOOL_PROMPT, 120, tools=TOOLS, choice="required")
        args = ""
        name = None
        if r["tool_calls"]:
            name = r["tool_calls"][0].get("function", {}).get("name")
            args = r["tool_calls"][0].get("function", {}).get("arguments", "")
        ok = ('MSFT' in args and '415.25' in args and '250' in args)
        res["tools"] = {"wall": round(r["wall"], 2), "ttft": round(r["ttft"] or 0, 2),
                        "finish": r["finish"], "name": name, "args": args, "correct": ok}
        print(f"  toolcall wall={r['wall']:.2f}s correct={ok} args={args}", flush=True)
        hits = []
        for q, exp in QUALITY:
            r = stream_chat(q, 250)
            hit = any(e in r["content"] for e in exp)
            hits.append({"q": q, "answer": r["content"][:200], "ok": hit})
            print(f"  Q {q[:46]:46s} -> {'OK' if hit else 'MISS'}", flush=True)
        res["quality"] = {"score": sum(h["ok"] for h in hits), "total": len(hits), "items": hits}
        d[cfg] = res
    else:
        d[cfg] = phase_concurrency(cfg, [int(x) for x in sys.argv[3].split(",")])
    save(d)
    print("saved", OUT)
