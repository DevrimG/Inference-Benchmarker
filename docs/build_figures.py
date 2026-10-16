#!/usr/bin/env python3
"""Build the SVG figures used by docs/USER_MANUAL.md.

    python docs/build_figures.py path/to/r1/results.json

Concept diagrams are drawn from fixed shapes. The three data charts are drawn
from the given run (the manual uses the Qwen3-30B-A3B run of 2026-10-16) and
their numbers are also written to docs/images/figure_data.json so the manual
can show them as tables. Colours are the dataviz reference palette (first
three categorical slots, validated all-pairs in light and dark); each SVG
switches to its dark steps with prefers-color-scheme.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent / "images"

STYLE = """<style>
  svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --grid:#e1e0d9; --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
        --band:rgba(42,120,214,0.10); --box:#f3f2ee; --ring:rgba(11,11,11,0.10); }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
          --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926; --s3:#199e70;
          --band:rgba(57,135,229,0.16); --box:#242422; --ring:rgba(255,255,255,0.10); }
  }
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--ink); }
  .t { font-size: 15px; font-weight: 600; }
  .l { font-size: 12.5px; fill: var(--ink2); }
  .m { font-size: 11.5px; fill: var(--muted); font-variant-numeric: tabular-nums; }
  .b { font-size: 12.5px; font-weight: 600; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .box { fill: var(--box); stroke: var(--ring); stroke-width: 1; }
  .arrow { stroke: var(--ink2); stroke-width: 1.5; fill: none; marker-end: url(#ah); }
  .dim { stroke: var(--ink2); stroke-width: 1; fill: none; }
</style>
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
  orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" style="fill:var(--ink2)"/></marker></defs>"""


def svg(width: int, height: int, title: str, desc: str, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-labelledby="title desc">\n'
        f"<title id=\"title\">{escape(title)}</title><desc id=\"desc\">{escape(desc)}</desc>\n"
        f"{STYLE}\n"
        f'<rect width="{width}" height="{height}" rx="8" style="fill:var(--surface)"/>\n'
        + "\n".join(body) + "\n</svg>\n"
    )


def text(x, y, s, cls="l", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{escape(str(s))}</text>'


def bar(x, y, w, h, color, r=4, top=True):
    """Column with a 4px rounded data-end and a square baseline."""
    h = max(h, 0.0)
    if h < r:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" style="fill:var({color})"/>'
    path = (f"M{x:.1f},{y + h:.1f} L{x:.1f},{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
            f"L{x + w - r:.1f},{y:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
            f"L{x + w:.1f},{y + h:.1f} Z")
    return f'<path d="{path}" style="fill:var({color})"/>'


# -- concept diagrams --------------------------------------------------------

def request_timeline() -> str:
    W, H = 760, 350
    x0, x1 = 60, 720
    y = 150
    body = [text(24, 32, "One streamed request, as the tool records it", "t"),
            text(24, 52, "Every network read is stored with its arrival time, bytes and cumulative token count.", "l"),
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="axis"/>']
    send, first, last, end = 60, 250, 650, 690
    chunks = [(250, 1), (290, 3), (330, 2), (370, 3), (410, 1), (540, 3), (580, 2), (620, 3), (650, 2)]
    for cx, tokens in chunks:
        h = 14 * tokens
        body.append(bar(cx - 5, y - h, 10, h, "--s1", r=3))
    for xx, label in [(send, "request sent"), (end, "stream ends")]:
        body.append(f'<line x1="{xx}" y1="{y - 60}" x2="{xx}" y2="{y + 6}" class="dim"/>')
        body.append(text(xx, y - 66, label, "m", "middle"))

    def span(a, b, yy, label, cls="b"):
        body.append(f'<line x1="{a}" y1="{yy}" x2="{b}" y2="{yy}" class="dim" marker-start="url(#ah)" marker-end="url(#ah)"/>')
        body.append(text((a + b) / 2, yy - 6, label, cls, "middle"))

    span(send, first, y + 34, "TTFT")
    span(first, last, y + 34, "decode window: decode tok/s = tokens after first read ÷ time")
    span(410, 540, y + 76, "max stall = largest gap between reads")
    span(290, 330, y + 76, "chunk gap", "m")
    span(send, end, y + 118, "end-to-end latency", "m")
    body.append(text(24, H - 34, "Each bar is one network read; its height is the number of tokens that read carried.", "l"))
    body.append(text(24, H - 14, "Per-token ITL = chunk gap ÷ tokens it carried.  TPOT = decode window ÷ tokens after the first read.", "l"))
    return svg(W, H, "Request timeline", "TTFT, chunk gaps, stall, decode window and latency on one streamed request.", body)


def arrival_modes() -> str:
    W, H = 760, 400
    body = [text(24, 32, "Arrival modes: when each request is sent", "t"),
            text(24, 52, "Each bar is one request, from send to last token.", "l")]
    rows = [
        ("closed", "C workers; each sends its next|request when the previous ends",
         [(0, 0, 120), (0, 125, 230), (0, 235, 330), (1, 0, 90), (1, 95, 210), (1, 215, 340), (2, 0, 140), (2, 145, 260), (2, 265, 350)]),
        ("poisson / gamma", "sent at precomputed random times;|overlap grows if the server slows",
         [(0, 10, 120), (1, 40, 170), (0, 130, 210), (2, 150, 300), (1, 190, 280), (0, 230, 350), (1, 300, 390)]),
        ("burst", "exactly C requests released|at the same instant",
         [(0, 0, 150), (1, 0, 170), (2, 0, 160)]),
        ("replay", "sent at the offsets in a JSONL|trace (timestamp ms or offset_s)",
         [(0, 0, 80), (0, 100, 190), (1, 105, 230), (2, 110, 200), (0, 300, 380)]),
    ]
    top = 76
    for index, (name, note, bars) in enumerate(rows):
        oy = top + index * 80
        body.append(text(24, oy + 16, name, "b"))
        for k, line in enumerate(note.split("|")):
            body.append(text(24, oy + 34 + 15 * k, line, "m"))
        body.append(f'<line x1="330" y1="{oy + 62}" x2="740" y2="{oy + 62}" class="axis"/>')
        for lane, a, b in bars:
            yy = oy + 6 + lane * 18
            body.append(f'<rect x="{330 + a}" y="{yy}" width="{max(4, b - a - 2)}" height="12" rx="3" style="fill:var(--s1)"/>')
            body.append(f'<line x1="{330 + a}" y1="{yy - 2}" x2="{330 + a}" y2="{yy + 14}" style="stroke:var(--ink);stroke-width:1.5"/>')
    body.append(text(330, H - 10, "time →   (tick = send time)", "m"))
    return svg(W, H, "Arrival modes", "Closed loop, open-loop Poisson or gamma, burst and replay schedules.", body)


def fanout() -> str:
    W, H = 760, 340
    body = [text(24, 32, "Multi-endpoint fan-out", "t"),
            text(24, 52, "One client process, one start instant, one merged result with an endpoint column.", "l")]

    def box(x, y, w, h, title, sub=""):
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="box"/>')
        body.append(text(x + w / 2, y + 22, title, "b", "middle"))
        if sub:
            body.append(text(x + w / 2, y + 40, sub, "m", "middle"))

    box(24, 130, 150, 60, "lab.toml", "concurrency 8, weights 2:1")
    box(214, 130, 150, 60, "StartGate", "warmup, then start together")
    box(414, 80, 160, 60, "endpoint A", "C = 5 (weight 2)")
    box(414, 200, 160, 60, "endpoint B", "C = 3 (weight 1)")
    box(614, 130, 122, 60, "merged result", "endpoint column")
    for (a, b) in [((174, 160), (214, 160)), ((364, 150), (414, 110)), ((364, 170), (414, 230)),
                   ((574, 110), (614, 150)), ((574, 230), (614, 170))]:
        body.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0] - 2}" y2="{b[1]}" class="arrow"/>')
    body.append(text(494, 156, "↕ /metrics scraped separately", "m", "middle"))
    body.append(text(494, 276, "↕ /metrics scraped separately", "m", "middle"))
    body.append(text(24, H - 34, "Concurrency is split by weight (largest remainder). Open-loop rates are split the same way;", "l"))
    body.append(text(24, H - 16, "replay traces are dealt round-robin. Server agreement is reported per endpoint.", "l"))
    return svg(W, H, "Fan-out", "Config splits concurrency across endpoints that share a start gate and merge results.", body)


def workflow() -> str:
    W, H = 760, 150
    steps = [("1  install", "requirements.txt"), ("2  configure", "lab.example.toml"),
             ("3  dry run", "--dry-run"), ("4  run", "--config lab.toml"),
             ("5  read", "lab_summary.json")]
    body = [text(24, 30, "Workflow", "t")]
    w, gap, y = 132, 15, 56
    for i, (title, sub) in enumerate(steps):
        x = 24 + i * (w + gap)
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="62" rx="6" class="box"/>')
        body.append(text(x + 10, y + 24, title, "b"))
        body.append(text(x + 10, y + 44, sub, "m"))
        if i < len(steps) - 1:
            body.append(f'<line x1="{x + w + 1}" y1="{y + 31}" x2="{x + w + gap - 1}" y2="{y + 31}" class="arrow"/>')
    return svg(W, H, "Workflow", "Install, configure, dry run, run, read outputs.", body)


# -- data charts --------------------------------------------------------------

def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    raws = data["raw_metrics"]
    origin = min(m["start_time"] for m in raws)
    intervals = [(m["start_time"] - origin, m["end_time"] - origin) for m in raws]
    return {"data": data, "raws": raws, "origin": origin, "intervals": intervals}


def axes(body, x0, y0, x1, y1, xmax, ymax, xticks, yticks, xlabel, ylabel, fmt=lambda v: f"{v:g}"):
    sx = lambda v: x0 + (x1 - x0) * v / xmax
    sy = lambda v: y1 - (y1 - y0) * v / ymax
    for v in yticks:
        body.append(f'<line x1="{x0}" y1="{sy(v):.1f}" x2="{x1}" y2="{sy(v):.1f}" class="grid"/>')
        body.append(text(x0 - 8, sy(v) + 4, fmt(v), "m", "end"))
    for v in xticks:
        body.append(text(sx(v), y1 + 18, fmt(v), "m", "middle"))
    body.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>')
    body.append(text((x0 + x1) / 2, y1 + 38, xlabel, "m", "middle"))
    body.append(text(x0 - 40, y0 - 12, ylabel, "m"))
    return sx, sy


def steady_window_chart(run: dict, summary: dict) -> tuple[str, dict]:
    intervals = run["intervals"]
    events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals], key=lambda e: (e[0], e[1]))
    steps, level = [(0.0, 0)], 0
    for t, d in events:
        level += d
        steps.append((t, level))
    end = max(b for _, b in intervals)
    steady = summary["steady_state"]
    s0, s1 = steady["start_seconds"], steady["end_seconds"]
    W, H = 760, 330
    x0, y0, x1, y1 = 70, 80, 730, 270
    body = [text(24, 32, "Requests in flight, Qwen3-30B-A3B run r1 (C = 4)", "t"),
            text(24, 52, "Steady-state throughput counts only tokens that arrive while all 4 slots are busy.", "l")]
    xmax = math.ceil(end / 5) * 5
    sx, sy = axes(body, x0, y0, x1, y1, xmax, 5, range(0, int(xmax) + 1, 5), range(0, 6),
                  "seconds since start", "in flight")
    body.append(f'<rect x="{sx(s0):.1f}" y="{y0}" width="{sx(s1) - sx(s0):.1f}" height="{y1 - y0}" style="fill:var(--band)"/>')
    body.append(text((sx(s0) + sx(s1)) / 2, y0 + 16, f"steady window {s1 - s0:.1f} s · {steady['output_tokens_per_second']:.0f} tok/s", "b", "middle"))
    path = []
    for i, (t, lv) in enumerate(steps):
        if i:
            path.append(f"L{sx(t):.1f},{sy(steps[i - 1][1]):.1f}")
        path.append(f"{'M' if not i else 'L'}{sx(t):.1f},{sy(lv):.1f}")
    body.append(f'<path d="{" ".join(path)}" style="fill:none;stroke:var(--s1);stroke-width:2"/>')
    full = summary["full_window"]
    body.append(text(sx(end) - 4, y1 - 8, f"full window {full['duration_seconds']:.1f} s · {full['output_tokens_per_second']:.0f} tok/s", "m", "end"))
    data = {"steady_window_s": [s0, s1], "steady_tok_s": steady["output_tokens_per_second"],
            "full_window_s": full["duration_seconds"], "full_tok_s": full["output_tokens_per_second"]}
    return svg(W, H, "In-flight timeline", "Step line of requests in flight with the steady window shaded.", body), data


def token_timeline_chart(run: dict) -> tuple[str, dict]:
    raws = sorted(run["raws"], key=lambda m: m["max_stall_seconds"] or 0)
    picks = [("smoothest", raws[0]), ("median", raws[len(raws) // 2]), ("largest stall", raws[-1])]
    colors = ["--s1", "--s2", "--s3"]
    W, H = 760, 360
    x0, y0, x1, y1 = 70, 80, 560, 300
    body = [text(24, 32, "Cumulative tokens per request (server's running usage count)", "t"),
            text(24, 52, "Slope = per-user decode rate. Flat steps are stalls between reads.", "l")]
    series, tmax = [], 0.0
    for label, m in picks:
        pts = [(c[0], c[4]) for c in m["chunk_records"] if c[3] and c[4] is not None]
        series.append((label, m, pts))
        tmax = max(tmax, pts[-1][0])
    xmax = math.ceil(tmax)
    sx, sy = axes(body, x0, y0, x1, y1, xmax, 260, range(0, xmax + 1, 1), range(0, 261, 50),
                  "seconds since the request was sent", "tokens")
    data = []
    for (label, m, pts), color in zip(series, colors):
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(t):.1f},{sy(c):.1f}" for i, (t, c) in enumerate(pts))
        body.append(f'<path d="{d}" style="fill:none;stroke:var({color});stroke-width:2;stroke-linejoin:round"/>')
        lt, lc = pts[-1]
        body.append(f'<circle cx="{sx(lt):.1f}" cy="{sy(lc):.1f}" r="4" style="fill:var({color});stroke:var(--surface);stroke-width:2"/>')
        data.append({"label": label, "ttft_s": round(m["ttft_seconds"], 3),
                     "decode_tok_s": round(m["decode_tokens_per_second"], 1),
                     "max_stall_s": round(m["max_stall_seconds"], 3), "end_s": round(lt, 2)})
    # Direct labels on the right, spaced so they never collide.
    ends = sorted(((sy(p[-1][1]), sx(p[-1][0]), lab, col) for (lab, _, p), col in zip(series, colors)))
    ys = []
    for yy, _, _, _ in ends:
        ys.append(max(yy, (ys[-1] + 30) if ys else yy))
    for (yy, xx, lab, col), ly in zip(ends, ys):
        row = next(r for r in data if r["label"] == lab)
        body.append(f'<line x1="{xx + 6:.1f}" y1="{yy:.1f}" x2="{x1 + 6}" y2="{ly:.1f}" class="dim"/>')
        body.append(f'<rect x="{x1 + 10}" y="{ly - 9}" width="10" height="10" rx="2" style="fill:var({col})"/>')
        body.append(text(x1 + 26, ly, lab, "b"))
        body.append(text(x1 + 26, ly + 15, f"{row['decode_tok_s']:.0f} tok/s · stall {row['max_stall_s']:.2f} s", "m"))
    return svg(W, H, "Token timelines", "Three requests' cumulative tokens over time.", body), {"requests": data}


def itl_bucket_chart(run: dict) -> tuple[str, dict]:
    buckets = run["data"]["server_metrics"]["summary"]["histograms"]["itl"]["buckets"]
    bounds = [b["le"] for b in buckets]
    cumulative = [b["count_delta"] for b in buckets]
    server = [c - (cumulative[i - 1] if i else 0) for i, c in enumerate(cumulative)]
    gaps = [b[0] - a[0] for m in run["raws"] if m["error"] == "none"
            for a, b in zip([c for c in m["chunk_records"] if c[3]], [c for c in m["chunk_records"] if c[3]][1:])]
    client = []
    lower = 0.0
    for bound in bounds:
        hi = math.inf if bound in ("+Inf", "Inf") else float(bound)
        client.append(sum(1 for g in gaps if lower < g <= hi))
        lower = hi
    keep = [i for i, b in enumerate(bounds) if isinstance(b, (int, float)) and b <= 1.0]
    labels = [f"{bounds[i] * 1000:g}" for i in keep]
    tail_client = sum(client[i] for i in range(len(bounds)) if i not in keep)
    tail_server = sum(server[i] for i in range(len(bounds)) if i not in keep)
    labels.append(">1000")
    cvals = [client[i] for i in keep] + [tail_client]
    svals = [server[i] for i in keep] + [tail_server]
    ctot, stot = sum(cvals), sum(svals)
    cpct = [100 * v / ctot for v in cvals]
    spct = [100 * v / stot for v in svals]
    W, H = 760, 360
    x0, y0, x1, y1 = 70, 96, 730, 290
    body = [text(24, 32, "Inter-token gaps by vLLM histogram bucket (log share)", "t"),
            text(24, 52, "Client gaps between reads vs the server's own ITL histogram, same run.", "l")]
    body.append(f'<rect x="{x1 - 230}" y="64" width="10" height="10" rx="2" style="fill:var(--s1)"/>')
    body.append(text(x1 - 214, 73, "client (this tool)", "l"))
    body.append(f'<rect x="{x1 - 100}" y="64" width="10" height="10" rx="2" style="fill:var(--s2)"/>')
    body.append(text(x1 - 84, 73, "server /metrics", "l"))
    # Log scale on share so the tail is visible: 0.01 % .. 100 %.
    lo, hi = -2, 2
    sy = lambda p: y1 - (y1 - y0) * (math.log10(max(p, 10 ** lo)) - lo) / (hi - lo)
    for p in (0.01, 0.1, 1, 10, 100):
        body.append(f'<line x1="{x0}" y1="{sy(p):.1f}" x2="{x1}" y2="{sy(p):.1f}" class="grid"/>')
        body.append(text(x0 - 8, sy(p) + 4, f"{p:g} %", "m", "end"))
    body.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" class="axis"/>')
    band = (x1 - x0) / len(labels)
    for i, label in enumerate(labels):
        cx = x0 + band * i + band / 2
        for j, (vals, color) in enumerate(((cpct, "--s1"), (spct, "--s2"))):
            if vals[i] > 0:
                bx = cx - 13 + j * 14
                body.append(bar(bx, sy(vals[i]), 12, y1 - sy(vals[i]), color, r=3))
        body.append(text(cx, y1 + 16, label, "m", "middle"))
    body.append(text((x0 + x1) / 2, y1 + 36, "bucket upper bound (ms): a gap of 18 ms falls in the 25 bucket", "m", "middle"))
    body.append(text(24, H - 12, f"Client gaps: {ctot}; server samples: {int(stot)}. Gaps over 50 ms appear on the client but rarely on the server.", "l"))
    rows = [{"bucket_le_ms": l, "client_percent": round(c, 3), "server_percent": round(s, 3)}
            for l, c, s in zip(labels, cpct, spct)]
    return svg(W, H, "ITL buckets", "Grouped bars of client and server inter-token gap shares per bucket.", body), {"buckets": rows}


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in [("workflow", workflow), ("request-timeline", request_timeline),
                        ("arrival-modes", arrival_modes), ("fanout", fanout)]:
        (OUT / f"{name}.svg").write_text(build())
    figure_data: dict = {}
    if argv:
        path = Path(argv[0])
        run = load(path)
        summary = run["data"]["summary"]
        for name, build in [("steady-window", lambda: steady_window_chart(run, summary)),
                            ("token-timeline", lambda: token_timeline_chart(run)),
                            ("itl-buckets", lambda: itl_bucket_chart(run))]:
            content, data = build()
            (OUT / f"{name}.svg").write_text(content)
            figure_data[name] = data
        figure_data["source"] = {"run": str(path), "model": summary["model"]}
        (OUT / "figure_data.json").write_text(json.dumps(figure_data, indent=2))
    print("wrote", sorted(p.name for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
