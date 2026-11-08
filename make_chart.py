"""Make the results chart (inline SVG) and table (HTML) from results/summary.json."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
S = json.loads((ROOT / "results/summary.json").read_text())

W, H = 640, 300
L, R, T, B = 56, 120, 20, 56  # margins
pw, ph = W - L - R, H - T - B
drops = [d["drop"] for d in S["drops"]]


def x(d): return L + pw * d
def y(v): return T + ph * (1 - v)


SERIES = [
    ("accuracy", f"All {S['n_examples']} questions", "s1"),
    ("accuracy_yes_subset", f"Answer is Yes ({S['drops'][0]['n_yes_subset']})", "s2"),
]

o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="res-title res-desc">',
     '<title id="res-title">POPE accuracy as image tokens are removed</title>',
     '<desc id="res-desc">' + "; ".join(
         f"{int(d['drop']*100)}% removed: all {d['accuracy']:.1%}, yes-subset {d['accuracy_yes_subset']:.1%}"
         for d in S["drops"]) + "</desc>"]
for v in (0, .25, .5, .75, 1):
    o.append(f'<line class="grid" x1="{L}" x2="{L+pw}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
    o.append(f'<text class="tick" x="{L-8}" y="{y(v)+4:.1f}" text-anchor="end">{int(v*100)}%</text>')
o.append(f'<line class="chance" x1="{L}" x2="{L+pw}" y1="{y(.5):.1f}" y2="{y(.5):.1f}"/>')
o.append(f'<text class="tick" x="{L+4}" y="{y(.5)-6:.1f}" text-anchor="start">guess</text>')
for d in S["drops"]:
    o.append(f'<text class="tick" x="{x(d["drop"]):.1f}" y="{T+ph+18}" text-anchor="middle">{int(d["drop"]*100)}%</text>')
    o.append(f'<text class="tick sub" x="{x(d["drop"]):.1f}" y="{T+ph+32}" text-anchor="middle">{d["mean_kept_tokens"]:g} kept</text>')
o.append(f'<text class="axis" x="{L+pw/2}" y="{H-6}" text-anchor="middle">Image tokens removed</text>')
for key, name, cls in SERIES:
    pts = [(x(d["drop"]), y(d[key])) for d in S["drops"]]
    o.append(f'<polyline class="line {cls}" points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
    for (a, b), d in zip(pts, S["drops"]):
        o.append(f'<g class="pt"><circle class="dot {cls}" cx="{a:.1f}" cy="{b:.1f}" r="4.5"/>'
                 f'<circle class="hit" cx="{a:.1f}" cy="{b:.1f}" r="14"/>'
                 f'<title>{name}, {int(d["drop"]*100)}% removed: {d[key]:.1%}</title></g>')
    last = S["drops"][-1][key]
    o.append(f'<text class="direct" x="{pts[-1][0]+10:.1f}" y="{pts[-1][1]+4:.1f}">{last:.0%}</text>')
o.append("</svg>")
svg = "\n".join(o)

legend = "".join(f'<span class="key"><i class="sw {cls}"></i>{name}</span>' for _, name, cls in SERIES)
rows = "".join(
    f"<tr><td>{int(d['drop']*100)}%</td><td>{d['mean_kept_tokens']:g}</td><td>{d['accuracy']:.1%}</td>"
    f"<td>{d['accuracy_yes_subset']:.1%}</td><td>{d['yes_rate']:.1%}</td><td>{d['unparsed']}</td></tr>"
    for d in S["drops"])
table = ("<table><thead><tr><th>Removed</th><th>Image tokens kept (mean)</th><th>Accuracy, all</th>"
         "<th>Accuracy, answer is Yes</th><th>Model says Yes</th><th>Not yes/no</th></tr></thead>"
         f"<tbody>{rows}</tbody></table>")
(ROOT / "results/chart.html").write_text(f'<div class="legend">{legend}</div>\n{svg}\n{table}\n')
print("wrote results/chart.html")
