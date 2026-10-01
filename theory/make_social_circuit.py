"""Minimal first-arrival tree with binary reach states from A4."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parent
COLOURS = {'Germany': '#009E73'}
INK = '#2B2F36'

MUTED = '#69747E'


def draw(data):
    width, height = 5.5, 2.15
    fig = plt.figure(figsize=(width, height), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height), aspect='equal')
    ax.axis('off')
    radius = .032

    def xy(column, agent):
        return .40 + .48 * column, 1.80 - (1.25 / 7) * agent

    def connection(e):
        if not e['within_round_relay']:
            return 'arc3,rad=0'
        start, end = xy(*e['source']), xy(*e['target'])
        # Arc3 uses control_x = midpoint_x + rad * (end_y-start_y).
        # The signed radius therefore gives the same rightward control-point
        # displacement (one-third column spacing) in both travel directions.
        return f"arc3,rad={(.48/3)/(end[1]-start[1])}"

    highlight = FancyBboxPatch((xy(4, 0)[0]-.15, .455), .30, 1.63,
                               boxstyle='round,pad=0,rounding_size=.055',
                               facecolor='#EDF8F3', edgecolor='none', zorder=0)
    highlight.set_gid('completion-highlight-R4')
    ax.add_patch(highlight)

    for e in data['contacts']:
        if not e['tree']:
            continue
        edge = FancyArrowPatch(xy(*e['source']), xy(*e['target']),
                               arrowstyle='-|>', mutation_scale=5.2,
                               color=COLOURS['Germany'], lw=1.4, alpha=1,
                               shrinkA=radius*72+.4, shrinkB=radius*72+.65,
                               connectionstyle=connection(e), zorder=2)
        edge.set_gid(f'tree-contact-{e["interaction"]}')
        if e['within_round_relay']:
            # Separate path crossings visually without adding false junctions.
            edge.set_zorder(2.2)
            edge.set_path_effects([pe.Stroke(linewidth=2.4, foreground='white'), pe.Normal()])
        ax.add_patch(edge)

    for n in data['nodes']:
        circle = Circle(xy(n['round'], n['agent']), radius,
                        facecolor=COLOURS['Germany'] if n['reached'] else 'white',
                        edgecolor='white' if n['reached'] else MUTED,
                        lw=.28 if n['reached'] else .65, zorder=3)
        circle.set_gid(f'node-A{n["agent"]}-R{n["round"]}')
        ax.add_patch(circle)
    for r in range(11):
        label = 'init' if r == 0 else 'final' if r == 10 else f'R{r}'
        ax.text(xy(r, 0)[0], 1.985, label, fontsize=9.5,
                color=COLOURS['Germany'] if r == 4 else MUTED,
                weight='bold' if r == 4 else 'normal', ha='center', va='center')
    for agent in range(8):
        ax.text(.24, xy(0, agent)[1], f'A{agent}', ha='right', va='center',
                fontsize=9.5, color=COLOURS['Germany'] if agent == 4 else INK,
                weight='bold' if agent == 4 else 'normal')
    ax.text(.40, .105, f"Mean earliest arrival: {data['mean_arrival_rounds']:.2f} rounds (7 peers)",
            fontsize=8, color=MUTED, va="center")

    handles = [
        Line2D([], [], ls='none', marker='o', markersize=3.5,
               markerfacecolor='white', markeredgecolor=MUTED, markeredgewidth=.65,
               label='not yet reached'),
        Line2D([], [], ls='none', marker='o', markersize=3.5,
               markerfacecolor=COLOURS['Germany'], markeredgecolor='none',
               label='reached from A4'),
    ]
    legend = fig.legend(handles=handles, loc='center', bbox_to_anchor=(.5, .31 / height),
                        ncol=2, frameon=False, fontsize=8.5, labelcolor=MUTED,
                        handlelength=.9, handletextpad=.35, columnspacing=.6, borderpad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in [*ax.texts, legend]:
        box = item.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        assert 0 <= box.x0 <= box.x1 <= 1 and 0 <= box.y0 <= box.y1 <= 1, str(item)
    assert len([p for p in ax.patches if (p.get_gid() or '').startswith('node-')]) == 88
    assert len([p for p in ax.patches if (p.get_gid() or '').startswith('tree-contact-')]) == 7
    assert len([p for p in [*ax.lines, *ax.patches]
                if (p.get_gid() or '').startswith('contact-')]) == 0
    assert not any(p.get_gid() == 'source-ring-A4' for p in ax.patches)
    return fig


def calculate(schedule):
    n, source = schedule['N'], schedule['source_agent']
    arrivals, parents = {source: 0}, {}
    events = schedule['events']
    if [e['t'] for e in events] != list(range(1, n * 10 + 1)):
        raise ValueError('Expected the complete ordered ten-round schedule')
    for event in events:
        sender, recipient = event['speaker_id'], event['listener_id']
        if not (0 <= sender < n and 0 <= recipient < n) or sender == recipient:
            raise ValueError('Invalid schedule contact')
        if sender in arrivals and recipient not in arrivals:
            arrivals[recipient] = event['t']
            parents[recipient] = event
    if len(arrivals) != n:
        raise ValueError('Some peers are unreached; an all-peer mean is undefined')
    contacts = []
    for e in parents.values():
        t = e['t']; round_ = (t - 1) // n + 1
        sender = e['speaker_id']; arrival = arrivals[sender]
        relay = 0 < arrival < t and (arrival - 1) // n + 1 == round_
        contacts.append({'interaction': t, 'tree': True,
                         'source': [round_ if relay else round_ - 1, sender],
                         'target': [round_, e['listener_id']], 'within_round_relay': relay})
    mean = sum(t for a, t in arrivals.items() if a != source) / (n - 1)
    return {'contacts': contacts,
            'nodes': [{'round': round_, 'agent': a, 'reached': arrivals[a] <= round_ * n}
                      for round_ in range(11) for a in range(n)],
            'arrivals_interactions': arrivals,
            'mean_arrival_interactions': mean, 'mean_arrival_rounds': mean / n,
            'completion_interaction': max(arrivals.values()),
            'source_agent': source, 'N': n, 'source_excluded_from_mean': True}


def main():
    schedule_path = ROOT / 'data/social_circuit_schedule.json'
    data = calculate(json.loads(schedule_path.read_text()))
    expected = {4: 0, 0: 1, 5: 4, 3: 9, 7: 15, 2: 17, 1: 18, 6: 26}
    assert data['arrivals_interactions'] == expected
    assert abs(data['mean_arrival_rounds'] - 90 / 56) < 1e-12
    assert [sum(n['reached'] for n in data['nodes'] if n['round'] == r)
            for r in range(11)] == [1, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8]
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9.5, 'text.color': INK,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = draw(data)
    output = ROOT.parent / 'paper/figures/final'; output.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(output / f'fig6.{ext}', dpi=500, facecolor='white')
    plt.close(fig)
    validation = ROOT / 'validation'; validation.mkdir(exist_ok=True)
    data['schedule_sha256'] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    (validation / 'social_circuit.json').write_text(json.dumps(data, indent=2) + '\n')
    print(f"A4 mean earliest arrival: {data['mean_arrival_rounds']:.8f} rounds; "
          f"all peers reached by interaction {data['completion_interaction']}.")


if __name__ == '__main__':
    main()
