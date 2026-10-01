#!/usr/bin/env python3
"""Reproduce the accepted four-panel theory figure from packaged inputs.

The theoretical panels use the exact discrete model evaluated at populations
2, 4, 8, 16, 32, 64, and 128. The default empirical panel uses slightly darker
0.05 soft regions, with enlarged typography and no dots or legend. Optional variants use the same data and
layout: --bandwidth 0.035 or --dots. Add --theory-background with --dots
to recover the empirical points over the theoretical regions. No model
calls or external data access.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'flag-haze-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir()) / 'flag-four-panel-cache'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D

import style

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
NAMES = ['correct_consensus', 'wrong_consensus', 'polarization', 'fragmentation']
LABELS = ['Correct consensus', 'Wrong consensus', 'Polarization', 'Fragmentation']
THEORY_NAMES = NAMES[:3]
COLORS = ['#15935B', '#347AB8', '#DC782B', '#858990']
PALE = ['#CDEADF', '#DCE8F5', '#FBE0D4', '#D9D9D9']
DOT_COLORS = dict(zip(NAMES, ['#15935B', '#347AB8', '#DC782B', '#9A9DA3']))
SHARE_AXIS_LABEL = r'Rival evidence share $\frac{a_R}{a_T+a_R}$'
TEXT_SCALE = 1.375
EMPIRICAL_MAX_OPACITY = .46


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_inputs():
    """Validate literal response fractions and load the compact final cohort."""
    rows = read_csv(DATA / 'empirical_runs.csv')
    crops = read_csv(DATA / 'yemen_crops.csv')
    example = json.loads((DATA / 'yemen_example.json').read_text())
    for row in rows:
        for key in ['N', 'truth_count', 'rival_count', 'valid_count']:
            row[key] = int(row[key])
        for key in ['rival_response_share', 'display_N', 'log2_N_offset']:
            row[key] = float(row[key])
        assert row['truth_count'] + row['rival_count'] == row['valid_count'] == 50 * row['N']
        assert row['rival_response_share'] == row['rival_count'] / row['valid_count']
        assert row['outcome'] in NAMES
    for row in crops:
        for key in ['agent_id', 'truth_count', 'rival_count', 'B_valid', 'left', 'top', 'width', 'height']:
            row[key] = int(row[key])
        for key in ['center_x', 'center_y', 'rival_response_share']:
            row[key] = float(row[key])
        assert row['truth_count'] + row['rival_count'] == row['B_valid'] == 50
        assert row['rival_response_share'] == row['rival_count'] / row['B_valid']
        assert row['center_x'] == row['left'] + row['width'] / 2
        assert row['center_y'] == row['top'] + row['height'] / 2
    assert len(rows) == len({row['run_id'] for row in rows}) == 217
    assert Counter(row['outcome'] for row in rows) == dict(zip(NAMES, [101, 15, 93, 8]))
    assert len(crops) == len({row['agent_id'] for row in crops}) == 128
    assert sum(row['rival_count'] for row in crops) == example['rival_count'] == 4192
    assert sum(row['B_valid'] for row in crops) == example['valid_count'] == 6400
    assert example['rival_response_share'] == 4192 / 6400 == .655
    with np.load(DATA / 'model_probabilities.npz') as cache:
        populations = cache['N'].copy()
        shares = cache['share'].copy()
        probabilities = cache['probabilities'].copy()
        reference_share = float(cache['reference_share'])
        assert str(cache['method']) == 'exact_binary_birth_death'
    assert np.array_equal(populations, [2, 4, 8, 16, 32, 64, 128])
    assert probabilities.shape == (len(THEORY_NAMES), len(populations), len(shares))
    assert np.all(probabilities >= 0)
    assert np.max(np.abs(probabilities.sum(axis=0) - 1)) < 1e-11
    assert reference_share == .35
    index = int(np.argmin(np.abs(shares - reference_share)))
    assert np.isclose(shares[index], reference_share, atol=1e-15, rtol=0)
    return rows, crops, example, populations, shares, probabilities, probabilities[:, :, index]


def blur_axis(values, sigma, axis):
    radius = int(np.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-.5 * (offsets / sigma)**2)
    kernel /= kernel.sum()

    def smooth(row):
        return np.convolve(np.pad(row, radius, mode='edge'), kernel, mode='valid')

    return np.apply_along_axis(smooth, axis, values)


def pchip_last_axis(x, values, query):
    """Shape-preserving cubic Hermite interpolation along the last axis.

    Interior derivatives use the weighted harmonic mean of adjacent secants
    when their signs agree, and zero otherwise. One-sided endpoint derivatives
    are limited to preserve the adjacent interval's shape. This interpolates
    tabulated values only; it does not evaluate dynamics at fractional N.
    """
    x, query = np.asarray(x, dtype=float), np.asarray(query, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(x) < 3 or values.shape[-1] != len(x) or np.any(np.diff(x) <= 0):
        raise ValueError('PCHIP requires at least three increasing anchor coordinates.')
    if np.any(query < x[0]) or np.any(query > x[-1]):
        raise ValueError('Theory interpolation is restricted to the evaluated population range.')
    steps = np.diff(x)
    secants = np.diff(values, axis=-1) / steps
    slopes = np.zeros_like(values)
    left, right = secants[..., :-1], secants[..., 1:]
    same_direction = ((left > 0) & (right > 0)) | ((left < 0) & (right < 0))
    w1, w2 = 2 * steps[1:] + steps[:-1], steps[1:] + 2 * steps[:-1]
    safe_left, safe_right = np.where(same_direction, left, 1), np.where(same_direction, right, 1)
    with np.errstate(over='ignore'):
        slopes[..., 1:-1] = np.where(same_direction,
            (w1 + w2) / (w1 / safe_left + w2 / safe_right), 0)

    def endpoint(first_step, second_step, first_secant, second_secant):
        slope = ((2 * first_step + second_step) * first_secant - first_step * second_secant) / (first_step + second_step)
        slope = np.where(np.sign(slope) != np.sign(first_secant), 0, slope)
        limit = (np.sign(first_secant) != np.sign(second_secant)) & (np.abs(slope) > 3 * np.abs(first_secant))
        return np.where(limit, 3 * first_secant, slope)

    slopes[..., 0] = endpoint(steps[0], steps[1], secants[..., 0], secants[..., 1])
    slopes[..., -1] = endpoint(steps[-1], steps[-2], secants[..., -1], secants[..., -2])
    interval = np.clip(np.searchsorted(x, query, side='right') - 1, 0, len(x) - 2)
    width = steps[interval]
    t = (query - x[interval]) / width
    return ((2 * t**3 - 3 * t**2 + 1) * values[..., interval]
            + (t**3 - 2 * t**2 + t) * width * slopes[..., interval]
            + (-2 * t**3 + 3 * t**2) * values[..., interval + 1]
            + (t**3 - t**2) * width * slopes[..., interval + 1])


def interpolate_theory(populations, probabilities, n_fine):
    """Connect the seven exact discrete calculations while retaining their values.

    Input order is (outcome, N, share); output is (outcome, share, display N).
    PCHIP is applied separately to each outcome. Clipping numerical excursions
    and normalizing across outcomes preserve nonnegative unit-sum vectors.
    The source grid evaluates the exact discrete model at N=2,4,8,16,32,64,128.
    This interpolation only draws connectors between those calculations;
    intermediate displayed values are not additional model evaluations.
    """
    anchor_values = np.moveaxis(probabilities, 1, -1)
    raw = pchip_last_axis(np.log2(populations), anchor_values, np.log2(n_fine))
    smooth = np.clip(raw, 0, 1)
    smooth /= smooth.sum(axis=0)
    anchor_columns = []
    for source, n in enumerate(populations):
        columns = np.flatnonzero(n_fine == n)
        if len(columns) != 1:
            raise ValueError('The display grid must include each exact population size once.')
        column = int(columns[0])
        smooth[..., column] = probabilities[:, source, :]
        anchor_columns.append(column)
    restored = np.moveaxis(smooth[..., anchor_columns], -1, 1)
    assert np.array_equal(restored, probabilities)
    # Computed source columns retain their floating-point summation roundoff.
    assert np.isfinite(smooth).all() and np.all((smooth >= 0) & (smooth <= 1 + 1e-11))
    maximum_sum_error = float(np.abs(smooth.sum(axis=0) - 1).max())
    assert maximum_sum_error < 1e-11
    diagnostics = dict(method='Shape-preserving cubic Hermite interpolation (PCHIP) of outcome probabilities in log2 N, then probability normalization',
        source_calculation='Exact discrete binary stationary/fixation probabilities',
        evaluated_integer_population_sizes=populations.tolist(), display_population_count=len(n_fine),
        interpolation_domain='Between the seven evaluated populations N=2,4,8,16,32,64,128',
        source_anchor_values_preserved_exactly=True, minimum_probability=float(smooth.min()),
        maximum_probability=float(smooth.max()), maximum_probability_sum_error=maximum_sum_error,
        raw_interpolation_minimum=float(raw.min()), raw_interpolation_maximum=float(raw.max()),
        fractional_display_populations_are_model_evaluations=False,
        source_values_are_exact_binary_endpoint_probabilities=True, parameters_fitted=False)
    return smooth, diagnostics


def soft_fields(rows, share_bandwidth=.05):
    """Estimate empirical frequencies and apply the accepted display taper.

    Runs are weighted equally before Gaussian distance weighting within each
    observed N. Smoothing and fade widths are display choices, not fitted
    physical parameters or confidence intervals. No theory enters this map.
    """
    if not np.isfinite(share_bandwidth) or share_bandwidth <= 0:
        raise ValueError('The evidence-share bandwidth must be positive and finite.')
    populations = np.array(sorted({row['N'] for row in rows}))
    shares = np.linspace(0, 1, 501)
    log_n = np.linspace(1, 7, 601)
    columns, support = [], []
    for n in populations:
        group = [row for row in rows if row['N'] == n]
        evidence = np.array([row['rival_response_share'] for row in group])
        log_weights = -.5 * ((shares[:, None] - evidence[None, :]) / share_bandwidth)**2
        weights = np.exp(log_weights)
        # Preserve the accepted arithmetic; stabilize only extreme user widths
        # whose weights would all underflow outside the sampled share range.
        empty = weights.sum(axis=1) == 0
        if np.any(empty):
            weights[empty] = np.exp(log_weights[empty] - log_weights[empty].max(axis=1)[:, None])
        one_hot = np.array([[row['outcome'] == name for name in NAMES] for row in group], dtype=float)
        columns.append((weights @ one_hot) / weights.sum(axis=1)[:, None])
        support.append(dict(N=int(n), runs=len(group), minimum_share=float(evidence.min()),
                            maximum_share=float(evidence.max())))
    columns = np.asarray(columns)
    interpolated = np.array([[np.interp(log_n, np.log2(populations), columns[:, j, c])
                             for j in range(len(shares))] for c in range(4)])
    smooth = blur_axis(interpolated, .12 / (log_n[1] - log_n[0]), axis=2)
    smooth /= smooth.sum(axis=0)
    lower = np.interp(log_n, np.log2(populations), [s['minimum_share'] for s in support])
    upper = np.interp(log_n, np.log2(populations), [s['maximum_share'] for s in support])
    dx = np.maximum(np.log2(populations.min()) - log_n, 0)
    dy = np.maximum(lower[None, :] - shares[:, None], 0) + np.maximum(shares[:, None] - upper[None, :], 0)
    extension = np.exp(-.5 * (dx[None, :] / .32)**2 - .5 * (dy / .065)**2)
    ordered = np.sort(smooth, axis=0)
    margin = ordered[-1] - ordered[-2]
    alpha = EMPIRICAL_MAX_OPACITY * np.minimum(margin / .30, 1)**.55 * extension
    winner = smooth.argmax(axis=0)
    colors = np.array([to_rgb(c) for c in COLORS])
    rgba = np.concatenate([colors[winner], alpha[:, :, None]], axis=-1)
    assert np.isfinite(rgba).all() and np.all((rgba >= 0) & (rgba <= 1))
    metadata = dict(share_bandwidth=share_bandwidth, log2N_smoothing_sigma=.12,
                    maximum_display_opacity=EMPIRICAL_MAX_OPACITY,
                    range_fade_log2N_sigma=.32, range_fade_share_sigma=.065,
                    sampled_ranges=support, theory_probabilities_used=False,
                    class_rebalancing=False,
                    max_probability_sum_error=float(np.abs(smooth.sum(axis=0) - 1).max()))
    return log_n, shares, rgba, metadata


def flag_panel(fig, ax, crops, example):
    ax.imshow(plt.imread(DATA / example['flag_file']), extent=[0, 24, 16, 0], alpha=.32, zorder=0)
    dots = ax.scatter([r['center_x'] for r in crops], [r['center_y'] for r in crops],
        c=[r['rival_response_share'] for r in crops], cmap=style.FIELD_CMAP.reversed(),
        vmin=0, vmax=1, s=24, edgecolors=style.WHITE, linewidths=.45, alpha=.92, zorder=3)
    ax.set_xlim(0, 24)
    ax.set_ylim(16, 0)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(style.LIGHT_GRAY)
        spine.set_linewidth(.8)
    style.panel_title(ax, 'a) Crop-level evidence share, Yemen example')
    cb = fig.colorbar(dots, ax=ax, orientation='horizontal', fraction=.060, pad=.035,
                      ticks=[0, .25, .5, .75, 1])
    cb.ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    cb.set_label('Rival response fraction', labelpad=1, fontsize=6.6)
    cb.ax.tick_params(labelsize=6.2, width=.7, length=2.2)
    cb.ax.text(0, -1.85, 'Yemen', transform=cb.ax.transAxes, ha='left', va='top',
               fontsize=6.3, color=style.BLUE_DARK)
    cb.ax.text(1, -1.85, 'Austria', transform=cb.ax.transAxes, ha='right', va='top',
               fontsize=6.3, color=style.ORANGE_DARK)
    return cb


def theory_phase(ax, n_fine, shares, probabilities, labels=True):
    winner = probabilities.argmax(axis=0)
    count = len(THEORY_NAMES)
    ax.pcolormesh(n_fine, shares, winner, shading='nearest', cmap=ListedColormap(PALE[:count]),
                  vmin=0, vmax=count - 1, rasterized=True, zorder=-2)
    for a in range(count):
        for b in range(a + 1, count):
            diff = np.where((winner == a) | (winner == b), probabilities[a] - probabilities[b], np.nan)
            finite = diff[np.isfinite(diff)]
            if len(finite) and finite.min() < 0 < finite.max():
                ax.contour(n_fine, shares, diff, levels=[0], colors='#25272B', linewidths=.85, zorder=1)
    ax.axhline(.35, c='#777777', ls='--', lw=.7, zorder=1)
    style.population_axis(ax)
    ax.set_ylabel(SHARE_AXIS_LABEL)
    if not labels:
        return
    ax.text(25, .10, 'Correct\nconsensus', ha='center', va='center', fontsize=8.1,
            weight='bold', color='#087E53')
    ax.text(40, .63, 'Polarization', ha='center', va='center', fontsize=8.1,
            weight='bold', color='#B74620')
    ax.text(25, .945, 'Wrong consensus', ha='center', va='center', fontsize=7.6,
            weight='bold', color='#2F6594')
    ax.text(116, .35 + .013, 'Panel (b)', ha='right', va='bottom', fontsize=6.4, color=style.GRAY)


def draw_figure(rows, crops, example, populations, shares, probabilities, reference, soft,
                dots=False, theory_background=False):
    n_fine = np.unique(np.concatenate([
        np.exp2(np.linspace(np.log2(populations[0]), np.log2(populations[-1]), 1201)),
        populations.astype(float)]))
    smooth, interpolation = interpolate_theory(populations, probabilities, n_fine)
    reference_smooth, _ = interpolate_theory(populations, reference[:, :, None], n_fine)
    fig = plt.figure(figsize=(13.9, 3.45), constrained_layout=False)
    grid = fig.add_gridspec(1, 4, width_ratios=[1.10, 1.18, 1.05, 1.05],
                            left=.045, right=.992, top=.82, bottom=.285, wspace=.42)
    axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
    colorbar = flag_panel(fig, axes[0], crops, example)
    fig.canvas.draw()
    box = colorbar.ax.get_position()
    fig.text((box.x0 + box.x1) / 2, box.y0 - .135,
              f"Rival evidence share: {example['rival_response_share']:.3f}",
              ha='center', va='top', fontsize=7.2, color=style.INK)
    for c in range(len(THEORY_NAMES)):
        axes[1].plot(n_fine, reference_smooth[c, 0],
                      c=COLORS[c], lw=1.65, label=LABELS[c])
    style.population_axis(axes[1])
    axes[1].set_ylabel('Outcome probability')
    style.panel_title(axes[1], 'b) Outcomes at rival share 0.35')
    axes[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(.50, -.27),
                   ncol=2, fontsize=6.5, handlelength=1.6, columnspacing=.9,
                   borderpad=0, labelspacing=.3)
    theory_phase(axes[2], n_fine, shares, smooth)
    style.panel_title(axes[2], 'c) Theoretical phase diagram')
    if theory_background:
        theory_phase(axes[3], n_fine, shares, smooth, labels=False)
    else:
        log_n, empirical_shares, rgba, _ = soft
        axes[3].pcolormesh(2**log_n, empirical_shares, rgba, shading='nearest', rasterized=True, zorder=-2)
        style.population_axis(axes[3])
    if dots:
        for row in rows:
            axes[3].scatter(row['display_N'], row['rival_response_share'], s=24,
                             c=DOT_COLORS[row['outcome']], edgecolors='white',
                             lw=.4, zorder=5, clip_on=False)
    axes[3].set_ylabel(SHARE_AXIS_LABEL)
    style.panel_title(axes[3], 'd) Empirical run results')
    style.panel_title(axes[0], 'a) Crop-level evidence share,\nYemen example')
    for collection in axes[0].collections:
        collection.set_sizes([32])
    for line in axes[1].lines:
        line.set_linewidth(2)
    fig.canvas.draw()
    for label in fig.findobj(matplotlib.text.Text):
        label.set_fontsize(label.get_fontsize() * TEXT_SCALE)
    for handle in axes[1].get_legend().legend_handles:
        if isinstance(handle, Line2D):
            handle.set_markersize(handle.get_markersize() * 1.35)
    axes[1].get_legend().set_bbox_to_anchor((.50, -.31))
    for ax in axes[1:]:
        title = ax.set_title(ax.get_title(loc='left'), loc='left', x=-.16, pad=8)
        title.set_fontsize(9.7 * TEXT_SCALE)
        title.set_fontweight('bold')
    fig.theory_interpolation = interpolation
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE.parent / 'paper/figures/final')
    parser.add_argument('--bandwidth', type=float, default=.05,
                        help='Gaussian width in evidence share; other smoothing/fading is unchanged.')
    parser.add_argument('--dots', action='store_true', help='Overlay the 217 measured run coordinates; no example ring.')
    parser.add_argument('--theory-background', action='store_true',
                        help='With --dots, show theory regions behind empirical points instead of soft regions.')
    parser.add_argument('--formats', nargs='+', choices=['pdf', 'png', 'svg'], default=['pdf', 'png', 'svg'])
    args = parser.parse_args()
    if args.theory_background and not args.dots:
        parser.error('--theory-background requires --dots.')
    rows, crops, example, populations, shares, probabilities, reference = load_inputs()
    style.setup_style()
    soft = soft_fields(rows, args.bandwidth)
    fig = draw_figure(rows, crops, example, populations, shares, probabilities, reference,
                      soft, args.dots, args.theory_background)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in dict.fromkeys(args.formats):
        output = args.output_dir / f'fig8.{extension}'
        fig.savefig(output, bbox_inches='tight')
        outputs.append(str(output))
    interpolation = fig.theory_interpolation
    plt.close(fig)
    validation = dict(status='PASS', empirical_runs=len(rows),
        empirical_outcome_counts=dict(Counter(row['outcome'] for row in rows)),
        panel_b_share=.35, total_anchor_probability=.45, h0=.3,
        theoretical_outcomes=THEORY_NAMES, panel_b_dots_drawn=False,
        theory_interpolation=interpolation,
        panel_d_dots_drawn=args.dots, panel_d_legend_drawn=False,
        panel_d_theory_background=args.theory_background,
        yemen_highlight_ring_drawn=False, yemen_rival_response_count=4192,
        yemen_total_responses=6400, yemen_rival_share=.655,
        typography_scale=TEXT_SCALE, title_x_axes_fraction_bcd=-.16,
        soft_regions=soft[3],
        input_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(DATA.iterdir()) if path.is_file()},
        new_paid_calls=0, outputs=outputs)
    (HERE / 'validation/figure_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(dict(outputs=outputs, empirical_runs=len(rows), bandwidth=args.bandwidth, dots=args.dots), indent=2))


if __name__ == '__main__':
    main()
