"""Paper step 3b (amendment v3.8): total cost of ownership (TCO) of every classification configuration.

Measured inputs come from the step-2 runs (notebook 17, `paper_step2_costs.csv` / `paper_step2_pareto.csv`): labels
needed, training seconds per run, inference seconds and API USD per review, whether the configuration calls the API.
Assumed inputs come from `03_configs/frozen/paper_cost_model.yaml` and are a sensitivity grid, not measurements:
cost per label, GPU/CPU price per hour, monthly volume, horizon, drift checks and deployment mode.

Over a horizon of T months with V reviews a month, a configuration costs
    labels      (initial labels + labels of every drift check) × cost per label
    training    (tuning runs + final run + one retrain per drift check) × training hours × hardware price
    inference   batch: V·T × hours per review × hardware price;  endpoint: hours per month · T × hardware price
    API         V·T × USD per review + one Validation pass at the start and at every drift check
so the total is a fixed part plus a part proportional to V·T, which gives closed-form break-even volumes.
"""
import math

import numpy as np
import pandas as pd

TRAINED = ('TF-IDF', 'Bi-LSTM + fastText', 'ViSoBERT', 'PhoBERT')
GPU_MODELS = ('Bi-LSTM + fastText', 'ViSoBERT', 'PhoBERT')
LLM = 'LLM few-shot'
# largest tuning grid of each strategy (configurations tried on Validation), used to cost the tuning runs
GRID_RUNS = {'TF-IDF': 18, 'Bi-LSTM + fastText': 24, 'ViSoBERT': 9, 'PhoBERT': 9, LLM: 0}


def parts(name):
    """Strategies used by a configuration name of notebook 17."""
    if name.startswith('Phẳng '):
        return [name[len('Phẳng '):]]
    if name.startswith('Cascade '):
        return [name[len('Cascade '):]]
    return name.split(' → ')


def components(costs, pareto, labels_initial):
    """One row per configuration with the measured cost drivers and its Hold-out Macro-F1 (canonical run, %)."""
    c = costs.merge(pareto[['cấu hình', 'Macro-F1']], on='cấu hình')
    rows = []
    for _, r in c.iterrows():
        p = parts(r['cấu hình'])
        trained = any(x in TRAINED for x in p)
        hw = 'GPU' if any(x in GPU_MODELS for x in p) else ('CPU' if 'TF-IDF' in p else 'không')
        rows.append({'cấu hình': r['cấu hình'], 'mã': r['mã'], 'kiến trúc': r['kiến trúc'], 'Macro-F1': r['Macro-F1'] * 100,
                     'gọi API': bool(r['gọi API']), 'có huấn luyện': trained, 'phần cứng': hw,
                     'nhãn ban đầu': labels_initial['trained'] if trained else labels_initial['llm_only'],
                     'lần chạy dò lưới': max(GRID_RUNS[x] for x in p),
                     'giây train / lần chạy': r['giây train / lần chạy'],
                     'giây suy luận / review': r['giây / 1.000 review'] / 1000,
                     'USD API / review': r['USD API / 1.000 review'] / 1000})
    return pd.DataFrame(rows)


def drift_checks(cfg):
    return max(math.ceil(cfg['horizon_months'] / cfg['drift']['check_every_months']) - 1, 0)


def fixed_and_variable(row, cfg, label_cost, gpu_price, mode):
    """(fixed USD over the horizon, USD per review) of one configuration."""
    k = drift_checks(cfg)
    d = cfg['drift']
    labels = row['nhãn ban đầu'] + k * (d['validation_labels_per_check'] + (d['extra_train_labels_per_check'] if row['có huấn luyện'] else 0))
    price = {'GPU': gpu_price, 'CPU': cfg['cpu_usd_per_hour'], 'không': 0.0}[row['phần cứng']]
    runs = (row['lần chạy dò lưới'] + 1 + (k if row['có huấn luyện'] else 0)) if row['có huấn luyện'] else 0
    fixed = labels * label_cost + runs * row['giây train / lần chạy'] / 3600 * price
    fixed += (1 + k) * d['validation_labels_per_check'] * row['USD API / review']          # Validation pass of the API part
    variable = row['USD API / review']
    if mode == 'batch':
        variable += row['giây suy luận / review'] / 3600 * price
    else:
        fixed += cfg['endpoint_hours_per_month'] * cfg['horizon_months'] * price
    return fixed, variable


def tco(comp, cfg, volume, label_cost, gpu_price, mode):
    """Total cost over the horizon of every configuration for one scenario."""
    n = volume * cfg['horizon_months']
    out = comp[['cấu hình', 'mã', 'kiến trúc', 'Macro-F1', 'gọi API']].copy()
    fv = [fixed_and_variable(r, cfg, label_cost, gpu_price, mode) for _, r in comp.iterrows()]
    out['USD cố định'] = [f for f, _ in fv]
    out['USD biến đổi'] = [v * n for _, v in fv]
    out['tổng USD'] = out['USD cố định'] + out['USD biến đổi']
    out[['review / tháng', 'chi phí một nhãn', 'giá GPU / giờ', 'triển khai']] = volume, label_cost, gpu_price, mode
    return out


def scenarios(comp, cfg):
    """TCO of every configuration in every scenario of the grid."""
    return pd.concat([tco(comp, cfg, v, lc, g, m) for m in cfg['deployment'] for g in cfg['gpu_usd_per_hour']
                      for lc in cfg['label_cost_usd'] for v in cfg['monthly_volume']], ignore_index=True)


def break_even(comp, cfg, a, b, label_cost, gpu_price, mode='batch'):
    """Monthly volume above which configuration a (fixed-cost heavy) becomes cheaper than b (variable-cost heavy);
    inf when a never becomes cheaper, 0 when a is cheaper at any volume."""
    ra, rb = (comp.set_index('cấu hình').loc[x] for x in (a, b))
    fa, va = fixed_and_variable(ra, cfg, label_cost, gpu_price, mode)
    fb, vb = fixed_and_variable(rb, cfg, label_cost, gpu_price, mode)
    if fa <= fb and va <= vb:
        return 0.0
    if vb <= va:
        return float('inf')
    return (fa - fb) / (vb - va) / cfg['horizon_months']


def cheapest(sc, threshold, allow_api=True, rel_tol=0.01, abs_tol=10.0):
    """Per scenario: the cheapest configuration whose Macro-F1 reaches `threshold`. Costs within `rel_tol` of the
    scenario minimum or within `abs_tol` USD over the horizon count as equal, and among those the higher Macro-F1 wins
    (a few dollars a year should not pick a weaker model)."""
    keys = ['triển khai', 'giá GPU / giờ', 'chi phí một nhãn', 'review / tháng']
    cols = keys + ['mã', 'cấu hình', 'Macro-F1', 'tổng USD']
    s = sc[(sc['Macro-F1'] >= threshold) & (allow_api | ~sc['gọi API'])]
    rows = []
    for _, g in s.groupby(keys):
        m = g['tổng USD'].min()
        rows.append(g[g['tổng USD'] <= max(m * (1 + rel_tol), m + abs_tol)].sort_values(['Macro-F1', 'tổng USD'], ascending=[False, True]).iloc[0][cols])
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']   # categorical slots 1–6, fixed order
INK2, GRID, AXIS = '#52514e', '#e1e0d9', '#c3c2b7'


def tco_curves(comp, cfg, names, label_cost, gpu_price, volumes):
    """Long table: total USD over the horizon for each configuration, deployment mode and monthly volume."""
    return pd.concat([tco(comp[comp['cấu hình'].isin(names)], cfg, v, label_cost, gpu_price, m)
                      for m in cfg['deployment'] for v in volumes], ignore_index=True)


def tco_figure(curves, names, labels, path, text, styles=None):
    """One panel per deployment mode: total cost over the horizon against monthly volume (log–log), one line per
    configuration in a fixed colour order; legend below (identity never by colour alone: line + legend label)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    modes = list(dict.fromkeys(curves['triển khai']))
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.2), sharey=True)
    for ax, m in zip(np.atleast_1d(axes), modes):
        for i, n in enumerate(names):
            d = curves[(curves['triển khai'] == m) & (curves['cấu hình'] == n)].sort_values('review / tháng')
            ax.plot(d['review / tháng'], d['tổng USD'], color=SERIES[i], lw=2, ls=(styles or {}).get(n, '-'), label=labels[n])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(text['modes'][m], fontsize=10, color=INK2)
        ax.set_xlabel(text['x'], color=INK2)
        ax.grid(True, which='major', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK2)
    np.atleast_1d(axes)[0].set_ylabel(text['y'], color=INK2)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path
