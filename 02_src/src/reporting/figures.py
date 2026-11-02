"""Thesis figures for GĐ1–GĐ11, drawn from results already saved in 05_experiments/ and 01_data/ (nothing is retrained,
the Hold-out is not re-opened: GĐ10 figures read the gd10_* result tables only).

Every function takes plain DataFrames / dicts and an output path, and returns the path. Look and colour rules: see
src/reporting/style.py.
"""
import numpy as np
import pandas as pd

from src.reporting.style import (APP_COLOR, GREY, GRID, INK, INK_2, SLOT, STRATEGY_COLOR, grid_lines, num, pct, save,
                                 seq_cmap, style, text_color_on, vn, vn_axes)


def _strategy_color(name):
    n = str(name).lower()
    if 'baseline' in n:                        # "Baseline phẳng ViSoBERT" is the reference, not the ViSoBERT cascade
        return GREY
    for k, c in STRATEGY_COLOR.items():
        if k.split(' ')[0].lower() in n:
            return c
    return GREY


# ---------------------------------------------------------------- GĐ1–2: data
def fig_rating(ratings, path):
    """ratings: {app: Series of 1–5}. Grouped bars of the share of each star rating, one colour per app."""
    import matplotlib.pyplot as plt
    style()
    apps = list(ratings)
    fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
    x = np.arange(1, 6)
    w = 0.38
    for i, app in enumerate(apps):
        share = ratings[app].value_counts(normalize=True).reindex(x, fill_value=0).to_numpy()
        pos = x + (i - (len(apps) - 1) / 2) * w
        ax.bar(pos, share, width=w - 0.04, color=APP_COLOR.get(app, SLOT[i]), label=f'{app} (n = {num(len(ratings[app]))})')
        for px, s in zip(pos, share):
            ax.text(px, s + 0.01, pct(s, 1), ha='center', va='bottom', fontsize=8, color=INK_2)
    ax.set_xticks(x, [f'{k} sao' for k in x])                  # '★' is not in Times New Roman
    ax.set_ylabel('Tỷ lệ review')
    vn_axes(ax, 'y', percent=True)
    grid_lines(ax)
    ax.legend(frameon=False, loc='upper center')
    return save(fig, path)


def fig_monthly(dates, path):
    """dates: {app: Series of datetimes}. One panel per app (different scales → no shared y-axis), reviews per month."""
    import matplotlib.pyplot as plt
    style()
    apps = list(dates)
    fig, axes = plt.subplots(len(apps), 1, figsize=(8.0, 2.3 * len(apps)), constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, app in zip(axes, apps):
        m = dates[app].dt.to_period('M').value_counts().sort_index()
        xs = np.arange(len(m))
        ax.bar(xs, m.to_numpy(), color=APP_COLOR.get(app, SLOT[0]), width=0.75)
        ax.set_title(f'{app} — số review mỗi tháng (tổng {num(m.sum())})', loc='left')
        ax.set_xticks(xs, [f'{p.month:02d}/{p.year}' for p in m.index], rotation=45, ha='right')
        vn_axes(ax, 'y')
        grid_lines(ax)
    return save(fig, path)


def fig_length(words, path, clip=80):
    """words: {app: Series of word counts}. Histogram per app (counts above `clip` folded into the last bin), median
    marked."""
    import matplotlib.pyplot as plt
    style()
    apps = list(words)
    fig, axes = plt.subplots(1, len(apps), figsize=(4.2 * len(apps), 2.9), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, app in zip(axes, apps):
        w = words[app].clip(upper=clip)
        ax.hist(w, bins=np.arange(0, clip + 2, 2), color=APP_COLOR.get(app, SLOT[0]), edgecolor='white', linewidth=0.6,
                weights=np.full(len(w), 1 / len(w)))
        med = float(words[app].median())
        ax.axvline(med, color=INK, lw=1.2, ls='--')
        ax.text(med + 1, ax.get_ylim()[1] * 0.92, f'trung vị {vn(med, 0)} từ', fontsize=8.5, color=INK)
        ax.set_title(app, loc='left')
        ax.set_xlabel(f'Số từ mỗi review (≥ {clip} gộp vào cột cuối)')
        ax.set_ylabel('Tỷ lệ review')
        vn_axes(ax, 'y', percent=True)
        grid_lines(ax)
    return save(fig, path)


# ---------------------------------------------------------------- GĐ5: learning curve
def fig_learning_curve(summary, panels, path):
    """summary: rows strategy, size, tier, metric, mean, std, n_seeds (both strategies). panels: [(tier, metric, title)].
    Lines per strategy, band = ± 1 std over seeds (none when a single seed)."""
    import matplotlib.pyplot as plt
    style()
    names = {'visobert': 'ViSoBERT', 'bilstm': 'Bi-LSTM + fastText'}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.0), constrained_layout=True, sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (tier, metric, title) in zip(axes, panels):
        for key, label in names.items():
            d = summary[(summary['strategy'] == key) & (summary['tier'] == tier) & (summary['metric'] == metric)].sort_values('size')
            if d.empty:
                continue
            c = STRATEGY_COLOR[label]
            seeds = int(d['n_seeds'].max())
            ax.plot(d['size'], d['mean'], color=c, lw=2, marker='o', ms=6, mec='white', mew=1.2,
                    label=f'{label} ({seeds} seed)')
            if seeds > 1:
                ax.fill_between(d['size'], d['mean'] - d['std'], d['mean'] + d['std'], color=c, alpha=0.15, lw=0)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Số review Train')
        ax.set_xticks(sorted(summary['size'].unique()))
        vn_axes(ax)
        grid_lines(ax)
    axes[0].legend(frameon=False, loc='lower right')
    return save(fig, path)


# ---------------------------------------------------------------- GĐ6: threshold
def fig_threshold(sweeps, chosen, min_recall, path):
    """sweeps: {model: DataFrame threshold, precision_actionable, recall_actionable, macro_f1}; chosen: {model: thr}."""
    import matplotlib.pyplot as plt
    style()
    series = [('precision_actionable', 'Precision ACTIONABLE', SLOT[0]), ('recall_actionable', 'Recall ACTIONABLE', SLOT[1]),
              ('macro_f1', 'Macro-F1 (2 lớp)', SLOT[2])]
    fig, axes = plt.subplots(1, len(sweeps), figsize=(4.4 * len(sweeps), 3.2), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (model, s) in zip(axes, sweeps.items()):
        for col, label, c in series:
            ax.plot(s['threshold'], s[col], color=c, lw=2, label=label)
        ax.axhline(min_recall, color=INK_2, lw=1, ls='--')
        ax.text(s['threshold'].max(), min_recall + 0.01, f'sàn recall {vn(min_recall, 2)}', ha='right', va='bottom',
                fontsize=8.5, color=INK_2)
        t = chosen[model]
        ax.axvline(t, color=INK, lw=1)
        ax.text(t + 0.01, 0.05, f'ngưỡng {vn(t, 2)}', fontsize=8.5, color=INK)
        ax.set_title(model, loc='left')
        ax.set_xlabel('Ngưỡng xác suất ACTIONABLE')
        ax.set_ylim(0, 1.02)
        vn_axes(ax)
        grid_lines(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False)
    return save(fig, path)


# ---------------------------------------------------------------- confusion matrices
def fig_confusions(mats, path, ncols=None, title_suffix=''):
    """mats: {panel title: DataFrame counts (rows = gold, columns = predicted)}. Cell colour = share of the gold row;
    text = count and row share."""
    import matplotlib.pyplot as plt
    style()
    ncols = ncols or len(mats)
    nrows = int(np.ceil(len(mats) / ncols))
    first = next(iter(mats.values()))
    size = 0.62 * max(first.shape) + 1.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols + 0.9, size * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    cmap = seq_cmap()
    im = None
    for ax, (title, m) in zip(axes, mats.items()):
        counts = m.to_numpy(dtype=float)
        share = counts / counts.sum(1, keepdims=True).clip(min=1e-12)
        im = ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                fill = '#%02x%02x%02x' % tuple(int(255 * v) for v in cmap(share[i, j])[:3])
                v = counts[i, j]
                ax.text(j, i, f'{vn(v, 0 if float(v).is_integer() else 1)}\n{pct(share[i, j], 0)}', ha='center', va='center',
                        fontsize=8.5, color=text_color_on(fill))
        ax.set_xticks(range(m.shape[1]), [str(c).replace('đoán ', '') for c in m.columns])
        ax.set_yticks(range(m.shape[0]), [str(r).replace('gold ', '') for r in m.index])
        ax.set_xlabel('Dự đoán'); ax.set_ylabel('Nhãn gold')
        ax.set_title(title + title_suffix, loc='left')
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    for ax in axes[len(mats):]:
        ax.axis('off')
    cb = fig.colorbar(im, ax=list(axes[:len(mats)]), shrink=0.8, pad=0.02)
    cb.set_label('Tỷ lệ theo hàng (trên số review của nhãn gold)')
    cb.ax.yaxis.set_major_formatter(__import__('matplotlib').ticker.FuncFormatter(lambda t, _: pct(t, 0)))
    cb.outline.set_visible(False)
    return save(fig, path)


# ---------------------------------------------------------------- GĐ10: CI forest
def fig_forest(panels, path, xlabel='Macro-F1 bốn lớp (CI 95%, bootstrap 10.000 lần)'):
    """panels: {title: DataFrame config, value, lo, hi} (shares 0–1). Dot = point estimate, whisker = 95% CI."""
    import matplotlib.pyplot as plt
    style()
    n = max(len(d) for d in panels.values())
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 0.55 * n + 1.3), constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (title, d) in zip(axes, panels.items()):
        d = d.iloc[::-1].reset_index(drop=True)
        for i, r in d.iterrows():
            c = _strategy_color(r['config'])
            ax.plot([r['lo'], r['hi']], [i, i], color=c, lw=2.2, solid_capstyle='round')
            ax.scatter([r['value']], [i], s=60, color=c, edgecolor='white', linewidth=1.5, zorder=3)
            ax.text(r['hi'] + 0.008, i, f"{vn(r['value'] * 100, 1)} ({vn(r['lo'] * 100, 1)}–{vn(r['hi'] * 100, 1)})",
                    va='center', fontsize=8.5, color=INK_2)
        ax.set_yticks(range(len(d)), d['config'])
        ax.set_title(title, loc='left')
        ax.set_xlabel(xlabel)
        vn_axes(ax, 'x', percent=True)
        grid_lines(ax, 'x')
        ax.tick_params(axis='y', length=0)
    lo = min(d['lo'].min() for d in panels.values())
    hi = max(d['hi'].max() for d in panels.values())
    axes[0].set_xlim(max(0, lo - 0.05), min(1, hi + 0.16))
    return save(fig, path)


# ---------------------------------------------------------------- GĐ10: hybrid 3×3
def fig_hybrid_heatmap(hybrid, path):
    """hybrid: rows 'tầng 1 (X)', 'tầng 2 (Y)', 'Macro-F1 (4 lớp)' (percentage points), 'CI 95%' ('a–b')."""
    import matplotlib.pyplot as plt
    style()
    order = ['ViSoBERT', 'Bi-LSTM + fastText', 'LLM few-shot']
    m = hybrid.pivot(index='tầng 1 (X)', columns='tầng 2 (Y)', values='Macro-F1 (4 lớp)').reindex(index=order, columns=order)
    ci = hybrid.set_index(['tầng 1 (X)', 'tầng 2 (Y)'])['CI 95%']
    v = m.to_numpy(dtype=float)
    cmap = seq_cmap()
    vmin, vmax = np.floor(v.min() / 5) * 5, np.ceil(v.max() / 5) * 5
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    im = ax.imshow(v, cmap=cmap, vmin=vmin, vmax=vmax)
    for i, x in enumerate(order):
        for j, y in enumerate(order):
            fill = '#%02x%02x%02x' % tuple(int(255 * c) for c in cmap((v[i, j] - vmin) / (vmax - vmin))[:3])
            ax.text(j, i, f'{vn(v[i, j], 1)}\n({str(ci[(x, y)]).replace(".", ",")})', ha='center', va='center', fontsize=9,
                    color=text_color_on(fill), fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(3), order); ax.set_yticks(range(3), order)
    ax.set_xlabel('Tầng 2 (BUG / FEATURE / UX)'); ax.set_ylabel('Tầng 1 (ACTIONABLE / NOT)')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('Macro-F1 bốn lớp (%)')
    cb.ax.yaxis.set_major_formatter(__import__('matplotlib').ticker.FuncFormatter(lambda t, _: vn(t, 0)))
    cb.outline.set_visible(False)
    return save(fig, path)


# ---------------------------------------------------------------- GĐ10: cost vs performance
def fig_cost_performance(hybrid, path):
    """Macro-F1 (with 95% CI) against inference seconds per 1,000 reviews (log scale) for every Tier 1 × Tier 2 pair.
    Colour = Tier 1 strategy, marker = Tier 2 strategy (composite encoding, both in the legend)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style()
    markers = {'ViSoBERT': 'o', 'Bi-LSTM + fastText': 's', 'LLM few-shot': '^'}
    short = {'ViSoBERT': 'V', 'Bi-LSTM + fastText': 'B', 'LLM few-shot': 'L'}
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    for _, r in hybrid.iterrows():
        x, y = r['giây / 1.000 review'], r['Macro-F1 (4 lớp)']
        lo, hi = (float(v) for v in str(r['CI 95%']).split('–'))
        c = STRATEGY_COLOR[r['tầng 1 (X)']]
        ax.plot([x, x], [lo, hi], color=c, lw=1.4, alpha=0.6)
        ax.scatter([x], [y], s=70, color=c, marker=markers[r['tầng 2 (Y)']], edgecolor='white', linewidth=1.3, zorder=3)
        ax.annotate(f"{short[r['tầng 1 (X)']]}→{short[r['tầng 2 (Y)']]}", (x, y), textcoords='offset points', xytext=(7, -3),
                    fontsize=8.5, color=INK_2)
    ax.set_xscale('log')
    ax.set_xlabel('Thời gian suy luận cho 1.000 review (giây, thang log)')
    ax.set_ylabel('Macro-F1 bốn lớp trên Hold-out (%)')
    ax.xaxis.set_major_formatter(__import__('matplotlib').ticker.FuncFormatter(lambda t, _: vn(t)))
    vn_axes(ax, 'y')
    grid_lines(ax)
    grid_lines(ax, 'x')
    h1 = [Line2D([], [], color=STRATEGY_COLOR[k], lw=6, label=k) for k in markers]
    h2 = [Line2D([], [], color=INK_2, marker=mk, ls='none', ms=7, label=k) for k, mk in markers.items()]
    leg = ax.legend(handles=h1, title='Màu = tầng 1', frameon=False, loc='lower right')
    ax.add_artist(leg)
    ax.legend(handles=h2, title='Hình dấu = tầng 2', frameon=False, loc='lower center')
    ax.text(0.01, 0.99, 'V = ViSoBERT, B = Bi-LSTM + fastText, L = LLM few-shot; X→Y = tầng 1 X, tầng 2 Y.\n'
                        'Đoạn dọc = CI 95% (seed 42 / một lần chạy LLM).', transform=ax.transAxes, va='top', fontsize=8,
            color=INK_2)
    return save(fig, path)


# ---------------------------------------------------------------- GĐ12: manual topic purity
def fig_manual_purity(per, names, path, flag_below=0.60):
    """Horizontal bars of manual purity per topic (mean of two raters, sorted), the two raters as small ticks, bar colour
    = dominant type of the topic, dashed line = BA flag threshold."""
    import matplotlib.pyplot as plt
    from src.clustering.figures import TYPE_COLOR, _type_handles, topic_label
    style()
    d = per.sort_values('purity', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.0, 0.34 * len(d) + 1.2), constrained_layout=True)
    y = np.arange(len(d))
    ax.barh(y, d['purity'], color=[TYPE_COLOR.get(t, GREY) for t in d['dominant_type']], height=0.68)
    ax.scatter(d['purity_A'], y, marker='|', s=90, color=INK, linewidths=1.4, zorder=3, label='Người chấm A')
    ax.scatter(d['purity_B'], y, marker='|', s=90, color=INK_2, linewidths=1.4, zorder=3, label='Người chấm B')
    for yi, (_, r) in zip(y, d.iterrows()):                  # label after the bar AND both rater ticks
        ax.text(max(r['purity'], r['purity_A'], r['purity_B']) + 0.02, yi, pct(r['purity'], 0), va='center', fontsize=8.5,
                color=INK_2)
    ax.axvline(flag_below, color=INK, lw=1, ls='--')
    ax.text(flag_below + 0.01, len(d) - 0.4, f'ngưỡng cờ BA {vn(flag_below, 2)}', fontsize=8.5, color=INK, va='bottom')
    ax.set_yticks(y, [topic_label(int(t), names.get(int(t), ''), 42) for t in d['topic_id']])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Manual topic purity (trung bình hai người chấm, 10 review mỗi cụm)')
    vn_axes(ax, 'x', percent=True)
    grid_lines(ax, 'x')
    ax.tick_params(axis='y', length=0)
    h = _type_handles()[:4]
    ax.legend(handles=h, title='Màu = loại chủ đạo', frameon=False, loc='lower right')
    return save(fig, path)


# ---------------------------------------------------------------- GĐ13: temporal evidence
def fig_temporal_heatmap(groups, names, window_labels, path):
    """Rows = issue groups (topics), columns = 30-day windows oldest → newest; cell colour = share of the group's reviews
    in that window, text = count; status and N written at the right."""
    import matplotlib.pyplot as plt
    from src.clustering.figures import topic_label
    style()
    d = groups.reset_index(drop=True)
    m = np.array([w[::-1] for w in d['per_window']], dtype=float)          # stored W0..W11 → oldest first
    share = m / m.sum(1, keepdims=True).clip(min=1)
    cmap = seq_cmap()
    fig, ax = plt.subplots(figsize=(11.0, 0.42 * len(d) + 1.6), constrained_layout=True)
    im = ax.imshow(share, cmap=cmap, aspect='auto', vmin=0, vmax=max(0.2, float(share.max())))
    vmax = max(0.2, float(share.max()))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            fill = '#%02x%02x%02x' % tuple(int(255 * c) for c in cmap(share[i, j] / vmax)[:3])
            ax.text(j, i, num(m[i, j]), ha='center', va='center', fontsize=7.5, color=text_color_on(fill))
        ax.text(m.shape[1] - 0.35, i, f"  {d.loc[i, 'status']}  (N = {num(d.loc[i, 'N'])})", ha='left', va='center', fontsize=8.5,
                color=INK)
    ax.set_xticks(range(m.shape[1]), window_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(d)), [topic_label(int(t), names.get(int(t), ''), 40) for t in d['topic_id']])
    ax.set_xlabel('Cửa sổ 30 ngày (cũ → mới; cột cuối = 30 ngày trước T0 = 06/10/2026)')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, m.shape[1] + 2.6)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label('Tỷ lệ review của group trong cửa sổ')
    cb.ax.yaxis.set_major_formatter(__import__('matplotlib').ticker.FuncFormatter(lambda t, _: pct(t, 0)))
    cb.outline.set_visible(False)
    return save(fig, path)


# ---------------------------------------------------------------- Validation vs Hold-out
def fig_val_vs_holdout(rows, path):
    """rows: DataFrame metric, strategy, validation, holdout (percentage points). One panel per metric, a dumbbell per
    strategy: hollow = Validation, filled = Hold-out."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style()
    metrics = list(dict.fromkeys(rows['metric']))
    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.6, 2.5 * nrows + 0.4), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(metrics):]:
        ax.axis('off')
    for ax, mt in zip(axes, metrics):
        d = rows[rows['metric'] == mt].reset_index(drop=True)
        for i, r in d.iterrows():
            c = _strategy_color(r['strategy'])
            ax.plot([r['validation'], r['holdout']], [i, i], color=c, lw=2, alpha=0.7)
            ax.scatter([r['validation']], [i], s=55, facecolor='white', edgecolor=c, linewidth=1.8, zorder=3)
            ax.scatter([r['holdout']], [i], s=55, color=c, edgecolor='white', linewidth=1.2, zorder=3)
        ax.set_yticks(range(len(d)), d['strategy'])
        ax.invert_yaxis()
        ax.set_title(mt, loc='left')
        vn_axes(ax, 'x')
        grid_lines(ax, 'x')
        ax.tick_params(axis='y', length=0)
    handles = [Line2D([], [], marker='o', ls='none', mfc='white', mec=INK_2, ms=7, label='Validation'),
               Line2D([], [], marker='o', ls='none', color=INK_2, ms=7, label='Hold-out')]
    fig.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False)
    return save(fig, path)
