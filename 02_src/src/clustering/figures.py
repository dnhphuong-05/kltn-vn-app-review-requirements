"""RQ2 (GĐ12) — static figures for the thesis (matplotlib, 300 dpi PNG).

Colour encodes the dominant TYPE of a topic, never the topic itself: BUG / FEATURE / UX take the first three categorical
slots of the validated palette (all-pairs CVD-safe), `mixed` and outliers are neutral greys. Topics are identified by
direct labels (id · name), so identity never rests on colour alone. One y-axis per panel; several measures → small
multiples.
"""
import textwrap

import numpy as np
import pandas as pd

TYPE_COLOR = {'BUG': '#2a78d6', 'FEATURE': '#eb6834', 'UX': '#1baf7a', 'mixed': '#8f8e89', 'outlier': '#d9d8d3'}
TYPE_LABEL = {'BUG': 'BUG', 'FEATURE': 'FEATURE', 'UX': 'UX', 'mixed': 'mixed', 'outlier': 'outlier'}
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _style():
    import matplotlib as mpl
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    # Times New Roman (thesis font) locally; on Kaggle/Linux the metric-compatible Liberation Serif, else DejaVu Serif —
    # all three carry the Vietnamese diacritics
    family = next(f for f in ('Times New Roman', 'Liberation Serif', 'DejaVu Serif') if f in fonts or f == 'DejaVu Serif')
    mpl.rcParams.update({
        'font.family': family,
        'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'axes.edgecolor': INK_2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.facecolor': 'white'})


def _num(x):
    return f'{int(x):,}'.replace(',', '.')


def _vn(x, decimals=None):
    """Vietnamese number format: '.' for thousands, ',' for decimals."""
    if decimals is None:
        decimals = 0 if float(x).is_integer() else len(f'{x:.6g}'.split('.')[-1]) if '.' in f'{x:.6g}' else 0
    s = f'{x:,.{decimals}f}'
    return s.replace(',', '\0').replace('.', ',').replace('\0', '.')


def _vn_axes(ax, which='both'):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: _vn(round(v, 6)))
    if which in ('both', 'x'):
        ax.xaxis.set_major_formatter(f)
    if which in ('both', 'y'):
        ax.yaxis.set_major_formatter(f)


def _save(fig, path):
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _type_handles(counts=None):
    from matplotlib.patches import Patch
    keys = [k for k in ('BUG', 'FEATURE', 'UX', 'mixed', 'outlier') if counts is None or counts.get(k, 0) > 0]
    return [Patch(facecolor=TYPE_COLOR[k], edgecolor='none',
                  label=TYPE_LABEL[k] if counts is None else f'{TYPE_LABEL[k]} ({_num(counts[k])})') for k in keys]


def topic_label(tid, name, width=38):
    name = str(name)
    return f'{tid} · ' + (name if len(name) <= width else name[:width - 1] + '…')


# ---------------------------------------------------------------- 1. Q12 grid
def fig_grid(grid, chosen, path):
    """Small multiples over min_topic_size: C_v, number of topics, outlier rate. Chosen value marked; degenerate
    configurations drawn hollow."""
    import matplotlib.pyplot as plt
    _style()
    panels = [('C_v', 'C_v', 1, 3), ('số topic', 'Số cụm', 1, 0), ('tỷ lệ outlier', 'Tỷ lệ outlier (%)', 100, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), constrained_layout=True)
    x = grid['min_topic_size'].to_numpy()
    for ax, (col, title, scale, fmt) in zip(axes, panels):
        y = grid[col].to_numpy() * scale
        ax.plot(x, y, color=TYPE_COLOR['BUG'], lw=2, zorder=2)
        deg = grid['suy biến'].to_numpy()
        ax.scatter(x[~deg], y[~deg], s=36, color=TYPE_COLOR['BUG'], edgecolor='white', linewidth=1.5, zorder=3)
        ax.scatter(x[deg], y[deg], s=36, facecolor='white', edgecolor=TYPE_COLOR['BUG'], linewidth=1.5, zorder=3)
        i = int(np.where(x == chosen)[0][0])
        ax.scatter([x[i]], [y[i]], s=110, facecolor='none', edgecolor=INK, linewidth=1.2, zorder=4)
        ax.annotate(_vn(y[i], fmt), (x[i], y[i]), textcoords='offset points', xytext=(0, 9), ha='center', color=INK)
        ax.set_title(title, loc='left', color=INK)
        ax.set_xticks(x)
        _vn_axes(ax, 'y')
        ax.set_xlabel('min_topic_size')
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        lo, hi = y.min(), y.max()
        pad = (hi - lo) * 0.25 or abs(hi) * 0.1 or 1
        ax.set_ylim(lo - pad, hi + pad)
    return _save(fig, path)


# ---------------------------------------------------------------- 2. document map
def fig_doc_map(embeddings, topics, dominant, path, seed=42):
    """2-D UMAP of review embeddings; points coloured by the dominant type of their topic, topic ids at the median of
    their points. dominant: {topic_id: 'BUG'|'FEATURE'|'UX'|'mixed'}. Returns (path, 2-D coordinates)."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from umap import UMAP
    _style()
    xy = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=seed).fit_transform(embeddings)
    t = np.asarray(topics)
    kind = np.array(['outlier' if x == -1 else dominant[int(x)] for x in t])
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    for k in ('outlier', 'mixed', 'UX', 'FEATURE', 'BUG'):
        m = kind == k
        ax.scatter(xy[m, 0], xy[m, 1], s=3 if k == 'outlier' else 4, color=TYPE_COLOR[k], alpha=0.35 if k == 'outlier' else 0.55,
                   linewidths=0, rasterized=True)
    for tid in sorted(set(t) - {-1}):
        cx, cy = np.median(xy[t == tid], axis=0)
        ax.text(cx, cy, str(tid), ha='center', va='center', fontsize=10, fontweight='bold', color=INK,
                path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    counts = pd.Series(kind).value_counts().to_dict()
    ax.legend(handles=_type_handles(counts), title='Loại chủ đạo của cụm (số review)', loc='upper left',
              bbox_to_anchor=(1.01, 1), frameon=False, alignment='left')
    lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)     # a few far-away points would
    pad = (hi - lo) * 0.05                                                          # squeeze the main cloud
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    n_hidden = int((~((xy >= lo - pad) & (xy <= hi + pad)).all(1)).sum())
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP-1' + (f' ({_num(n_hidden)} điểm xa nằm ngoài khung)' if n_hidden else '')); ax.set_ylabel('UMAP-2')
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    _save(fig, path)
    return path, xy


# ---------------------------------------------------------------- 3. size and type composition
def fig_composition(topics, names, path):
    """Horizontal stacked bars: reviews per topic split by Tier 2 type, largest topic on top; end label = size and
    dominant type with purity."""
    import matplotlib.pyplot as plt
    _style()
    d = topics[topics['topic_id'] != -1].sort_values('size', ascending=True)
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(d) + 1.2), constrained_layout=True)
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for k in ('BUG', 'FEATURE', 'UX'):
        v = d[f'n_{k}'].to_numpy()
        ax.barh(y, v, left=left, color=TYPE_COLOR[k], edgecolor='white', linewidth=1.5, height=0.72, label=k)
        left += v
    for yi, (_, r) in zip(y, d.iterrows()):
        dom = r['dominant_type']
        ax.text(r['size'] + left.max() * 0.01, yi, f"{_num(r['size'])} · {dom} {r['type_purity']:.0%}",
                va='center', ha='left', fontsize=8.5, color=INK_2)
    ax.set_yticks(y, [topic_label(int(r['topic_id']), names.get(int(r['topic_id']), r['keyword_name']), 44)
                      for _, r in d.iterrows()])
    ax.set_xlim(0, left.max() * 1.22)
    _vn_axes(ax, 'x')
    ax.set_xlabel('Số review (nhãn tầng 2 dự đoán)')
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, ncols=3)
    ax.tick_params(axis='y', length=0)
    return _save(fig, path)


# ---------------------------------------------------------------- 4. keywords
def fig_keywords(words_scores, names, dominant, path, n_words=8, ncols=4):
    """Small multiples: top c-TF-IDF words per topic, bars in the topic's dominant-type colour."""
    import matplotlib.pyplot as plt
    _style()
    ids = sorted(words_scores)
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.55 * ncols, 2.05 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, tid in zip(axes, ids):
        ws = [(w, s) for w, s in words_scores[tid] if w][:n_words][::-1]
        ax.barh([w for w, _ in ws], [s for _, s in ws], color=TYPE_COLOR[dominant[tid]], height=0.7)
        ax.set_title('\n'.join(textwrap.wrap(topic_label(tid, names.get(tid, ''), 60), 30)[:2]), loc='left', fontsize=9)
        ax.tick_params(axis='y', length=0, labelsize=8.5)
        ax.tick_params(axis='x', labelsize=7.5)
        _vn_axes(ax, 'x')
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[len(ids):]:
        ax.axis('off')
    fig.legend(handles=_type_handles()[:4], loc='outside lower center', ncols=4, frameon=False,
               title='Màu cột = loại chủ đạo của cụm; trục hoành = trọng số c-TF-IDF')
    return _save(fig, path)


# ---------------------------------------------------------------- 5. hierarchy
def fig_hierarchy(embeddings, topics, names, dominant, path):
    """Dendrogram of topic centroids (cosine distance, average linkage): which issues are close in meaning."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    from src.clustering.core import centroids
    _style()
    ids, c = centroids(embeddings, topics)
    z = linkage(pdist(c, metric='cosine'), method='average')
    labels = [f"{topic_label(t, names.get(t, ''), 46)} [{dominant[t]}]" for t in ids]
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(ids) + 1.0), constrained_layout=True)
    dendrogram(z, orientation='left', labels=labels, ax=ax, color_threshold=0, above_threshold_color=INK_2,
               leaf_font_size=9)
    _vn_axes(ax, 'x')
    ax.set_xlabel('Khoảng cách cosine giữa tâm cụm (liên kết trung bình)')
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    return _save(fig, path)
