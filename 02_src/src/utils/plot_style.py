"""Shared matplotlib look (font, ink colours, Vietnamese number format, validated palette); used by the clustering figures.

Categorical slots 1–3 (blue, orange, aqua) pass the all-pairs colour-vision check, so a figure never colours more than three
entities; a fourth is a neutral grey. What the colours stand for is fixed per figure family:
  - strategies: ViSoBERT = slot 1, Bi-LSTM + fastText = slot 2, LLM few-shot = slot 3, baseline = grey;
  - Tier 2 types: BUG = slot 1, FEATURE = slot 2, UX = slot 3, mixed / outlier = greys;
  - apps: Shopee = slot 1, Lazada = slot 2.
"""
import numpy as np

SLOT = ['#2a78d6', '#eb6834', '#1baf7a']
GREY, GREY_LIGHT = '#8f8e89', '#d9d8d3'
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ_BLUE = ['#f4f8fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#1c5cab', '#104281']

STRATEGY_COLOR = {'ViSoBERT': SLOT[0], 'Bi-LSTM + fastText': SLOT[1], 'LLM few-shot': SLOT[2], 'Baseline phẳng': GREY}
APP_COLOR = {'Shopee': SLOT[0], 'Lazada': SLOT[1]}


def style():
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


def num(x):
    """Integer with '.' for thousands (Vietnamese)."""
    return f'{int(round(x)):,}'.replace(',', '.')


def vn(x, decimals=None):
    """Vietnamese number format: '.' for thousands, ',' for decimals."""
    if decimals is None:
        s6 = f'{x:.6g}'
        decimals = 0 if float(x).is_integer() or 'e' in s6 or '.' not in s6 else len(s6.split('.')[-1])
    s = f'{x:,.{decimals}f}'
    return s.replace(',', '\0').replace('.', ',').replace('\0', '.')


def pct(x, decimals=1):
    return vn(x * 100, decimals) + '%'


def vn_axes(ax, which='both', percent=False):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: (vn(round(v * 100, 6)) + '%') if percent else vn(round(v, 6)))
    if which in ('both', 'x'):
        ax.xaxis.set_major_formatter(f)
    if which in ('both', 'y'):
        ax.yaxis.set_major_formatter(f)


def grid_lines(ax, axis='y'):
    (ax.yaxis if axis == 'y' else ax.xaxis).grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    import matplotlib.pyplot as plt
    fig.savefig(path)
    plt.close(fig)
    return path


def text_color_on(fill_hex):
    """Ink that stays readable on a sequential cell (white on dark steps)."""
    h = fill_hex.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return '#ffffff' if lum < 0.5 else INK


def seq_cmap():
    """Sequential single-hue blue for heatmaps: matplotlib 'Blues' (near-white → navy)."""
    import matplotlib.pyplot as plt
    return plt.get_cmap('Blues')


def seq_color(v, vmin, vmax):
    return seq_cmap()(np.clip((v - vmin) / (vmax - vmin + 1e-12), 0, 1))
