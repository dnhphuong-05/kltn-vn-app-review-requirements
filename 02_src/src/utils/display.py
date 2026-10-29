"""Bordered tables for notebook output (Jupyter, VS Code, Kaggle).

Every cell is ruled, the header row is shaded, numbers use the Vietnamese format (',' decimal, '.' thousands) and text
columns can wrap. Rendering only — the DataFrame itself is never changed, so saved CSVs keep plain numbers.
"""
import pandas as pd

_RULE = '1px solid #8f8e89'
TABLE_CSS = [
    {'selector': '', 'props': [('border-collapse', 'collapse'), ('font-size', '12.5px'), ('margin', '4px 0 14px 0')]},
    {'selector': 'th, td', 'props': [('border', _RULE), ('padding', '4px 8px'), ('vertical-align', 'top')]},
    {'selector': 'th', 'props': [('background-color', '#f0efec'), ('color', '#0b0b0b'), ('font-weight', 'bold'),
                                 ('text-align', 'center')]},
    {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-weight', 'bold'), ('text-align', 'left'),
                                      ('padding', '2px 0 6px 0')]},
]


def _pct(v, decimals=1):
    return '' if pd.isna(v) else f'{v * 100:.{decimals}f}%'.replace('.', ',')


def table(df, caption=None, decimals=3, percent=(), wrap=(), wrap_width='320px'):
    """Styler with ruled cells. percent: columns holding shares (0–1) shown as '12,3%'; wrap: long text columns."""
    sty = (df.style.hide(axis='index').set_table_styles(TABLE_CSS)
           .format(precision=decimals, decimal=',', thousands='.', na_rep=''))
    percent = [c for c in percent if c in df.columns]
    if percent:
        sty = sty.format(_pct, subset=percent)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if num:
        sty = sty.set_properties(subset=num, **{'text-align': 'right'})
    txt = [c for c in df.columns if c not in num]
    if txt:
        sty = sty.set_properties(subset=txt, **{'text-align': 'left'})
    wrap = [c for c in wrap if c in df.columns]
    if wrap:
        sty = sty.set_properties(subset=wrap, **{'white-space': 'normal', 'min-width': wrap_width})
    if caption:
        sty = sty.set_caption(caption)
    return sty


def show(df, caption=None, **kw):
    from IPython.display import display
    display(table(df, caption, **kw))
