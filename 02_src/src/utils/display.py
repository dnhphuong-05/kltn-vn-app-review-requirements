"""Tables for notebook output (Jupyter, VS Code, Kaggle), ready to screenshot into the thesis.

Two looks:
  - 'booktabs' (default): academic-paper table — white background, black serif text, three horizontal rules (top, under
    the header, bottom), no vertical lines; forced white even under a dark notebook theme so screenshots stay clean;
  - 'grid': every cell ruled, header shaded.
Numbers use the Vietnamese format (',' decimal, '.' thousands). Rendering only — the DataFrame is never changed, so saved
CSVs keep plain numbers.
"""
import pandas as pd

_SERIF = "'Times New Roman', Times, 'Liberation Serif', serif"
_W = '#ffffff !important'          # notebook themes stripe / darken rows: force white
_K = '#000000 !important'

BOOKTABS_CSS = [
    {'selector': '', 'props': [('border-collapse', 'collapse'), ('font-family', _SERIF), ('font-size', '14px'),
                               ('color', _K), ('background-color', _W), ('border-top', '1.6px solid #000'),
                               ('border-bottom', '1.6px solid #000'), ('margin', '6px 0 16px 0')]},
    {'selector': 'thead th', 'props': [('border-bottom', '0.8px solid #000'), ('font-weight', 'normal'), ('color', _K),
                                       ('background-color', _W), ('padding', '5px 16px 5px 6px'), ('text-align', 'left'),
                                       ('font-family', _SERIF)]},
    {'selector': 'tbody td', 'props': [('border', 'none'), ('color', _K), ('background-color', _W),
                                       ('padding', '3px 16px 3px 6px'), ('text-align', 'left'), ('font-family', _SERIF)]},
    {'selector': 'tbody tr, tbody tr:nth-child(odd), tbody tr:nth-child(even), tbody tr:hover',
     'props': [('background-color', _W)]},
    {'selector': 'caption', 'props': [('caption-side', 'top'), ('text-align', 'left'), ('font-family', _SERIF),
                                      ('font-style', 'italic'), ('font-size', '14px'), ('color', _K),
                                      ('background-color', _W), ('padding', '4px 6px 6px 6px')]},
]
_RULE = '1px solid #8f8e89'
GRID_CSS = [
    {'selector': '', 'props': [('border-collapse', 'collapse'), ('font-size', '12.5px'), ('margin', '4px 0 14px 0')]},
    {'selector': 'th, td', 'props': [('border', _RULE), ('padding', '4px 8px'), ('vertical-align', 'top')]},
    {'selector': 'th', 'props': [('background-color', '#f0efec'), ('color', '#0b0b0b'), ('font-weight', 'bold'),
                                 ('text-align', 'center')]},
    {'selector': 'caption', 'props': [('caption-side', 'top'), ('font-weight', 'bold'), ('text-align', 'left'),
                                      ('padding', '2px 0 6px 0')]},
]
TABLE_CSS = BOOKTABS_CSS


def _pct(v, decimals=1):
    return '' if pd.isna(v) else f'{v * 100:.{decimals}f}%'.replace('.', ',')


def table(df, caption=None, decimals=3, percent=(), wrap=(), wrap_width='320px', look='booktabs'):
    """Styler for display. percent: columns holding shares (0–1) shown as '12,3%'; wrap: long text columns."""
    sty = (df.style.hide(axis='index').set_table_styles(BOOKTABS_CSS if look == 'booktabs' else GRID_CSS)
           .format(precision=decimals, decimal=',', thousands='.', na_rep=''))
    percent = [c for c in percent if c in df.columns]
    if percent:
        sty = sty.format(_pct, subset=percent)
    if look == 'grid':
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
