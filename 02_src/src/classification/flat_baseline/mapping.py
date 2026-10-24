"""Flat 4-class baseline data (reference only, not used downstream — CLAUDE.md §3).

All reviews, one 4-class label each (ambiguous rows included: every review has one gold label). Class ids
match `src.classification.evaluate.FOUR_CLASSES` (BUG=0, FEATURE=1, UX=2, OTHER=3)."""
from src.classification.evaluate import FOUR_CLASSES

FLAT_LABELS = FOUR_CLASSES


def to_flat(df, label_col='gold_label'):
    out = df[df[label_col].isin(FLAT_LABELS)].copy()
    if len(out) != len(df):
        raise ValueError(f'{len(df) - len(out)} review có nhãn ngoài {FLAT_LABELS}')
    out['y'] = out[label_col].map({l: i for i, l in enumerate(FLAT_LABELS)})
    return out
