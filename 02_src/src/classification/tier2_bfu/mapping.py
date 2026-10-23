"""Tier 2 data: gold-actionable reviews only, 3 classes (CLAUDE.md §2)."""
TIER2_LABELS = ['BUG', 'FEATURE', 'UX']


def to_tier2(df, label_col='gold_label', drop_ambiguous=True):
    """Keeps BUG/FEATURE/UX reviews and adds integer column `y`. For training, ambiguous reviews are
    dropped (protocol: not used to train Tier 2 in the first round); for evaluation keep them
    (drop_ambiguous=False) — every Validation/Hold-out review has one gold label."""
    out = df[df[label_col].isin(TIER2_LABELS)].copy()
    if drop_ambiguous and 'ambiguous_flag' in out.columns:
        out = out[out['ambiguous_flag'].astype(str) != '1']
    out['y'] = out[label_col].map({l: i for i, l in enumerate(TIER2_LABELS)})
    return out
