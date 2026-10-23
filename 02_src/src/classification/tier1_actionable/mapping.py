"""Tier 1 labels derived from the 4-class gold (CLAUDE.md §2): ACTIONABLE = BUG ∪ FEATURE ∪ UX."""
ACTIONABLE_CLASSES = ('BUG', 'FEATURE', 'UX')
TIER1_LABELS = ['NOT_ACTIONABLE', 'ACTIONABLE']   # index = class id; 1 = positive class


def to_tier1(df, label_col='gold_label'):
    """All reviews (ambiguous included: they are valid ACTIONABLE for Tier 1). Adds integer column `y`."""
    out = df.copy()
    out['y'] = out[label_col].isin(ACTIONABLE_CLASSES).astype(int)
    return out
