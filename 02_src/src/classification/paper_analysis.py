"""Paper, step 1 (amendment v3.1): re-analysis of the saved GĐ10 predictions.

No retraining, no API call, no change to any frozen config, prompt or threshold. The module covers:
  - a text-level duplicate audit between the splits (exact match after normalisation; near-duplicates reported only);
  - rebuilding Tier 1 decisions, Tier 2 labels and 4-class predictions of every configuration (3 cascades, flat
    ViSoBERT, hybrid 3×3) from the saved probabilities and LLM labels with the frozen thresholds, exactly as notebook
    10 does, so the same predictions can be re-scored on a subset (the de-duplicated sets);
  - exact McNemar tests with Holm correction;
  - per-class errors, an error decomposition by stage (missed actionable / kept OTHER / wrong type) and the
    cost–quality Pareto front with its figure.
"""
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.classification.evaluate import FOUR_CLASSES, OTHER_ID, cascade_labels, confusion, metrics
from src.classification.final_eval import bootstrap_table, llm_ids, neural_probs, pm, point_metric
from src.classification.tier2_bfu.mapping import TIER2_LABELS

NEURAL = {'ViSoBERT': 'visobert', 'Bi-LSTM + fastText': 'bilstm'}
LLM = 'LLM few-shot'
STRATEGIES = list(NEURAL) + [LLM]
FLAT = 'Baseline phẳng ViSoBERT'
SHORT = {'ViSoBERT': 'V', 'Bi-LSTM + fastText': 'B', LLM: 'L', 'TF-IDF': 'T', 'PhoBERT': 'P'}
FLAT_PREFIX = 'Phẳng '
ORDER = ['TF-IDF', 'Bi-LSTM + fastText', 'ViSoBERT', 'PhoBERT', LLM]       # step 2: traditional → LLM


# ---------------------------------------------------------------- duplicates
def normalize_text(s):
    """NFC, lower case, punctuation → space, collapsed whitespace."""
    s = unicodedata.normalize('NFC', str(s)).lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def duplicate_audit(sets: dict, pairs, label_col='gold_label', other='OTHER'):
    """Exact duplicates after normalisation. sets: {name: DataFrame with text + gold_label}; pairs: [(reference, target)].
    One row per pair: distinct shared texts, target rows whose text occurs in the reference, how many of those rows are
    actionable, their longest length in words, and the share whose label equals the reference label of the same text
    (most frequent label when the reference holds the text several times)."""
    norm = {k: df['text'].map(normalize_text) for k, df in sets.items()}
    rows = []
    for ref, tgt in pairs:
        shared = set(norm[ref]) & set(norm[tgt])
        hit = norm[tgt].isin(shared).to_numpy()
        lab_t = sets[tgt][label_col].to_numpy()[hit]
        ref_label = pd.DataFrame({'n': norm[ref], 'l': sets[ref][label_col]}).groupby('n')['l'].agg(lambda x: x.mode().iloc[0])
        same = float((lab_t == norm[tgt][hit].map(ref_label).to_numpy()).mean()) if hit.any() else np.nan
        rows.append({'tập tham chiếu': ref, 'tập kiểm tra': tgt, 'n tập kiểm tra': len(sets[tgt]),
                     'văn bản chung': len(shared), 'dòng trùng': int(hit.sum()), 'tỷ lệ dòng trùng': float(hit.mean()),
                     'dòng trùng actionable': int((lab_t != other).sum()),
                     'số chữ tối đa': int(norm[tgt][hit].str.split().str.len().max()) if hit.any() else 0,
                     'nhãn khớp tập tham chiếu': same})
    return pd.DataFrame(rows)


def within_duplicates(sets: dict):
    """Duplicates inside each set: rows in a group of identical normalised texts, distinct repeated texts, and rows
    that would go if one copy per text were kept."""
    rows = []
    for k, df in sets.items():
        n = df['text'].map(normalize_text)
        grp = n.duplicated(keep=False)
        rows.append({'tập': k, 'n': len(df), 'dòng thuộc nhóm trùng': int(grp.sum()),
                     'văn bản lặp': int(n[grp].nunique()), 'dòng thừa': int(n.duplicated().sum())})
    return pd.DataFrame(rows)


def top_shared(ref_df, tgt_df, k=10):
    """Most frequent texts of the target that also occur in the reference (normalised form, counts in both)."""
    r, t = ref_df['text'].map(normalize_text), tgt_df['text'].map(normalize_text)
    shared = t[t.isin(set(r))]
    out = shared.value_counts().head(k).rename('số dòng ở tập kiểm tra').to_frame()
    out['số dòng ở tập tham chiếu'] = [int((r == x).sum()) for x in out.index]
    out.index.name = 'văn bản (chuẩn hóa)'
    return out.reset_index()


def near_duplicates(ref_texts, tgt_texts, threshold=0.90):
    """Closest reference text of every target row by cosine on character 3–5-gram TF-IDF of the normalised text.
    near = cosine ≥ threshold without an exact match (reported only, never removed)."""
    r = pd.Series(list(ref_texts)).map(normalize_text)
    t = pd.Series(list(tgt_texts)).map(normalize_text)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5)).fit(pd.concat([r, t]))
    best = np.asarray((vec.transform(t) @ vec.transform(r).T).max(axis=1).todense()).ravel()
    exact = t.isin(set(r)).to_numpy()
    return pd.DataFrame({'max_cosine': best, 'exact': exact, 'near': (best >= threshold) & ~exact})


def keep_mask(target_texts, reference_texts):
    """True for target rows whose normalised text does not occur in the reference (the de-duplicated subset)."""
    ref = set(pd.Series(list(reference_texts)).map(normalize_text))
    return ~pd.Series(list(target_texts)).map(normalize_text).isin(ref).to_numpy()


# ---------------------------------------------------------------- saved predictions → configurations
def canonical(runs):
    """Seed 42 for neural runs, otherwise the single run (LLM, hybrid)."""
    if 42 in runs:
        return runs[42]
    if len(runs) != 1:
        raise RuntimeError(f'không xác định được lần chạy chuẩn: {list(runs)}')
    return next(iter(runs.values()))


def saved_predictions(res_dir, set_name, ids, frozen):
    """(tier1, tier2, flat) of a locked set rebuilt from the GĐ10 files as in notebook 12: neural Tier 1 =
    P(ACTIONABLE) ≥ frozen threshold per seed, Tier 2 / flat = argmax, LLM = saved labels (Tier 2: −1 where the LLM
    was not called). tier1/tier2: {strategy: {run: array}}; flat: {seed: array} or {} when not saved."""
    pred = pd.read_csv(os.path.join(res_dir, f'gd10_pred_{set_name}.csv'))
    seeds = frozen['tier1']['seeds']
    have = set(zip(pred['tier'], pred['model']))
    t1, t2, flat = {}, {}, {}
    for name, m in NEURAL.items():
        if ('tier1', m) in have:
            thr = frozen['tier1'][m]['threshold']
            t1[name] = {s: p[:, 1] >= thr for s, p in neural_probs(pred, ids, 'tier1', m, seeds).items()}
        if ('tier2', m) in have:
            t2[name] = {s: p.argmax(1) for s, p in neural_probs(pred, ids, 'tier2', m, seeds).items()}
    if ('flat', 'visobert') in have:
        flat = {s: p.argmax(1) for s, p in neural_probs(pred, ids, 'flat', 'visobert', seeds).items()}
    for tier, store in ((1, t1), (2, t2)):
        f = os.path.join(res_dir, f'gd10_llm_{set_name}_tier{tier}.csv')
        if not os.path.exists(f):
            continue
        a = llm_ids(ids, pd.read_csv(f))
        if tier == 1:
            if (a < 0).any():
                raise RuntimeError(f'{set_name}: LLM tầng 1 thiếu {(a < 0).sum()} review')
            a = a == 1
        store[LLM] = {'llm': a}
    return t1, t2, flat


def _cascade(act, t2):
    act, t2 = np.asarray(act, dtype=bool), np.asarray(t2)
    gap = int(((t2 < 0) & act).sum())
    return (None, gap) if gap else (cascade_labels(act, np.where(t2 < 0, 0, t2)), 0)


def hybrid_name(x, y):
    return f'Cascade {x}' if x == y else f'{x} → {y}'


def configurations(t1, t2, flat, hybrids=True):
    """4-class predictions {name: {run: ids}}: pure cascades (every run), flat ViSoBERT (every seed) and, with hybrids,
    Tier 1 of X × Tier 2 of Y for X ≠ Y on the canonical runs. A Tier 2 gap on a review that Tier 1 kept raises for a
    pure cascade; for a hybrid the pair is skipped and listed. Returns (configs, skipped)."""
    out, skipped = {}, []
    for x in STRATEGIES:
        if x in t1 and x in t2:
            runs = {}
            for k, act in t1[x].items():
                p, gap = _cascade(act, t2[x][k])
                if gap:
                    raise RuntimeError(f'Cascade {x}/{k}: tầng 2 thiếu {gap} dự đoán cho review tầng 1 giữ lại')
                runs[k] = p
            out[hybrid_name(x, x)] = runs
    if flat:
        out[FLAT] = dict(flat)
    if hybrids:
        for x in STRATEGIES:
            for y in STRATEGIES:
                if x == y or x not in t1 or y not in t2:
                    continue
                p, gap = _cascade(canonical(t1[x]), canonical(t2[y]))
                if gap:
                    skipped.append({'tầng 1': x, 'tầng 2': y, 'review thiếu tầng 2': gap})
                else:
                    out[hybrid_name(x, y)] = {'canon': p}
    return out, skipped


def short_label(name, flat='V phẳng'):
    """V→L style code of a configuration (V = ViSoBERT, B = Bi-LSTM + fastText, L = LLM, T = TF-IDF, P = PhoBERT).
    Flat configurations read '<code> <word>' with the last word of `flat` ('phẳng' or 'flat')."""
    if name == FLAT:
        return flat
    if name.startswith(FLAT_PREFIX):
        return f'{SHORT[name[len(FLAT_PREFIX):]]} {flat.split()[-1]}'
    if name.startswith('Cascade '):
        s = SHORT[name[len('Cascade '):]]
        return f'{s}→{s}'
    x, y = name.split(' → ')
    return f'{SHORT[x]}→{SHORT[y]}'


# ---------------------------------------------------------------- scoring on a (sub)set
def _mask(n, mask):
    return np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)


def tier1_scores(y4, t1, mask=None, n_boot=10_000, seed=42):
    """F1 / recall / precision of ACTIONABLE and 2-class Macro-F1 (mean ± std over runs, as gd10_tier1_holdout.csv),
    plus point value and 95% CI of F1 and recall on the canonical run."""
    m = _mask(len(y4), mask)
    y1 = (np.asarray(y4)[m] != OTHER_ID).astype(int)
    rows, canon = [], {}
    for name, runs in t1.items():
        v = {met: [point_metric(y1, a[m].astype(int), 2, met, None if met == 'macro_f1' else 1) for a in runs.values()]
             for met in ('f1', 'recall', 'precision', 'macro_f1')}
        rows.append({'chiến lược': name, 'n': int(m.sum()), 'F1 ACT': pm(v['f1']), 'Recall ACT': pm(v['recall']),
                     'Precision ACT': pm(v['precision']), 'Macro-F1 (2 lớp)': pm(v['macro_f1']), 'số lần chạy': len(runs),
                     'F1 ACT (trung bình)': float(np.mean(v['f1'])), 'Recall ACT (trung bình)': float(np.mean(v['recall']))})
        canon[name] = canonical(runs)[m].astype(int)
    tab = pd.DataFrame(rows)
    for met in ('f1', 'recall'):
        ci, _ = bootstrap_table(y1, canon, 2, met, 1, n_boot, seed)
        ci = ci.set_index('cấu hình')
        tab[f'{met} chuẩn'] = tab['chiến lược'].map(ci['giá trị'])
        tab[f'{met} CI'] = tab['chiến lược'].map(lambda s: f"{ci.loc[s, 'CI 2,5%'] * 100:.1f}–{ci.loc[s, 'CI 97,5%'] * 100:.1f}")
    return tab, y1, canon


def tier2_scores(y4, t2, mask=None, n_boot=10_000, seed=42):
    """Macro-F1 (3 classes) and per-class F1 on gold-actionable reviews (mean ± std over runs, as
    gd10_tier2_holdout.csv), plus canonical point value and 95% CI of Macro-F1."""
    y4 = np.asarray(y4)
    m = _mask(len(y4), mask) & (y4 != OTHER_ID)
    y2 = y4[m]
    rows, canon = [], {}
    for name, runs in t2.items():
        ps = [np.asarray(p)[m] for p in runs.values()]
        if any((p < 0).any() for p in ps):
            raise RuntimeError(f'{name}: tầng 2 thiếu dự đoán trên gold-actionable')
        ms = [metrics(y2, p, TIER2_LABELS) for p in ps]
        rows.append({'chiến lược': name, 'n': int(m.sum()), 'Macro-F1': pm([x['macro_f1'] for x in ms]),
                     **{f'F1 {c}': pm([x[f'f1_{c}'] for x in ms]) for c in TIER2_LABELS}, 'số lần chạy': len(runs),
                     'Macro-F1 (trung bình)': float(np.mean([x['macro_f1'] for x in ms]))})
        canon[name] = canonical(runs)[m]
    tab = pd.DataFrame(rows)
    ci, _ = bootstrap_table(y2, canon, 3, n_boot=n_boot, seed=seed)
    ci = ci.set_index('cấu hình')
    tab['Macro-F1 chuẩn'] = tab['chiến lược'].map(ci['giá trị'])
    tab['Macro-F1 CI'] = tab['chiến lược'].map(lambda s: f"{ci.loc[s, 'CI 2,5%'] * 100:.1f}–{ci.loc[s, 'CI 97,5%'] * 100:.1f}")
    return tab, y2, canon


def e2e_scores(y4, configs, mask=None, n_boot=10_000, seed=42):
    """Macro-F1 (4 classes) and per-class F1 (mean ± std over runs, as gd10_e2e_holdout.csv), canonical point value
    with 95% CI, and the paired bootstrap differences between configurations. Returns (table, diffs, y, canon)."""
    m = _mask(len(y4), mask)
    y = np.asarray(y4)[m]
    rows, canon = [], {}
    for name, runs in configs.items():
        ms = [metrics(y, np.asarray(p)[m], FOUR_CLASSES) for p in runs.values()]
        rows.append({'cấu hình': name, 'n': int(m.sum()), 'Macro-F1 (4 lớp)': pm([x['macro_f1'] for x in ms]),
                     **{f'F1 {c}': pm([x[f'f1_{c}'] for x in ms]) for c in FOUR_CLASSES}, 'số lần chạy': len(runs),
                     'Macro-F1 (trung bình)': float(np.mean([x['macro_f1'] for x in ms]))})
        canon[name] = np.asarray(canonical(runs))[m]
    tab = pd.DataFrame(rows)
    ci, diff = bootstrap_table(y, canon, 4, n_boot=n_boot, seed=seed)
    ci = ci.set_index('cấu hình')
    tab['Macro-F1 chuẩn'] = tab['cấu hình'].map(ci['giá trị'])
    tab['CI 2,5%'] = tab['cấu hình'].map(ci['CI 2,5%'])
    tab['CI 97,5%'] = tab['cấu hình'].map(ci['CI 97,5%'])
    return tab, diff, y, canon


# ---------------------------------------------------------------- McNemar
def mcnemar_exact(y, a, b):
    """Exact (two-sided binomial) McNemar test on per-review correctness. Returns (A right & B wrong, A wrong & B
    right, p)."""
    ca, cb = np.asarray(a) == y, np.asarray(b) == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def holm(pvalues):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar_table(y, preds: dict, family):
    """Every pair of configurations, Holm-adjusted within the family."""
    names, rows = list(preds), []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n01, n10, p = mcnemar_exact(y, preds[a], preds[b])
            rows.append({'họ so sánh': family, 'A': a, 'B': b, 'tỷ lệ đúng A': float((preds[a] == y).mean()),
                         'tỷ lệ đúng B': float((preds[b] == y).mean()), 'A đúng, B sai': n01, 'A sai, B đúng': n10,
                         'p (exact)': p})
    df = pd.DataFrame(rows)
    df['p (Holm)'] = holm(df['p (exact)'])
    df['khác biệt (α = 0,05)'] = df['p (Holm)'] < 0.05
    return df


# ---------------------------------------------------------------- errors
def per_class(y, configs, labels=FOUR_CLASSES, mask=None):
    """Support, precision, recall and F1 per class (mean ± std over runs) for every configuration."""
    m = _mask(len(y), mask)
    yy = np.asarray(y)[m]
    rows = []
    for name, runs in configs.items():
        ms = [metrics(yy, np.asarray(p)[m], labels) for p in runs.values()]
        for i, c in enumerate(labels):
            rows.append({'cấu hình': name, 'lớp': c, 'support': int((yy == i).sum()),
                         'precision': pm([x[f'precision_{c}'] for x in ms]), 'recall': pm([x[f'recall_{c}'] for x in ms]),
                         'F1': pm([x[f'f1_{c}'] for x in ms])})
    return pd.DataFrame(rows)


ERROR_KINDS = ('bỏ sót actionable', 'giữ nhầm OTHER', 'sai loại')


def error_decomposition(y, configs, mask=None):
    """End-to-end errors split by stage, mean over runs: gold BUG/FEATURE/UX predicted OTHER (missed actionable),
    gold OTHER predicted BUG/FEATURE/UX (kept OTHER), gold and prediction both actionable but different (wrong type)."""
    m = _mask(len(y), mask)
    yy = np.asarray(y)[m]
    rows = []
    for name, runs in configs.items():
        c = []
        for p in runs.values():
            p = np.asarray(p)[m]
            g_act, p_act = yy != OTHER_ID, p != OTHER_ID
            c.append(((g_act & ~p_act).sum(), (~g_act & p_act).sum(), (g_act & p_act & (p != yy)).sum()))
        c = np.asarray(c, dtype=float).mean(axis=0)
        tot = c.sum()
        rows.append({'cấu hình': name, 'n': int(m.sum()), 'tổng lỗi': round(tot, 1),
                     **{k: round(v, 1) for k, v in zip(ERROR_KINDS, c)},
                     **{f'% {k}': round(100 * v / tot, 1) if tot else 0.0 for k, v in zip(ERROR_KINDS, c)}})
    return pd.DataFrame(rows)


def confusion_long(y, configs, labels=FOUR_CLASSES):
    """Confusion matrix (gold × predicted) of the canonical run of every configuration, stacked."""
    frames = []
    for name, runs in configs.items():
        cm = confusion(np.asarray(y), np.asarray(canonical(runs)), labels).reset_index().rename(columns={'index': 'gold'})
        cm.insert(0, 'cấu hình', name)
        frames.append(cm)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- Pareto
def pareto_front(quality, cost):
    """True where no other configuration has quality ≥ and cost ≤ with at least one strict."""
    q, c = np.asarray(quality, dtype=float), np.asarray(cost, dtype=float)
    return np.array([not (((q >= q[i]) & (c <= c[i]) & ((q > q[i]) | (c < c[i]))).any()) for i in range(len(q))])


INK, INK2, MUTED, GRID, AXIS, ACCENT = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#2a78d6'
MARKER = {'cascade': 'o', 'phẳng': 's', 'hybrid': '^'}


def architecture(name):
    if name == FLAT or name.startswith(FLAT_PREFIX):
        return 'phẳng'
    return 'cascade' if name.startswith('Cascade ') else 'hybrid'


def pareto_figure(df, path, panels, text, label_col=None):
    """Two panels sharing the Macro-F1 axis, one per cost measure; 95% CI bars on the Pareto points. df: one row per configuration with columns
    'cấu hình', 'Macro-F1', 'CI 2,5%', 'CI 97,5%' (fractions) and the cost columns named in panels.
    panels: [(cost column, x-axis title, log scale?, Pareto column)]. text: legend and y-axis strings.
    label_col: None labels every point; a column name (or one per panel) labels the Pareto points plus the rows
    where that column is True. Labels of Pareto points alternate above and below the front to avoid overlaps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.3), sharey=True)
    label_cols = label_col if isinstance(label_col, (list, tuple)) else [label_col] * len(panels)
    for ax, (col, xtitle, log, pcol), label_col in zip(np.atleast_1d(axes), panels, label_cols):
        d = df.copy()
        d['x'] = d[col].astype(float)
        if log:
            ax.set_xscale('log')
        front = d[d[pcol]].sort_values('x')
        ax.plot(front['x'], front['Macro-F1'] * 100, color=ACCENT, lw=1.5, zorder=2)
        labels = {}
        for _, r in d.iterrows():
            on = bool(r[pcol])
            color = ACCENT if on else MUTED
            y = r['Macro-F1'] * 100
            if on:                       # CIs of dominated points stack on the same x and are kept in the table only
                ax.errorbar(r['x'], y, yerr=[[y - r['CI 2,5%'] * 100], [r['CI 97,5%'] * 100 - y]], fmt='none',
                            ecolor=color, elinewidth=1, alpha=0.55, zorder=1)
            ax.scatter(r['x'], y, s=46 if on else 34, marker=MARKER[architecture(r['cấu hình'])], color=color,
                       edgecolors='white', linewidths=1.2, zorder=3)
            if label_col is not None and not (on or bool(r[label_col])):
                continue                 # unlabeled dominated point (crowded charts)
            key = (round(r['x'], 4), round(y, 1))
            labels.setdefault(key, []).append((short_label(r['cấu hình'], text.get('flat_code', 'V phẳng')), on))
        rank = {(round(x, 4), round(f * 100, 1)): i for i, (x, f) in enumerate(zip(front['x'], front['Macro-F1']))}
        for (x, y), items in labels.items():
            on = any(o for _, o in items)
            dy = -11 if on and label_col is not None and rank.get((x, y), 0) % 2 else 3
            ax.annotate(' / '.join(s for s, _ in items), (x, y), xytext=(6, dy), textcoords='offset points',
                        fontsize=8.5, color=INK if on else INK2, fontweight='bold' if on else 'normal')
        ax.set_xlabel(xtitle, color=INK2)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=INK2)
    np.atleast_1d(axes)[0].set_ylabel(text['y'], color=INK2)
    handles = [Line2D([], [], marker=MARKER[k], ls='none', color=INK2, label=text[k]) for k in ('cascade', 'phẳng', 'hybrid')]
    handles += [Line2D([], [], marker='o', ls='-', color=ACCENT, label=text['front']),
                Line2D([], [], marker='o', ls='none', color=MUTED, label=text['dominated'])]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ---------------------------------------------------------------- step 2: five strategies (amendments v3.2–v3.5)
def _tier_runs(pred, ids, tier, model, seeds, threshold=None):
    probs = neural_probs(pred, ids, tier, model, seeds)
    if tier == 'tier1':
        return {s: p[:, 1] >= threshold for s, p in probs.items()}
    return {s: p.argmax(1) for s, p in probs.items()}


def step2_predictions(res_dir, set_name, ids, frozen, tfidf_cfg, neural_cfg, phobert_threshold=None):
    """(tier1, tier2, flats) of the five strategies in ORDER. tier1: {strategy: {run: bool}}; tier2 and flats:
    {strategy: {run: class ids}} (flats in the 4-class space). Sources: gd10_* (ViSoBERT, Bi-LSTM, LLM, flat
    ViSoBERT), paper_pred_tfidf_* (TF-IDF, threshold from its frozen config), paper_pred_neural_* (PhoBERT with its
    frozen threshold unless `phobert_threshold` is given; flat Bi-LSTM) and paper_flat_llm_* (flat LLM labels)."""
    t1, t2, flat_v = saved_predictions(res_dir, set_name, ids, frozen)
    seeds = frozen['tier1']['seeds']
    flats = {'ViSoBERT': flat_v} if flat_v else {}
    f = os.path.join(res_dir, f'paper_pred_tfidf_{set_name}.csv')
    if os.path.exists(f):
        pt = pd.read_csv(f)
        t1['TF-IDF'] = _tier_runs(pt, ids, 'tier1', 'tfidf', seeds, tfidf_cfg['tier1']['threshold'])
        t2['TF-IDF'] = _tier_runs(pt, ids, 'tier2', 'tfidf', seeds)
        flats['TF-IDF'] = _tier_runs(pt, ids, 'flat', 'tfidf', seeds)
    f = os.path.join(res_dir, f'paper_pred_neural_{set_name}.csv')
    if os.path.exists(f):
        pn = pd.read_csv(f)
        thr = neural_cfg['phobert']['tier1']['threshold'] if phobert_threshold is None else phobert_threshold
        t1['PhoBERT'] = _tier_runs(pn, ids, 'tier1', 'phobert', seeds, thr)
        t2['PhoBERT'] = _tier_runs(pn, ids, 'tier2', 'phobert', seeds)
        flats['PhoBERT'] = _tier_runs(pn, ids, 'flat', 'phobert', seeds)
        flats['Bi-LSTM + fastText'] = _tier_runs(pn, ids, 'flat', 'bilstm', seeds)
    f = os.path.join(res_dir, f'paper_flat_llm_{set_name}.csv')
    if os.path.exists(f):
        d = pd.read_csv(f)
        m, lab = dict(zip(d['review_id'], d['pred_label'])), {l: i for i, l in enumerate(FOUR_CLASSES)}
        a = np.array([lab.get(m.get(r), -1) for r in ids])
        if (a < 0).any():
            raise RuntimeError(f'{set_name}: flat LLM thiếu hoặc sai nhãn ở {(a < 0).sum()} review')
        flats[LLM] = {'llm': a}
    order = lambda dct: {k: dct[k] for k in ORDER if k in dct}
    return order(t1), order(t2), order(flats)


def configurations_step2(t1, t2, flats, hybrids=True):
    """Pure cascades and flats (every run), then hybrids X → Y (X ≠ Y, canonical runs), all in ORDER. A Tier 2 gap
    on a review kept by Tier 1 raises for a pure cascade and skips (and lists) a hybrid. Returns (configs, skipped)."""
    out, skipped = {}, []
    for x in ORDER:
        if x in t1 and x in t2:
            runs = {}
            for k, act in t1[x].items():
                p, gap = _cascade(act, t2[x][k])
                if gap:
                    raise RuntimeError(f'Cascade {x}/{k}: tầng 2 thiếu {gap} dự đoán cho review tầng 1 giữ lại')
                runs[k] = p
            out[hybrid_name(x, x)] = runs
    for x in ORDER:
        if x in flats:
            out[FLAT_PREFIX + x] = dict(flats[x])
    if hybrids:
        for x in ORDER:
            for y in ORDER:
                if x == y or x not in t1 or y not in t2:
                    continue
                p, gap = _cascade(canonical(t1[x]), canonical(t2[y]))
                if gap:
                    skipped.append({'tầng 1': x, 'tầng 2': y, 'review thiếu tầng 2': gap})
                else:
                    out[hybrid_name(x, y)] = {'canon': p}
    return out, skipped


def actionable_scores(y, configs, mask=None):
    """Recall, precision and F1 of ACTIONABLE (BUG ∪ FEATURE ∪ UX collapsed) of every configuration, mean over runs."""
    m = _mask(len(y), mask)
    g = np.asarray(y)[m] != OTHER_ID
    rows = []
    for name, runs in configs.items():
        r = []
        for p in runs.values():
            a = np.asarray(p)[m] != OTHER_ID
            tp = (a & g).sum()
            rec, prec = tp / max(g.sum(), 1), tp / max(a.sum(), 1)
            r.append((rec, prec, 2 * rec * prec / max(rec + prec, 1e-12)))
        r = np.asarray(r, dtype=float).mean(axis=0)
        rows.append({'cấu hình': name, 'Recall ACT': r[0], 'Precision ACT': r[1], 'F1 ACT': r[2]})
    return pd.DataFrame(rows)


def _pair(df, a, b):
    """Row of a pairwise table for the unordered pair {a, b} and the sign that orients it as a − b."""
    r = df[((df['A'] == a) & (df['B'] == b)) | ((df['A'] == b) & (df['B'] == a))].iloc[0]
    return r, (1 if r['A'] == a else -1)


def architecture_contrast(tab, diff, mc, act):
    """Cascade X against flat X for every strategy that has both (RQ2): mean Macro-F1 of each, paired bootstrap
    difference of the canonical runs with 95% CI, McNemar p (Holm within `mc`'s family) and recall ACTIONABLE."""
    t, a = tab.set_index('cấu hình'), act.set_index('cấu hình')
    rows = []
    for x in ORDER:
        c, f = hybrid_name(x, x), FLAT_PREFIX + x
        if c not in t.index or f not in t.index:
            continue
        d, sg = _pair(diff, c, f)
        lo, hi = sorted([sg * d['CI 2,5%'], sg * d['CI 97,5%']])
        m, _ = _pair(mc, c, f)
        rows.append({'họ mô hình': x, 'Macro-F1 cascade': t.loc[c, 'Macro-F1 (trung bình)'] * 100,
                     'Macro-F1 phẳng': t.loc[f, 'Macro-F1 (trung bình)'] * 100,
                     'chênh cascade − phẳng (lần chạy chuẩn)': sg * d['chênh (A − B)'] * 100, 'CI 2,5%': lo * 100,
                     'CI 97,5%': hi * 100, 'McNemar p (Holm)': m['p (Holm)'],
                     'Recall ACT cascade': a.loc[c, 'Recall ACT'] * 100, 'Recall ACT phẳng': a.loc[f, 'Recall ACT'] * 100})
    return pd.DataFrame(rows)


def config_cost(name, keep, sec, usd, train, seg=None):
    """(inference seconds, API USD, training seconds) of a configuration. sec/usd/train: {(position, strategy): value},
    position 'tier1' | 'tier2' | 'flat', sec and usd per 1,000 reviews, train per run; keep: Tier 1 keep rate per
    strategy; seg: segmentation seconds per 1,000 reviews (PhoBERT), counted once for every review the strategy reads."""
    seg = seg or {}
    if name == FLAT or name.startswith(FLAT_PREFIX):
        x = 'ViSoBERT' if name == FLAT else name[len(FLAT_PREFIX):]
        return sec[('flat', x)] + seg.get(x, 0.0), usd[('flat', x)], train[('flat', x)]
    x, y = (name[len('Cascade '):],) * 2 if name.startswith('Cascade ') else name.split(' → ')
    s = sec[('tier1', x)] + seg.get(x, 0.0) + keep[x] * (sec[('tier2', y)] + (0.0 if y == x else seg.get(y, 0.0)))
    return s, usd[('tier1', x)] + keep[x] * usd[('tier2', y)], train[('tier1', x)] + train[('tier2', y)]


# ---------------------------------------------------------------- step 3a: prompt variants of the LLM (amendment v3.6)
def llm_variant(ids, t1_df, t2_df, flat_df, tag):
    """Cascade and flat 4-class predictions of one prompt variant from saved per-review LLM results (column `pred`).
    Returns (configs {'Cascade LLM – tag', 'Phẳng LLM – tag'}, tier1 bool, tier2 ids with −1 where not called)."""
    t1, t2, fl = llm_ids(ids, t1_df), llm_ids(ids, t2_df), llm_ids(ids, flat_df)
    if (t1 < 0).any() or (fl < 0).any():
        raise RuntimeError(f'{tag}: tầng 1 hoặc phẳng thiếu dự đoán')
    cas, gap = _cascade(t1 == 1, t2)
    if gap:
        raise RuntimeError(f'{tag}: tầng 2 thiếu {gap} dự đoán cho review tầng 1 giữ lại')
    return {f'Cascade LLM – {tag}': {'llm': cas}, f'Phẳng LLM – {tag}': {'llm': fl}}, t1 == 1, t2


def variant_differences(diff, mc, pairs):
    """Paired bootstrap difference (a − b, %) with 95% CI and McNemar p (Holm) for each (a, b) in pairs."""
    rows = []
    for a, b in pairs:
        d, sg = _pair(diff, a, b)
        m, _ = _pair(mc, a, b)
        lo, hi = sorted([sg * d['CI 2,5%'], sg * d['CI 97,5%']])
        rows.append({'so sánh': f'{a} − {b}', 'chênh Macro-F1': sg * d['chênh (A − B)'] * 100, 'CI 2,5%': lo * 100,
                     'CI 97,5%': hi * 100, 'CI chứa 0': bool(lo <= 0 <= hi), 'McNemar p (Holm)': m['p (Holm)']})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step 3a': kept-OTHER error analysis (amendment v3.7)
# Cue groups taken from the OTHER definition and edge cases of annotation_guideline_v1 (logistics, seller/product,
# price/fee/voucher value, policy/account/order outcome, customer service), plus a flag for app vocabulary. Fixed
# before looking at the misclassified reviews; matched on NFC lower-case text, with common unaccented spellings.
OTHER_CUES = {
    'giao hàng': r'giao hàng|giao hang|giao trễ|giao tre|giao chậm|giao cham|giao lâu|giao lau|giao nhầm|giao sai|giao thiếu|shipper|\bship\b|'
                 r'vận chuyển|van chuyen|nhận hàng|nhan hang|nhận được hàng|bưu cục|buu cuc|\bghn\b|\bghtk\b|j&t|\blex\b|express|\bspx\b',
    'người bán, sản phẩm': r'người bán|nguoi ban|\bshop\b|cửa hàng|sản phẩm|san pham|\bsp\b|hàng giả|hang gia|\bfake\b|hàng lỗi|hang loi|'
                           r'chất lượng|chat luong|đóng gói|dong goi|hàng nhái|giống hình|khác hình|sai mẫu|thiếu hàng',
    'giá, phí, voucher': r'giá cả|giá rẻ|giá cao|giá tiền|\bgiá\b|\bđắt\b|\bphí\b|phí ship|freeship|free ship|voucher|vocher|mã giảm|ma giam|'
                         r'khuyến mãi|khuyen mai|ưu đãi|uu dai|\bxu\b|\bcoin|lazcoin|giảm giá|giam gia|flash sale|\bsale\b',
    'chính sách, tài khoản, đơn hàng': r'khóa tài khoản|khoá tài khoản|khoa tai khoan|bị khóa|bị khoá|bất thường|bat thuong|hủy đơn|huỷ đơn|'
                                       r'huy don|tự hủy|tự huỷ|hoàn tiền|hoan tien|trả hàng|tra hang|đổi trả|doi tra|hoàn hàng|chính sách|'
                                       r'chinh sach|tài khoản|tai khoan|bảo hành',
    'chăm sóc khách hàng': r'chăm sóc khách hàng|cham soc khach hang|\bcskh\b|tổng đài|tong dai|nhân viên|nhan vien|hỗ trợ|ho tro|liên hệ|'
                           r'lien he|phản hồi|chatbot|chat bot',
}
APP_CUE = (r'\bapp\b|ứng dụng|ung dung|phần mềm|cập nhật|cap nhat|update|\blỗi\b|\bloi\b|\blag\b|\bđơ\b|văng|crash|đăng nhập|dang nhap|'
           r'\btải\b|\bload')


def cue_flags(texts):
    """Boolean frame: one column per OTHER cue group, 'không nhóm nào' (none of them) and 'nhắc app hoặc lỗi'."""
    t = pd.Series(list(texts)).map(lambda s: unicodedata.normalize('NFC', str(s)).lower())
    out = pd.DataFrame({k: t.str.contains(v, regex=True) for k, v in OTHER_CUES.items()})
    out['không nhóm nào'] = ~out[list(OTHER_CUES)].any(axis=1)
    out['nhắc app hoặc lỗi'] = t.str.contains(APP_CUE, regex=True)
    return out


def kept_other_table(flags, gold_other, flagged):
    """Per cue group among gold-OTHER reviews: count, share of OTHER, false ACTIONABLE (`flagged`) count and rate."""
    f, g, a = flags[np.asarray(gold_other)], np.asarray(gold_other), np.asarray(flagged)
    fp = a[g]
    rows = [{'nhóm': 'tất cả OTHER', 'n OTHER': int(g.sum()), '% OTHER': 100.0, 'n giữ nhầm': int(fp.sum()),
             'tỷ lệ giữ nhầm (%)': 100 * fp.mean()}]
    for c in f.columns:
        m = f[c].to_numpy()
        rows.append({'nhóm': c, 'n OTHER': int(m.sum()), '% OTHER': 100 * m.mean(), 'n giữ nhầm': int((m & fp).sum()),
                     'tỷ lệ giữ nhầm (%)': 100 * (m & fp).sum() / max(m.sum(), 1)})
    return pd.DataFrame(rows)
