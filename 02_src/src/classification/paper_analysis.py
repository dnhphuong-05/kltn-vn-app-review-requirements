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
SHORT = {'ViSoBERT': 'V', 'Bi-LSTM + fastText': 'B', LLM: 'L'}


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
    """(tier1, tier2, flat) of a locked set rebuilt from the GĐ10 files as in notebook 10: neural Tier 1 =
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
    """V→L style code of a configuration (V = ViSoBERT, B = Bi-LSTM + fastText, L = LLM)."""
    if name == FLAT:
        return flat
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
    return 'phẳng' if name == FLAT else ('cascade' if name.startswith('Cascade ') else 'hybrid')


def pareto_figure(df, path, panels, text):
    """Two panels sharing the Macro-F1 axis, one per cost measure; 95% CI bars on the Pareto points. df: one row per configuration with columns
    'cấu hình', 'Macro-F1', 'CI 2,5%', 'CI 97,5%' (fractions) and the cost columns named in panels.
    panels: [(cost column, x-axis title, log scale?, Pareto column)]. text: legend and y-axis strings."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.3), sharey=True)
    for ax, (col, xtitle, log, pcol) in zip(np.atleast_1d(axes), panels):
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
            key = (round(r['x'], 4), round(y, 1))
            labels.setdefault(key, []).append((short_label(r['cấu hình'], text.get('flat_code', 'V phẳng')), on))
        for (x, y), items in labels.items():
            on = any(o for _, o in items)
            ax.annotate(' / '.join(s for s, _ in items), (x, y), xytext=(6, 3), textcoords='offset points',
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
