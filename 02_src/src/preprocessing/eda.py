"""GĐ2 — descriptive EDA on the interim corpora (no labels exist yet; nothing here estimates
actionable/BUG/... shares). Figures are static PNGs for the thesis; colours follow the
validated reference palette (slot 1 = Shopee, slot 2 = Lazada)."""
import os
from collections import Counter
from datetime import date, datetime, timedelta, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

VN_TZ = timezone(timedelta(hours=7))
PARTIAL_DAY = date(2026, 10, 6)            # Google Play shows reviews ~1 day late
COLORS = {'shopee': '#2a78d6', 'lazada': '#eb6834'}
LABELS = {'shopee': 'Shopee', 'lazada': 'Lazada'}
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SHORT_WORDS = 2                           # "cực ngắn" = ≤ 2 từ

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2, 'ytick.color': INK_2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.frameon': False,
})


def to_frame(records, app):
    df = pd.DataFrame(records)
    df['app'] = app
    df['ts'] = pd.to_datetime(df['review_date'], utc=True).dt.tz_convert(VN_TZ)
    df['day'] = df['ts'].dt.date
    df['month'] = df['ts'].dt.strftime('%Y-%m')
    df['n_words'] = df['text_clean'].str.split().str.len()
    df['n_chars'] = df['text_clean'].str.len()
    df['has_reply'] = df['reply_content'].notna() & (df['reply_content'] != '')
    df['near_dup'] = df['flags'].apply(lambda f: 'near_duplicate' in f)
    return df


def summarize(df, n_dropped_no_letters, n_after_window):
    q = df['n_words'].quantile([.25, .5, .75, .9]).round(1)
    top_versions = df['app_version'].dropna().value_counts().head(5)
    return {
        'n_reviews': len(df),
        'date_min': str(df['ts'].min()), 'date_max': str(df['ts'].max()),
        'rating_pct': (df['rating'].value_counts(normalize=True).sort_index() * 100).round(1).to_dict(),
        'words_q25_q50_q75_q90': q.tolist(),
        'chars_median': float(df['n_chars'].median()),
        'short_le2_words_pct': round((df['n_words'] <= SHORT_WORDS).mean() * 100, 1),
        'app_version_missing_pct': round(df['app_version'].isna().mean() * 100, 1),
        'top_versions': top_versions.to_dict(),
        'reply_pct': round(df['has_reply'].mean() * 100, 1),
        'lang_pct': (df['lang'].value_counts(normalize=True) * 100).round(1).to_dict(),
        'near_dup_pct': round(df['near_dup'].mean() * 100, 1),
        'no_letters_dropped_pct': round(n_dropped_no_letters / n_after_window * 100, 2),
    }


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return path


def fig_rating(dfs, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    width = 0.38
    for i, (app, df) in enumerate(dfs.items()):
        pct = df['rating'].value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0) * 100
        xs = [x + (i - 0.5) * (width + 0.02) for x in pct.index]
        ax.bar(xs, pct.values, width=width, color=COLORS[app], label=LABELS[app])
        for x, v in zip(xs, pct.values):
            if v >= 10:
                ax.text(x, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5, color=INK_2)
    ax.set_xticks([1, 2, 3, 4, 5], [f'{k}★' for k in range(1, 6)])
    ax.set_ylabel('% review của app')
    ax.set_title('Phân bố mức đánh giá')
    ax.legend(loc='upper center')
    return _save(fig, out_dir, 'eda_rating_distribution.png')


def fig_length(dfs, out_dir, cap=80):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = list(range(1, cap + 2))
    for app, df in dfs.items():
        w = df['n_words'].clip(upper=cap)
        pct = w.value_counts(normalize=True).reindex(bins[:-1], fill_value=0) * 100
        ax.step(pct.index, pct.values, where='mid', color=COLORS[app], linewidth=2, label=LABELS[app])
    ax.set_xlabel(f'Số từ trong text_clean (gộp ≥{cap})')
    ax.set_ylabel('% review của app')
    ax.set_title('Phân bố độ dài review')
    ax.legend()
    return _save(fig, out_dir, 'eda_length_words.png')


def fig_monthly(dfs, out_dir):
    fig, axes = plt.subplots(len(dfs), 1, figsize=(7.2, 2.4 * len(dfs)), sharex=True)
    for ax, (app, df) in zip(axes, dfs.items()):
        m = df['month'].value_counts().sort_index()
        partial = {m.index[0], m.index[-1]}   # window starts 06/10/2025, ends 06/10/2026
        for x, (month, v) in enumerate(m.items()):
            ax.bar(x, v, width=0.8, color=COLORS[app], alpha=0.35 if month in partial else 1.0)
        ax.set_xticks(range(len(m)), m.index, rotation=45, ha='right')
        ax.set_ylabel('Số review')
        ax.set_title(f'{LABELS[app]} — số review theo tháng (cột nhạt: tháng không trọn trong cửa sổ)',
                     fontsize=9)
    fig.tight_layout()
    return _save(fig, out_dir, 'eda_monthly_volume.png')


def fig_daily_tail(dfs, out_dir, days=45):
    end = PARTIAL_DAY
    rng = [end - timedelta(days=d) for d in range(days - 1, -1, -1)]
    fig, axes = plt.subplots(len(dfs), 1, figsize=(7.2, 2.4 * len(dfs)), sharex=True)
    for ax, (app, df) in zip(axes, dfs.items()):
        c = df['day'].value_counts().reindex(rng, fill_value=0)
        for x, (d, v) in enumerate(c.items()):
            is_partial = d == PARTIAL_DAY
            ax.bar(x, v, width=0.8, color=SURFACE if is_partial else COLORS[app],
                   edgecolor=COLORS[app], hatch='////' if is_partial else None, linewidth=0.8)
        x_last = len(rng) - 1
        ax.annotate('06/10: thiếu một phần\n(Google Play trễ ~1 ngày)', xy=(x_last, c.iloc[-1]),
                    xytext=(x_last - 12, c.max() * 0.85), fontsize=7.5, color=INK_2,
                    arrowprops={'arrowstyle': '-', 'color': INK_2, 'linewidth': 0.6})
        ax.set_ylabel('Số review')
        ax.set_title(f'{LABELS[app]} — số review theo ngày, {days} ngày cuối cửa sổ', fontsize=9)
        ticks = list(range(x_last, -1, -7))[::-1]   # weekly, anchored on the last day (06/10)
        ax.set_xticks(ticks, [rng[t].strftime('%d/%m') for t in ticks])
    fig.tight_layout()
    return _save(fig, out_dir, 'eda_daily_tail.png')


def fig_versions(dfs, out_dir, top=8):
    fig, axes = plt.subplots(1, len(dfs), figsize=(7.6, 3.2))
    for ax, (app, df) in zip(axes, dfs.items()):
        vc = df['app_version'].dropna().value_counts().head(top)[::-1]
        ax.barh(range(len(vc)), vc.values, color=COLORS[app], height=0.7)
        ax.set_yticks(range(len(vc)), vc.index)
        miss = df['app_version'].isna().mean() * 100
        ax.set_title(f'{LABELS[app]} — top {top} app_version\n(thiếu version: {miss:.1f}%)', fontsize=9)
        ax.set_xlabel('Số review')
        ax.grid(axis='y', visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, 'eda_app_version_top.png')


def all_figures(dfs, out_dir):
    return [fig_rating(dfs, out_dir), fig_length(dfs, out_dir), fig_monthly(dfs, out_dir),
            fig_daily_tail(dfs, out_dir), fig_versions(dfs, out_dir)]


# ---------------------------------------------------------------- report
def _fmt_pct(d):
    return ' / '.join(f'{k}: {v}%' for k, v in d.items())


def write_report(path, summaries, steps, qa, figures, notes):
    """Markdown report: tables are generated from the numbers; `notes` holds the per-app
    remarks written by the researcher after reading the numbers."""
    apps = list(summaries)
    L = ['# EDA — Giai đoạn 2 (mô tả thuần, chưa có nhãn)', '',
         'Báo cáo được sinh từ `src/preprocessing/eda.py`. Không có ước lượng tỷ lệ actionable/BUG/FEATURE/UX ở giai đoạn này.', '',
         '## 1. Đối soát số lượng', '',
         '| Bước | ' + ' | '.join(LABELS[a] for a in apps) + ' |', '|---|' + '---|' * len(apps)]
    rows = [('Raw', 'raw'), ('Trùng review_id (loại)', 'duplicate_review_id'),
            ('Ngoài cửa sổ — trước 2025-10-06 (loại)', 'outside_window_before'),
            ('Ngoài cửa sổ — sau 2026-10-06 23:59 (loại)', 'outside_window_after'),
            ('Sau cắt cửa sổ', 'after_window'), ('Text rỗng (loại)', 'drop_empty'),
            ('Không có chữ cái — emoji/ký hiệu/số (loại)', 'drop_no_letters'),
            ('Sau lọc (interim)', 'after_filter')]
    for label, k in rows:
        L.append(f'| {label} | ' + ' | '.join(f'{steps[a][k]:,}' for a in apps) + ' |')
    L += ['', '## 2. Tóm tắt mô tả', '', '| Chỉ số | ' + ' | '.join(LABELS[a] for a in apps) + ' |',
          '|---|' + '---|' * len(apps)]
    srows = [('Số review', lambda s: f"{s['n_reviews']:,}"),
             ('Khoảng ngày (giờ VN)', lambda s: f"{s['date_min'][:10]} → {s['date_max'][:16]}"),
             ('Phân bố rating', lambda s: _fmt_pct({f'{k}★': v for k, v in s['rating_pct'].items()})),
             ('Số từ Q25/Q50/Q75/Q90', lambda s: ' / '.join(str(x) for x in s['words_q25_q50_q75_q90'])),
             ('Số ký tự (trung vị)', lambda s: s['chars_median']),
             (f'Review ≤ {SHORT_WORDS} từ', lambda s: f"{s['short_le2_words_pct']}%"),
             ('Thiếu app_version', lambda s: f"{s['app_version_missing_pct']}%"),
             ('Top version', lambda s: ', '.join(f'{k} ({v:,})' for k, v in s['top_versions'].items())),
             ('Có phản hồi của sàn', lambda s: f"{s['reply_pct']}%"),
             ('Ngôn ngữ (rule-based)', lambda s: _fmt_pct(s['lang_pct'])),
             ('near_duplicate', lambda s: f"{s['near_dup_pct']}%"),
             ('Emoji/ký hiệu-only đã loại (trên số sau cắt cửa sổ)', lambda s: f"{s['no_letters_dropped_pct']}%")]
    for label, fn in srows:
        L.append(f'| {label} | ' + ' | '.join(str(fn(summaries[a])) for a in apps) + ' |')
    L += ['', '## 3. Nhận xét', '']
    for a in apps:
        L += [f'**{LABELS[a]}.** {notes.get(a, "")}', '']
    L += ['## 4. QA ẩn danh hóa (PII)', '', '| | ' + ' | '.join(LABELS[a] for a in apps) + ' |',
          '|---|' + '---|' * len(apps),
          '| Mẫu reply_content soi | ' + ' | '.join(str(qa[a]['n_sample_reply']) for a in apps) + ' |',
          '| Mẫu text soi | ' + ' | '.join(str(qa[a]['n_sample_text']) for a in apps) + ' |',
          '| Nghi sót trong mẫu (regex phụ) | ' + ' | '.join(str(qa[a]['n_suspects_in_sample']) for a in apps) + ' |',
          '| Nghi sót toàn corpus (regex phụ) | ' + ' | '.join(
              (', '.join(f'{k}={v}' for k, v in qa[a]['suspects_full_corpus'].items()) or '0') for a in apps) + ' |',
          '', f"**Hành động:** {notes.get('qa_action', '')}", '',
          '## 5. Hạn chế', '',
          '- Ngày 06/10/2026 chỉ có một phần (Google Play hiển thị review trễ khoảng 1 ngày); không bù dữ liệu. '
          'Ảnh hưởng nhẹ tới cửa sổ last_30.',
          '- Ngôn ngữ được gắn bằng luật (dấu tiếng Việt, chữ ngoài hệ Latin, từ điển tiếng Việt không dấu); '
          'review rất ngắn không dấu được để `und`. Review không phải tiếng Việt chỉ được đánh dấu, chưa loại.',
          '- near_duplicate = text_clean trùng y hệt; phần lớn là câu ngắn chung chung ("tốt", "ok"); chỉ đánh dấu, không xóa.',
          '', '## 6. Hình', ''] + [f'- `{os.path.basename(p)}`' for p in figures]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return path
