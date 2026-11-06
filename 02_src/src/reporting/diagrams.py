"""Method diagrams of the thesis (matplotlib): research framework, data, annotation, RQ1 cascade, RQ2, RQ3, D6 data model.

Every number on a diagram comes from the `facts` dict built in notebook 17 from saved results and frozen configs; the
diagram functions only lay them out. Coordinates are in inches (axes = whole figure), y from the bottom. Each diagram is
saved as PNG (300 dpi) and SVG (text kept as text, editable).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle, Wedge

from src.reporting.style import GREY, GREY_LIGHT, INK, INK_2, SLOT, num, style

LS = 1.18                                         # line spacing of every text block
STRAT = {'A': SLOT[1], 'B': SLOT[0], 'C': SLOT[2]}   # A Bi-LSTM + fastText, B ViSoBERT, C LLM (strategy colours of the results)
TYPE = {'BUG': SLOT[0], 'FEATURE': SLOT[1], 'UX': SLOT[2]}
# phase tints (background only, never encode data)
PH = {'data': ('#eef4fc', '#8fb3e3'), 'label': ('#f3eff9', '#b3a1d3'), 'rq1': ('#fdf1ea', '#eeb08b'),
      'rq2': ('#ebf6f1', '#8ccdb1'), 'rq3': ('#fcf7e6', '#dcc16a'), 'out': ('#f2f2f0', '#b5b4af')}
LOCKED = '#c0392b'


# ---------------------------------------------------------------- primitives
def _fig(w, h):
    style()
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


def _n(s):
    return 0 if not s else s.count('\n') + 1


def text_block(ax, cx, cy, title=None, body=None, fs=7.8, tfs=None, color=INK, tcolor=None, ha='center', z=5):
    """Bold title over regular body, the pair centred on (cx, cy)."""
    tfs = tfs or fs + 0.6
    ht, hb = _n(title) * tfs * LS / 72, _n(body) * fs * LS / 72
    gap = 0.035 if title and body else 0
    top = cy + (ht + hb + gap) / 2
    if title:
        ax.text(cx, top - ht / 2, title, ha=ha, va='center', fontsize=tfs, fontweight='bold', color=tcolor or color,
                linespacing=LS, zorder=z)
    if body:
        ax.text(cx, top - ht - gap - hb / 2, body, ha=ha, va='center', fontsize=fs, color=color, linespacing=LS, zorder=z)


def box(ax, cx, cy, w, h, title=None, body=None, fc='white', ec=INK_2, lw=0.9, fs=7.8, tfs=None, r=0.07, ls='-',
        color=INK, tcolor=None, z=3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec,
                                lw=lw, ls=ls, zorder=z))
    text_block(ax, cx, cy, title, body, fs, tfs, color, tcolor, z=z + 1)
    return (cx, cy, w, h)


def header_box(ax, cx, cy, w, h, title, body, head_fc, fc='white', ec=None, fs=7.6, tfs=8.2, head_h=0.27, z=3):
    """Card with a coloured title band (white bold text) and a body below."""
    ec = ec or head_fc
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=0.06', fc=fc, ec=ec, lw=1.0, zorder=z))
    ax.add_patch(FancyBboxPatch((x0, y0 + h - head_h), w, head_h, boxstyle='round,pad=0,rounding_size=0.06', fc=head_fc,
                                ec=ec, lw=1.0, zorder=z + 0.1))
    ax.add_patch(Rectangle((x0 + 0.01, y0 + h - head_h), w - 0.02, head_h / 2, fc=head_fc, ec='none', zorder=z + 0.2))
    ax.text(cx, y0 + h - head_h / 2, title, ha='center', va='center', fontsize=tfs, fontweight='bold', color='white',
            zorder=z + 1)
    text_block(ax, cx, y0 + (h - head_h) / 2, None, body, fs, z=z + 1)
    return (cx, cy, w, h)


def db(ax, cx, cy, w, h, title=None, body=None, fc='white', ec=INK_2, fs=7.6, tfs=None, z=3):
    """Database cylinder."""
    e = min(0.16, h * 0.22)
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(Ellipse((cx, y0 + e / 2), w, e, fc=fc, ec=ec, lw=0.9, zorder=z))
    ax.add_patch(Rectangle((x0, y0 + e / 2), w, h - e, fc=fc, ec='none', zorder=z + 0.1))
    ax.plot([x0, x0], [y0 + e / 2, y0 + h - e / 2], color=ec, lw=0.9, zorder=z + 0.2)
    ax.plot([x0 + w, x0 + w], [y0 + e / 2, y0 + h - e / 2], color=ec, lw=0.9, zorder=z + 0.2)
    ax.add_patch(Ellipse((cx, y0 + h - e / 2), w, e, fc=fc, ec=ec, lw=0.9, zorder=z + 0.3))
    text_block(ax, cx, cy - e / 3, title, body, fs, tfs, z=z + 1)
    return (cx, cy, w, h)


def diamond(ax, cx, cy, w, h, text, fc='white', ec=INK_2, fs=7.4, z=3):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True, fc=fc,
                         ec=ec, lw=0.9, zorder=z))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, linespacing=LS, zorder=z + 1, color=INK)
    return (cx, cy, w, h)


def panel(ax, x0, y0, w, h, title, tint, fs=9.2, ls='-', align='center'):
    fc, ec = tint
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=0.1', fc=fc, ec=ec, lw=1.1, ls=ls,
                                zorder=1))
    if title:
        x = x0 + w / 2 if align == 'center' else x0 + 0.15
        ax.text(x, y0 + h - 0.2, title, ha=align, va='center', fontsize=fs, fontweight='bold', color=INK,
                zorder=2.8, linespacing=LS, bbox=dict(fc=fc, ec='none', pad=1.5))   # connectors pass behind the title


def arrow(ax, pts, color=INK_2, lw=1.0, ls='-', label=None, label_at=None, fs=7.0, label_ha='center', head=True, z=2.5):
    """Poly-line arrow through pts (list of (x, y)); head on the last segment."""
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle='butt')
    if head:
        ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=z,
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls, mutation_scale=9, shrinkA=0, shrinkB=0))
    else:
        ax.plot(xs[-2:], ys[-2:], color=color, lw=lw, ls=ls, zorder=z)
    if label:
        lx, ly = label_at or ((xs[-2] + xs[-1]) / 2, (ys[-2] + ys[-1]) / 2)
        ax.text(lx, ly, label, ha=label_ha, va='center', fontsize=fs, color=INK_2, zorder=z + 2, linespacing=LS,
                bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.9))


def person(ax, cx, cy, s=0.3, color=INK_2, z=6):
    """Small person icon of height s centred on (cx, cy)."""
    ax.add_patch(Circle((cx, cy + 0.22 * s), 0.17 * s, fc=color, ec='white', lw=0.6, zorder=z))
    ax.add_patch(Wedge((cx, cy - 0.5 * s), 0.4 * s, 0, 180, fc=color, ec='white', lw=0.6, zorder=z))


def lock(ax, cx, cy, s=0.16, color=LOCKED, z=6):
    ax.add_patch(Arc((cx, cy + 0.12 * s), 0.55 * s, 0.75 * s, theta1=0, theta2=180, color=color, lw=1.3, zorder=z))
    ax.add_patch(FancyBboxPatch((cx - 0.4 * s, cy - 0.45 * s), 0.8 * s, 0.6 * s, boxstyle='round,pad=0,rounding_size=0.01',
                                fc=color, ec=color, zorder=z))


def badge(ax, cx, cy, label, color=INK_2, r=0.13, fs=8.5, z=6):
    ax.add_patch(Circle((cx, cy), r, fc=color, ec='white', lw=0.8, zorder=z))
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fs, fontweight='bold', color='white', zorder=z + 1)


def chip(ax, cx, cy, text, color, w=None, h=0.19, fs=6.8, z=6):
    w = w or 0.09 * len(text) + 0.16
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.05', fc=color,
                                ec='none', zorder=z))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, fontweight='bold', color='white', zorder=z + 1)
    return w


def chips_row(ax, cx, cy, items, gap=0.06, **kw):
    widths = [kw.get('w') or 0.09 * len(t) + 0.16 for t, _ in items]
    x = cx - (sum(widths) + gap * (len(items) - 1)) / 2
    for (t, c), w in zip(items, widths):
        chip(ax, x + w / 2, cy, t, c, w=w, **{k: v for k, v in kw.items() if k != 'w'})
        x += w + gap


def _save(fig, path):
    fig.savefig(path, dpi=300, facecolor='white')
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(path[:-4] + '.svg', facecolor='white')
    plt.close(fig)
    return path


def _top(b):
    return (b[0], b[1] + b[3] / 2)


def _bot(b):
    return (b[0], b[1] - b[3] / 2)


def _left(b):
    return (b[0] - b[2] / 2, b[1])


def _right(b):
    return (b[0] + b[2] / 2, b[1])


# ---------------------------------------------------------------- 0. DSR process
def fig_dsr(f, path):
    """Six DSR activities (Peffers et al., 2007) mapped to the thesis, snake layout 3 + 3."""
    W, H = 8.6, 5.0
    fig, ax = _fig(W, H)
    w, h = 2.55, 1.72
    xs = [1.5, 4.3, 7.1]
    y1, y2 = 3.85, 1.42
    cards = [
        ('1. Xác định vấn đề', f"Review Google Play tiếng Việt\nkhối lượng lớn, nhiễu;\nrút requirement thủ công tốn công\nvà khó truy vết nguồn\n\nChương 1", PH['data']),
        ('2. Mục tiêu giải pháp', "RQ1: chọn chiến lược phân loại\ncân nhắc chi phí\nRQ2: nhóm issue có nghĩa\nRQ3: requirement đúng cấu trúc,\ntruy vết được · Chương 1–2", PH['label']),
        ('3. Thiết kế & phát triển', "Pipeline cascade hai tầng\n+ quy tắc chọn chiến lược;\nBERTopic + temporal;\nsinh requirement bằng LLM\nChương 3 · GĐ1–GĐ14", PH['rq1']),
        ('4. Minh họa', f"Chạy trên {num(f['n_shopee'])} review Shopee:\n{num(f['n_actionable'])} actionable → {f['n_topics']} cụm\n→ {f['n_requirements']} Candidate Requirement\n(artifact D6)\nChương 4 · GĐ11–GĐ16", PH['rq2']),
        ('5. Đánh giá', f"RQ1: Hold-out {num(f['n_holdout'])}, Lazada {f['n_lazada_test']};\nRQ2: C_v, manual purity;\nRQ3: QUS, chấm tay;\nchi phí đo thực tế\nChương 4 · GĐ10, 12, 15", PH['rq3']),
        ('6. Truyền đạt', "Khóa luận; artifact D1–D7;\nmã nguồn, config và prompt\nđã khóa để tái lập\n\nChương 5", PH['out']),
    ]
    pos = [(xs[0], y1), (xs[1], y1), (xs[2], y1), (xs[2], y2), (xs[1], y2), (xs[0], y2)]
    boxes = []
    for (t, b, (fc, ec)), (x, y) in zip(cards, pos):
        boxes.append(header_box(ax, x, y, w, h, t, b, head_fc=ec, fc=fc, ec=ec, fs=8.0, tfs=8.8, head_h=0.32))
    arrow(ax, [_right(boxes[0]), _left(boxes[1])])
    arrow(ax, [_right(boxes[1]), _left(boxes[2])])
    arrow(ax, [_bot(boxes[2]), _top(boxes[3])])
    arrow(ax, [_left(boxes[3]), _right(boxes[4])])
    arrow(ax, [_left(boxes[4]), _right(boxes[5])])
    # iteration: evaluation -> design
    xa, xb = boxes[4][0] + 0.55, boxes[2][0] - 0.75
    ym = (y1 - h / 2 + y2 + h / 2) / 2
    arrow(ax, [(xa, y2 + h / 2), (xa, ym), (xb, ym), (xb, y1 - h / 2)], ls='--', color=INK_2,
          label='lặp lại khi đánh giá chưa đạt\n(amendment, prompt v1 → v3)', label_at=((xa + xb) / 2, ym), fs=6.8)
    ax.text(W / 2, 0.2, 'Đường liền: trình tự hoạt động; đường nét đứt: vòng lặp thiết kế – đánh giá.', ha='center',
            fontsize=7, color=INK_2, style='italic')
    return _save(fig, path)


# ---------------------------------------------------------------- 1. research framework
def fig_framework(f, path):
    """Overview of the six phases, four steps each. One short line per box: the numbers and details are in figures 3.1-3.6,
    so this figure only shows input, main steps, output and where evaluation happens (thick dark border). `f` is kept for
    the common diagram signature."""
    W = 8.6
    rows = 6
    rh, gap, top_m, bot_m = 0.98, 0.3, 0.2, 0.62
    H = top_m + rows * rh + (rows - 1) * gap + bot_m
    fig, ax = _fig(W, H)
    lx0 = 0.12
    bx0, bw, bgap = 1.72, 1.5, 0.27
    cx = [bx0 + bw / 2 + i * (bw + bgap) for i in range(4)]
    bh, fs, tfs = 0.6, 8.0, 8.8
    EV = dict(ec=INK, lw=1.6)                       # evaluation step
    tints = [PH['data'], PH['label'], PH['rq1'], PH['rq2'], PH['rq3'], PH['out']]
    labels = [('Dữ liệu', 'GĐ1–GĐ2'), ('Gán nhãn\n& chia tập', 'GĐ3–GĐ5'), ('RQ1 · Phân loại\nhai tầng', 'GĐ6–GĐ11'),
              ('RQ2 · Gom cụm\n& thời gian', 'GĐ12–GĐ13'), ('RQ3 · Sinh yêu\ncầu & đánh giá', 'GĐ14–GĐ15'),
              ('Đầu ra\ncho BA', 'GĐ16')]
    ycs = [H - top_m - rh / 2 - k * (rh + gap) for k in range(rows)]
    for k, ((name, gd), (fc, ec), y) in enumerate(zip(labels, tints, ycs)):
        panel(ax, 0.08, y - rh / 2, W - 0.16, rh, None, (fc, ec))
        badge(ax, lx0 + 0.2, y + 0.14, str(k + 1), color=ec, r=0.14)
        ax.text(lx0 + 0.42, y + 0.14, name, ha='left', va='center', fontsize=8.6, fontweight='bold', linespacing=LS)
        ax.text(lx0 + 0.42, y - 0.3, gd, ha='left', va='center', fontsize=7.6, color=INK_2)

    def bx(i, y, title, body=None, ev=False, h=bh):
        return box(ax, cx[i], y, bw, h, title, body, fs=fs, tfs=tfs, **(EV if ev else {}))

    def corner_person(b):
        person(ax, b[0] + b[2] / 2 - 0.15, b[1] + b[3] / 2 - 0.02, s=0.28)

    def corner_lock(b):
        lock(ax, b[0] + b[2] / 2 - 0.13, b[1] + b[3] / 2 - 0.17)

    B = {}
    y = ycs[0]
    B['src'] = db(ax, cx[0], y, bw, bh, 'Google Play', 'Shopee · Lazada', fs=fs, tfs=tfs)
    B['crawl'] = bx(1, y, 'Thu thập', '12 tháng đến T0')
    B['clean'] = bx(2, y, 'Làm sạch', 'ẩn danh, chuẩn hóa')
    B['corpus'] = db(ax, cx[3], y, bw, bh, 'Corpus', 'review tiếng Việt', fs=fs, tfs=tfs)
    y = ycs[1]
    B['guide'] = bx(0, y, 'Guideline', '4 lớp nhãn')
    corner_person(B['guide'])
    B['pilot'] = bx(1, y, 'Pilot ngẫu nhiên', 'ước lượng tỷ lệ lớp')
    B['gold'] = db(ax, cx[2], y, bw, bh, 'Gold Dataset', 'Train · Val · Hold-out', fs=fs, tfs=tfs)
    corner_lock(B['gold'])
    B['lc'] = bx(3, y, 'Learning curve', 'chốt cỡ Train')
    y = ycs[2]
    h3 = bh + 0.16
    for key, i, title, body in [('t1', 0, 'Tầng 1', 'lọc ACTIONABLE'), ('t2', 1, 'Tầng 2', 'BUG / FEATURE / UX')]:
        B[key] = box(ax, cx[i], y, bw, h3)
        text_block(ax, cx[i], y + 0.1, title, body, fs, tfs)
        chips_row(ax, cx[i], y - 0.22, [('A', STRAT['A']), ('B', STRAT['B']), ('C', STRAT['C'])], w=0.3)
    B['eval1'] = bx(2, y, 'Đánh giá Hold-out', 'hiệu năng · chi phí', ev=True, h=h3)
    corner_lock(B['eval1'])
    B['full'] = bx(3, y, 'Phân loại corpus', 'chiến lược đã chọn', h=h3)
    y = ycs[3]
    B['bt'] = bx(0, y, 'Gom cụm', 'BERTopic, chọn theo C_v')
    B['topics'] = bx(1, y, 'Loại chủ đạo', 'gắn cho từng cụm')
    B['purity'] = bx(2, y, 'Độ thuần của cụm', 'hai người chấm', ev=True)
    corner_person(B['purity'])
    B['temp'] = bx(3, y, 'Bằng chứng thời gian', 'theo cửa sổ 30 ngày')
    y = ycs[4]
    B['sel'] = bx(0, y, 'Chọn evidence', 'theo cụm và loại')
    B['gen'] = bx(1, y, 'Sinh bằng LLM', 'User Story + AC')
    B['check'] = bx(2, y, 'Kiểm tra truy vết', 'evidence ID hợp lệ')
    B['eval3'] = bx(3, y, 'Đánh giá', 'QUS · hai người chấm', ev=True)
    corner_person(B['eval3'])
    y = ycs[5]
    B['cand'] = bx(0, y, 'Candidate Requirement', 'kèm evidence ID')
    B['freq'] = bx(1, y, 'Sắp xếp', 'theo tần suất phản ánh')
    B['d6'] = db(ax, cx[2], y, bw, bh, 'D6', '5 bảng quan hệ', fs=fs, tfs=tfs)
    B['ba'] = bx(3, y, 'BA xem xét', 'chấp nhận · sửa · loại')
    corner_person(B['ba'])
    order = [['src', 'crawl', 'clean', 'corpus'], ['guide', 'pilot', 'gold', 'lc'], ['t1', 't2', 'eval1', 'full'],
             ['bt', 'topics', 'purity', 'temp'], ['sel', 'gen', 'check', 'eval3'], ['cand', 'freq', 'd6', 'ba']]
    for r, keys in enumerate(order):
        for a, b in zip(keys[:-1], keys[1:]):
            arrow(ax, [_right(B[a]), _left(B[b])])
        if r < rows - 1:
            a, b = B[keys[-1]], B[order[r + 1][0]]
            ym = (ycs[r] - rh / 2 + ycs[r + 1] + rh / 2) / 2
            arrow(ax, [_bot(a), (a[0], ym), (b[0], ym), _top(b)])
    # legend
    ly = 0.3
    for x, k, text in [(1.0, 'A', 'Bi-LSTM + fastText'), (2.55, 'B', 'ViSoBERT fine-tune'), (4.1, 'C', 'LLM few-shot')]:
        chips_row(ax, x, ly, [(k, STRAT[k])], w=0.3)
        ax.text(x + 0.2, ly, text, va='center', fontsize=7.2)
    person(ax, 5.3, ly, s=0.26)
    ax.text(5.47, ly, 'có con người', va='center', fontsize=7.2)
    lock(ax, 6.4, ly)
    ax.text(6.55, ly, 'tập khóa', va='center', fontsize=7.2)
    box(ax, 7.35, ly, 0.3, 0.18, **EV)
    ax.text(7.56, ly, 'bước đánh giá', va='center', fontsize=7.2)
    return _save(fig, path)


# ---------------------------------------------------------------- 2. data collection and cleaning
def fig_data(f, path):
    W, H = 8.6, 8.75
    fig, ax = _fig(W, H)
    cs, cl, cw = 5.15, 7.45, 1.9
    sx = 0.25
    ax.text(sx, H - 0.3, 'Bước', fontsize=9, fontweight='bold', va='center')
    header_box(ax, cs, H - 0.3, cw, 0.32, 'Shopee', None, head_fc=SLOT[0], head_h=0.32)
    header_box(ax, cl, H - 0.3, cw, 0.32, 'Lazada', None, head_fc=SLOT[1], head_h=0.32)
    steps = [
        ('Thu thập từ Google Play', 'google-play-scraper · ngày cào 07/10/2026\nShopee: theo từng mức 1–5 sao rồi gộp; Lazada: mới nhất',
         (num(f['crawl_shopee']), 'review cào được'), (num(f['crawl_lazada']), 'review cào được')),
        ('Loại trùng review_id', 'kiểm lại sau khi gộp các mẻ cào', ('0', 'trùng'), ('0', 'trùng')),
        ('Cắt cửa sổ nghiên cứu', '06/10/2025 – 06/10/2026 (giờ Việt Nam)',
         (f"−{num(f['out_window_shopee'])}", f"còn {num(f['crawl_shopee'] - f['out_window_shopee'])}"),
         (f"−{num(f['out_window_lazada'])}", f"còn {num(f['crawl_lazada'] - f['out_window_lazada'])}")),
        ('Ẩn danh hóa', 'áp lại luật ẩn danh (tên, số điện thoại, …)\ncho văn bản review và phản hồi',
         ('Áp dụng', 'không đổi số review'), ('Áp dụng', 'không đổi số review')),
        ('Chuẩn hóa văn bản tối thiểu', 'Unicode NFC, bỏ ký tự điều khiển, gộp khoảng trắng;\nkhông hạ chữ, giữ emoji và viết tắt',
         ('Áp dụng', 'không đổi số review'), ('Áp dụng', 'không đổi số review')),
        ('Loại review không có chữ cái', 'rỗng, chỉ emoji, ký hiệu hoặc số\n(lưu riêng kèm lý do để truy vết)',
         (f"−{num(f['no_letter_shopee'])}", 'review'), (f"−{num(f['no_letter_lazada'])}", 'review')),
        ('Gắn cờ, không xóa', 'trùng nội dung với review khác;\nngôn ngữ khác hoặc không xác định',
         (f"{num(f['dup_flag_shopee'])} · {num(f['lang_flag_shopee'])}", 'trùng nội dung · ngôn ngữ'),
         (f"{num(f['dup_flag_lazada'])} · {num(f['lang_flag_lazada'])}", 'trùng nội dung · ngôn ngữ')),
    ]
    sh, sgap = 0.72, 0.2
    y = H - 0.75 - sh / 2
    prev = None
    for k, (name, desc, vs, vl) in enumerate(steps):
        badge(ax, sx + 0.14, y + 0.12, str(k + 1), color=INK_2, r=0.12, fs=7.5)
        ax.text(sx + 0.34, y + 0.12, name, fontsize=8.4, fontweight='bold', va='center')
        ax.text(sx + 0.34, y - 0.15, desc, fontsize=6.9, va='center', color=INK_2, linespacing=LS)
        bs = box(ax, cs, y, cw, sh - 0.08, vs[0], vs[1], fc=PH['data'][0], ec=PH['data'][1], fs=7, tfs=9.4)
        bl = box(ax, cl, y, cw, sh - 0.08, vl[0], vl[1], fc=PH['rq1'][0], ec=PH['rq1'][1], fs=7, tfs=9.4)
        if prev:
            arrow(ax, [_bot(prev[0]), _top(bs)])
            arrow(ax, [_bot(prev[1]), _top(bl)])
        prev = (bs, bl)
        y -= sh + sgap
    y -= 0.12
    ds = db(ax, cs, y, cw, 0.95, f"{num(f['n_shopee'])} review", 'corpus Shopee\n(interim, đã ẩn danh)', fc='white', ec=SLOT[0], tfs=9.6)
    dl = db(ax, cl, y, cw, 0.95, f"{num(f['n_lazada'])} review", 'corpus Lazada\n(interim, đã ẩn danh)', fc='white', ec=SLOT[1], tfs=9.6)
    arrow(ax, [_bot(prev[0]), _top(ds)])
    arrow(ax, [_bot(prev[1]), _top(dl)])
    ax.text(sx + 0.34, y + 0.1, 'Corpus nghiên cứu', fontsize=8.6, fontweight='bold', va='center')
    ax.text(sx + 0.34, y - 0.2, 'Shopee: gán nhãn, huấn luyện, phân loại toàn corpus\nLazada: rút 500 review làm external test (RQ1)',
            fontsize=6.9, va='center', color=INK_2, linespacing=LS)
    ax.text(sx, 0.22, 'Dữ liệu thô (01_data/raw/) giữ nguyên, không sửa; mọi bước chạy trên bản sao interim.', fontsize=7,
            color=INK_2, style='italic', va='center')
    return _save(fig, path)


# ---------------------------------------------------------------- 3. annotation and splits
def fig_annotation(f, path):
    W, H = 8.6, 8.95
    fig, ax = _fig(W, H)
    c = [1.45, 4.3, 7.15]
    dS = db(ax, c[1], H - 0.5, 2.5, 0.66, f"Corpus Shopee {num(f['n_shopee'])}", 'nguồn rút mọi mẫu (seed 42)', ec=SLOT[0], tfs=8.4)
    # panel A — guideline
    pa_y0, pa_h = H - 3.42, 2.5
    panel(ax, 0.12, pa_y0, W - 0.24, pa_h, 'A · Xây dựng và chốt guideline (GĐ3)', PH['label'], align='left')
    r1, r2 = pa_y0 + pa_h - 0.74, pa_y0 + 0.6
    bw, bh = 2.1, 0.72
    g1 = box(ax, c[0], r1, bw, bh, 'Guideline v1', '4 lớp BUG / FEATURE / UX / OTHER\n+ decision tree')
    c1 = box(ax, c[1], r1, bw, bh, 'Calibration batch 1', f"{f['n_calib_batch']} review ngẫu nhiên\n2 người gán độc lập")
    person(ax, c[1] + bw / 2 - 0.15, r1 + bh / 2 - 0.02, s=0.26)
    fx = box(ax, c[2], r1, bw, bh, 'Thảo luận ca lệch', 'sửa guideline,\nghi nhật ký ca khó')
    c2 = box(ax, c[2], r2, bw, bh, 'Calibration batch 2', f"{f['n_calib_batch']} review ngẫu nhiên\ngán bằng guideline đã sửa")
    person(ax, c[2] + bw / 2 - 0.15, r2 + bh / 2 - 0.02, s=0.26)
    dg = diamond(ax, c[1], r2, 1.65, 0.78, f"κ ≥ 0,6 ?\n(đạt κ = {f['kappa_calib']})")
    ok = box(ax, c[0], r2, bw, bh, 'Chốt guideline v1.2', f"+ {f['n_calib_fu']} ca FEATURE/UX\nđể thảo luận ranh giới", ec=PH['label'][1], lw=1.4)
    arrow(ax, [_bot(dS), _top(c1)])
    arrow(ax, [_right(g1), _left(c1)])
    arrow(ax, [_right(c1), _left(fx)])
    arrow(ax, [_bot(fx), _top(c2)])
    arrow(ax, [_left(c2), _right(dg)])
    arrow(ax, [_left(dg), _right(ok)], label='có', label_at=((_left(dg)[0] + _right(ok)[0]) / 2, r2 + 0.13))
    ymid = (r1 - bh / 2 + r2 + 0.39) / 2
    arrow(ax, [_top(dg), (c[1], ymid), (c[2] - 0.6, ymid), (c[2] - 0.6, r1 - bh / 2)], ls='--',
          label='không', label_at=((c[1] + c[2] - 0.6) / 2, ymid))
    # panel B — pilot
    pb_y0, pb_h = pa_y0 - 1.6, 1.42
    panel(ax, 0.12, pb_y0, W - 0.24, pb_h, 'B · Ước lượng tỷ lệ lớp và cỡ Hold-out (GĐ4)', PH['label'], align='left')
    yb = pb_y0 + 0.57
    pw = 2.35
    p1 = box(ax, c[0], yb, pw, 0.74, 'Pilot ngẫu nhiên thuần', f"{f['n_pilot']} review, 2 người gán\nκ = {f['kappa_pilot']} (guideline v1.2)")
    person(ax, c[0] + pw / 2 - 0.15, yb + 0.35, s=0.26)
    p2 = box(ax, c[1], yb, pw, 0.74, 'Tỷ lệ lớp (CI 95% Wilson)', f"ACTIONABLE {f['prev_actionable']}\nFEATURE {f['prev_feature']} (lớp hiếm nhất)")
    p3 = box(ax, c[2], yb, pw, 0.74, 'Cỡ Hold-out', f"{num(f['n_holdout'])} = {f['n_pilot']} pilot\n+ {num(f['n_holdout'] - f['n_pilot'])} ngẫu nhiên thuần (trần)")
    arrow(ax, [_right(p1), _left(p2)])
    arrow(ax, [_right(p2), _left(p3)])
    arrow(ax, [_bot(ok), _top(p1)])
    # panel C — splits
    pc_y0, pc_h = pb_y0 - 2.32, 2.12
    panel(ax, 0.12, pc_y0, W - 0.24, pc_h, 'C · Chia tập rời nhau (seed 42)', PH['label'], align='left')
    cw, ch = 1.5, 1.1
    gap = (W - 0.24 - 0.3 - 5 * cw) / 4
    xs = [0.12 + 0.15 + cw / 2 + i * (cw + gap) for i in range(5)]
    yc = pc_y0 + 1.0
    cards = [('Train', num(f['n_train']), f"{f['n_calib_train']} calibration\n+ {num(f['n_train'] - f['n_calib_train'])} mới, làm giàu\ncụm từ FEATURE/UX\n1 người + audit 15%", False),
             ('Validation', str(f['n_val']), 'rút ngẫu nhiên\n2 người gán\ncân lại FEATURE\n(amendment v1.8)', False),
             ('Hold-out', num(f['n_holdout']), f"{f['n_pilot']} pilot\n+ {num(f['n_holdout'] - f['n_pilot'])} ngẫu nhiên\n2 người gán", True),
             ('Focused', str(f['n_focused']), 'FEATURE/UX theo\ncụm từ gợi ý\n2 người gán\nbáo cáo riêng', True),
             ('Lazada test', str(f['n_lazada_test']), f"từ corpus Lazada\n{num(f['n_lazada'])} review\nngẫu nhiên thuần\n2 người gán", True)]
    cardb = []
    for x, (t, n, body, locked) in zip(xs, cards):
        b = header_box(ax, x, yc, cw, ch, f'{t} {n}', body, head_fc=LOCKED if locked else PH['label'][1],
                       ec=LOCKED if locked else PH['label'][1], fs=7.2, tfs=8.2)
        if locked:
            lock(ax, x + cw / 2 - 0.15, yc + ch / 2 - 0.135, color='white')
        cardb.append(b)
    yb_ = yc - ch / 2 - 0.08
    xl, xr = xs[2] - cw / 2, xs[4] + cw / 2
    ax.plot([xl, xl, xr, xr], [yb_ + 0.05, yb_, yb_, yb_ + 0.05], color=LOCKED, lw=1.0, zorder=3)
    ax.text((xl + c[2] - 0.15) / 2, pc_y0 + 0.11, 'Tập khóa: không dùng để huấn luyện, chọn ngưỡng hay tune prompt; mở một lần (26/10/2026)',
            ha='center', va='center', fontsize=6.6, color=LOCKED, fontweight='bold')
    gy = (pb_y0 + pc_y0 + pc_h) / 2
    arrow(ax, [_bot(p3), (c[2], gy), (xs[2], gy), _top(cardb[2])], color=GREY)
    # panel D — use of the sets
    pd_y0, pd_h = pc_y0 - 1.42, 1.25
    panel(ax, 0.12, pd_y0, W - 0.24, pd_h, 'D · Dùng các tập (GĐ5–GĐ10)', PH['label'], align='left')
    yd = pd_y0 + 0.52
    l1 = box(ax, c[0], yd, pw, 0.66, 'Learning curve', '400 → 800 → 1.200 (3 seed)\nchốt cỡ Train')
    l2 = box(ax, c[1], yd, pw, 0.66, 'Tune trên Validation', 'ngưỡng, hyperparameter, prompt\nrồi đóng băng config')
    l3 = box(ax, c[2], yd, pw, 0.66, 'Đánh giá cuối (GĐ10)', 'mở tập khóa một lần\nchấm điểm RQ1', ec=LOCKED, lw=1.3)
    arrow(ax, [_right(l1), _left(l2)])
    arrow(ax, [_right(l2), _left(l3)])
    gy2 = (pd_y0 + pd_h + pc_y0) / 2
    arrow(ax, [_bot(cardb[0]), (xs[0], gy2), (c[0], gy2), _top(l1)], color=GREY)
    arrow(ax, [_bot(cardb[1]), (xs[1], gy2), (c[1], gy2), _top(l2)], color=GREY)
    arrow(ax, [(xs[4], yb_), (xs[4], l3[1] + 0.33)], color=LOCKED, ls='--')
    return _save(fig, path)


# ---------------------------------------------------------------- 4. RQ1 cascade
def fig_cascade(f, path):
    W, H = 8.6, 9.2
    fig, ax = _fig(W, H)
    # left: cascade
    panel(ax, 0.12, 3.62, 4.3, H - 3.74, 'Cascade hai tầng (dùng cho mỗi chiến lược)', PH['rq1'])
    xc = 2.27
    rv = db(ax, xc, H - 0.85, 2.2, 0.58, 'Review (text_clean)', None, tfs=8.2)
    t1 = box(ax, xc, H - 1.8, 2.9, 0.86, 'Tầng 1 · ACTIONABLE / NOT_ACTIONABLE', 'P(ACTIONABLE) từ chiến lược A, B hoặc C\n ', tfs=8.0)
    chips_row(ax, xc, H - 2.02, [('A', STRAT['A']), ('B', STRAT['B']), ('C', STRAT['C'])], w=0.3)
    dg = diamond(ax, xc, H - 2.95, 2.15, 0.9, f"P ≥ ngưỡng ?\nA: {f['thr_bilstm']} · B: {f['thr_visobert']}\nC: nhãn trực tiếp", fs=7.0)
    other = box(ax, 3.75, H - 3.95, 1.15, 0.62, 'NOT_ACTIONABLE', '= OTHER\n(dừng)', fc=GREY_LIGHT, ec=GREY, fs=7, tfs=7.2)
    t2 = box(ax, 1.75, H - 4.45, 2.6, 0.86, 'Tầng 2 · BUG / FEATURE / UX', 'argmax, chỉ review ACTIONABLE\n ', tfs=8.0)
    chips_row(ax, 1.75, H - 4.67, [('A', STRAT['A']), ('B', STRAT['B']), ('C', STRAT['C'])], w=0.3)
    out = box(ax, 1.75, H - 5.25, 2.6, 0.45, None, None, fc='white')
    chips_row(ax, 1.75, H - 5.25, [('BUG', TYPE['BUG']), ('FEATURE', TYPE['FEATURE']), ('UX', TYPE['UX'])], h=0.24, fs=7.4)
    arrow(ax, [_bot(rv), _top(t1)])
    arrow(ax, [_bot(t1), _top(dg)])
    arrow(ax, [_right(dg), (other[0], dg[1]), _top(other)], label='không', label_at=(3.6, dg[1] + 0.12))
    arrow(ax, [_bot(dg), (xc, H - 3.72), (1.75, H - 3.72), _top(t2)], label='có (ACTIONABLE)', label_at=(xc + 0.55, H - 3.56))
    arrow(ax, [_bot(t2), _top(out)])
    ax.text(0.3, H - 3.72, f"Ngưỡng chọn trên Validation:\nRecall ACTIONABLE ≥ {f['min_recall']}\nrồi max Macro-F1 (seed 42)",
            fontsize=6.5, color=INK_2, va='center', linespacing=LS)
    # right: evaluation
    ex0 = 4.58
    panel(ax, ex0, 3.62, W - ex0 - 0.12, H - 3.74, 'Đánh giá RQ1 (Hold-out, mở một lần)', PH['rq1'])
    ecx = ex0 + (W - ex0 - 0.12) / 2
    items = [('Từng tầng', 'Tầng 1: F1 + Recall ACTIONABLE\nTầng 2: Macro-F1 + F1 từng lớp\n(trên review gold-actionable)'),
             ('Đầu-cuối 4 lớp', 'Macro-F1 trên toàn Hold-out\nso với baseline phẳng ViSoBERT\n(chỉ đối chứng, không dùng downstream)'),
             ('Chi phí (hạng nhất)', 'giây suy luận / 1.000 review\nUSD API / 1.000 review · GPU\nthời gian huấn luyện · giờ gán nhãn'),
             ('Tổng quát hóa', f"Lazada external test {f['n_lazada_test']}\nfocused FEATURE/UX {f['n_focused']} (báo riêng)")]
    y = H - 0.92
    for t, b in items:
        bb = box(ax, ecx, y, 3.55, 0.7, t, b, fs=7.2, tfs=7.8)
        y -= 0.82
    # 3x3 hybrid grid
    gy = 4.5
    ax.text(ecx, gy + 0.68, 'Ghép chéo 3 × 3 (tầng 1 × tầng 2)', ha='center', fontsize=7.8, fontweight='bold', va='center')
    s = 0.26
    gx0 = ecx - 1.5 * s + 0.2
    for i, a in enumerate('ABC'):
        ax.text(gx0 - 0.16, gy + 0.3 - i * s - s / 2, a, ha='center', va='center', fontsize=7.4, fontweight='bold', color=STRAT[a])
        ax.text(gx0 + i * s + s / 2, gy + 0.42, a, ha='center', va='center', fontsize=7.4, fontweight='bold', color=STRAT[a])
        for j in range(3):
            ax.add_patch(Rectangle((gx0 + j * s, gy + 0.3 - (i + 1) * s), s, s, fc=STRAT[a] if i == j else 'white',
                                   ec=INK_2, lw=0.6, alpha=0.85 if i == j else 1, zorder=4))
    ax.text(gx0 + 3 * s + 0.1, gy + 0.42, '← tầng 2', fontsize=6.6, va='center', color=INK_2)
    ax.text(gx0 + 3 * s + 0.12, gy - 0.08, '9 cấu hình;\nđường chéo =\ncascade thuần', fontsize=6.6, va='center', color=INK_2,
            linespacing=LS)
    ax.text(gx0 - 0.38, gy - 0.09, 'tầng 1', rotation=90, fontsize=6.6, va='center', ha='center', color=INK_2)
    # bottom: strategy lanes
    panel(ax, 0.12, 0.12, W - 0.24, 3.36, 'Ba chiến lược (cùng dữ liệu, cùng seed, dùng cho cả hai tầng)', PH['out'])
    lanes = [('A', 'Bi-LSTM + fastText', [('Tách âm tiết', 'max_len 128'), ('fastText cc.vi.300', 'pretrained, cố định'),
                                         ('Bi-LSTM', 'tầng 1: 256 · tầng 2: 128'), ('Linear + softmax', 'CE có trọng số lớp')]),
             ('B', 'ViSoBERT fine-tune', [('SentencePiece', 'max_len 256'), ('uitnlp/visobert', 'fine-tune toàn bộ'),
                                         ('Đầu phân loại', 'sequence classification'), ('softmax', 'CE có trọng số lớp')]),
             ('C', 'LLM few-shot', [('Prompt khóa', 'system + hash registry'), ('Ví dụ few-shot', f"chỉ từ Train: {f['fewshot_t1']} / {f['fewshot_t2']}"),
                                   ('gpt-4o-mini', '2024-07-18, T = 0'), ('Nhãn (JSON)', 'không huấn luyện')])]
    ly = 2.68
    for key, name, steps in lanes:
        ax.add_patch(FancyBboxPatch((0.3, ly - 0.36), 1.45, 0.72, boxstyle='round,pad=0,rounding_size=0.06', fc=STRAT[key],
                                    ec='none', zorder=3))
        ax.text(1.025, ly + 0.1, key, ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=4)
        ax.text(1.025, ly - 0.17, name, ha='center', va='center', fontsize=7.4, fontweight='bold', color='white', zorder=4)
        x = 2.55
        prevb = None
        for t, b in steps:
            bb = box(ax, x, ly, 1.42, 0.62, t, b, fs=6.7, tfs=7.5, ec=STRAT[key])
            if prevb:
                arrow(ax, [_right(prevb), _left(bb)])
            else:
                arrow(ax, [(1.75, ly), _left(bb)])
            prevb = bb
            x += 1.67
        ly -= 0.95
    ax.text(W / 2, 0.24, f"Train {num(f['n_train'])} · Validation {f['n_val']} · 3 seed (42, 1337, 2024); config và prompt khóa trước khi mở Hold-out",
            ha='center', fontsize=6.8, color=INK_2, style='italic', va='center')
    return _save(fig, path)


# ---------------------------------------------------------------- 5. RQ2 topics + temporal
def fig_rq2(f, path):
    W, H = 8.6, 8.4
    fig, ax = _fig(W, H)
    p1x, p1w = 0.12, 2.42
    p2x, p2w = p1x + p1w + 0.14, 2.42
    p3x = p2x + p2w + 0.14
    p3w = W - p3x - 0.12
    py0, ph = 0.42, H - 0.54
    panel(ax, p1x, py0, p1w, ph, 'Biểu diễn và gom cụm\n(BERTopic)', PH['rq2'])
    panel(ax, p2x, py0, p2w, ph, 'Gắn loại và\nđánh giá cụm', PH['rq2'])
    panel(ax, p3x, py0, p3w, ph, 'Temporal Issue Validation', PH['rq2'])
    bw = p1w - 0.28
    # panel 1
    c1 = p1x + p1w / 2
    y = H - 0.72
    seq1 = [('db', f"{num(f['n_actionable'])} review", 'actionable (GĐ11)', 0.62),
            ('box', 'Lọc review đủ thông tin', '≥ 3 token nội dung\n(review ngắn gắn lại sau)', 0.64),
            ('box', 'Embedding câu', 'paraphrase-multilingual-\nmpnet-base-v2 (revision cố định)', 0.64),
            ('box', 'UMAP → HDBSCAN', '5 chiều, cosine,\nrandom_state 42', 0.64),
            ('box', 'c-TF-IDF', 'token tách từ underthesea,\nbỏ từ dừng', 0.64),
            ('dia', f"min_topic_size trong\n{{15, 20, 30, 50, 75}}\nchọn theo C_v → {f['min_topic_size']}", None, 0.92),
            ('box', 'Hậu xử lý', 'giảm outlier (ngưỡng p10 cosine)\ngộp cặp cụm cosine ≥ 0,90\ngắn review ngắn theo cosine', 0.8),
            ('out', f"{f['n_topics']} cụm + {num(f['n_outlier'])} outlier", 'kết quả khóa bằng sha256', 0.6)]
    prev = None
    for kind, t, b, hh in seq1:
        y -= hh / 2
        if kind == 'db':
            cur = db(ax, c1, y, bw, hh, t, b, fs=7, tfs=8)
        elif kind == 'dia':
            cur = diamond(ax, c1, y, bw + 0.1, hh, t, fs=6.9)
        elif kind == 'out':
            cur = box(ax, c1, y, bw, hh, t, b, ec=PH['rq2'][1], lw=1.5, fs=7, tfs=8.2)
        else:
            cur = box(ax, c1, y, bw, hh, t, b, fs=6.9, tfs=7.8)
        if prev:
            arrow(ax, [_bot(prev), _top(cur)])
        prev = cur
        y -= hh / 2 + 0.2
    last1 = prev
    # panel 2
    c2 = p2x + p2w / 2
    y = H - 0.72
    seq2 = [('type', 'Loại chủ đạo của cụm', 'một loại ≥ 60% review → loại đó\nngược lại mixed → tách theo loại\n(Q14: ≥ 3 review hoặc ≥ 10%)\n ', 1.08),
            ('box', 'Đặt tên cụm bằng LLM', 'một lượt cho mọi cụm (prompt v4)\ntừ khóa + review tiêu biểu\ntên tiếng Việt + tiếng Anh', 0.84),
            ('box', 'Coherence & type purity', 'C_v theo từng cụm\ntỷ lệ loại đông nhất', 0.62),
            ('person', 'Manual topic purity', f"10 review ngẫu nhiên / cụm\n2 người chấm độc lập\n{f['n_purity_items']} phán đoán, κ = {f['kappa_purity']}", 0.84),
            ('box', 'Cờ cho BA', f"purity < 0,60 → gắn cờ\n({f['n_flag_ba']}/{f['n_topics']} cụm)", 0.6),
            ('out', 'Cụm + loại + tên + purity', 'đầu vào GĐ13 và GĐ14', 0.6)]
    prev = None
    first2 = None
    for kind, t, b, hh in seq2:
        y -= hh / 2
        cur = box(ax, c2, y, bw, hh, t, b, fs=6.9, tfs=7.8, ec=PH['rq2'][1] if kind == 'out' else INK_2,
                  lw=1.5 if kind == 'out' else 0.9)
        if kind == 'type':
            chips_row(ax, c2, y - hh / 2 + 0.17, [('BUG', TYPE['BUG']), ('FEATURE', TYPE['FEATURE']), ('UX', TYPE['UX']),
                                                   ('mixed', GREY)], h=0.19, fs=6.3, w=0.47, gap=0.04)
        if kind == 'person':
            person(ax, c2 + bw / 2 - 0.15, y + hh / 2 - 0.02, s=0.26)
        if prev:
            arrow(ax, [_bot(prev), _top(cur)])
        first2 = first2 or cur
        prev = cur
        y -= hh / 2 + 0.2
    last2 = prev
    xg1 = p1x + p1w + 0.07
    arrow(ax, [_right(last1), (xg1, last1[1]), (xg1, H - 0.62), (c2, H - 0.62), _top(first2)])
    # panel 3 — rule tree
    c3 = p3x + p3w / 2
    b1 = box(ax, c3, H - 1.08, p3w - 0.3, 0.62, 'Cửa sổ thời gian', f"T0 = 06/10/2026 · 12 cửa sổ 30 ngày\nB90 = {f['b90']} (tỷ lệ 90 ngày của corpus)", fs=6.9, tfs=7.8)
    b2 = box(ax, c3, H - 1.94, p3w - 0.3, 0.74, 'Bằng chứng mỗi issue group', 'N (12 cửa sổ), c30, c90,\nR90 = (c90 / N) / B90,\nn_win = số cửa sổ có ≥ 3 review', fs=6.9, tfs=7.8)
    arrow(ax, [_bot(b1), _top(b2)])
    dx = p3x + 0.9
    xj = p3x + 1.75
    d1 = diamond(ax, dx, H - 3.0, 1.3, 0.6, 'N ≥ 20 ?', fs=7)
    d2 = diamond(ax, dx, H - 4.0, 1.45, 0.8, 'c30 ≥ 5 và\nR90 ≥ 0,5 ?', fs=6.9)
    d3 = diamond(ax, dx, H - 5.1, 1.3, 0.62, 'n_win ≥ 6 ?', fs=7)
    d4 = diamond(ax, dx, H - 6.25, 1.3, 0.62, 'R90 < 0,5 ?', fs=7)
    ow = 1.2
    ox = p3x + p3w - 0.15 - ow / 2
    grn = dict(fc=PH['rq2'][1], ec=PH['rq2'][1], color='white', tcolor='white')
    gry = dict(fc=GREY_LIGHT, ec=GREY)
    st = {'ins1': box(ax, ox, d1[1], ow, 0.4, 'INSUFFICIENT', None, tfs=7, **gry),
          'rec': box(ax, ox, d3[1] + 0.3, ow, 0.4, 'RECURRING', None, tfs=7, **grn),
          'act': box(ax, ox, d3[1] - 0.3, ow, 0.4, 'ACTIVE', None, tfs=7, **grn),
          'pos': box(ax, ox, d4[1] + 0.3, ow, 0.4, 'POSSIBLY_RESOLVED', None, tfs=6.4, **gry),
          'ins2': box(ax, ox, d4[1] - 0.3, ow, 0.4, 'INSUFFICIENT', None, tfs=7, **gry)}
    arrow(ax, [_bot(b2), (b2[0], H - 2.52), (dx, H - 2.52), _top(d1)])
    arrow(ax, [_right(d1), _left(st['ins1'])], label='không', label_at=((_right(d1)[0] + _left(st['ins1'])[0]) / 2, d1[1] + 0.13), fs=6.4)
    arrow(ax, [_bot(d1), _top(d2)], label='có', label_at=(dx + 0.15, (d1[1] - 0.3 + d2[1] + 0.4) / 2), fs=6.4)
    arrow(ax, [_bot(d2), _top(d3)], label='có', label_at=(dx + 0.15, (d2[1] - 0.4 + d3[1] + 0.31) / 2), fs=6.4)
    for d, a, b in ((d3, 'rec', 'act'), (d4, 'pos', 'ins2')):
        ax.plot([d[0] + d[2] / 2, xj], [d[1], d[1]], color=INK_2, lw=1.0, zorder=2.5)
        arrow(ax, [(xj, d[1]), (xj, st[a][1]), _left(st[a])], label='có', label_at=(xj - 0.13, (d[1] + st[a][1]) / 2), fs=6.4)
        arrow(ax, [(xj, d[1]), (xj, st[b][1]), _left(st[b])], label='không', label_at=(xj - 0.2, (d[1] + st[b][1]) / 2), fs=6.4)
    xl = p3x + 0.1
    arrow(ax, [_left(d2), (xl, d2[1]), (xl, d4[1]), _left(d4)], label='không', label_at=(xl + 0.08, (d2[1] + d4[1]) / 2), fs=6.4)
    res = box(ax, c3, py0 + 0.55, p3w - 0.3, 0.72, 'Chỉ ACTIVE / RECURRING sang GĐ14', f"kết quả: {f['n_recurring']}/{f['n_topics']} cụm RECURRING\nđộ nhạy: đổi c30, R90, n_win từng tham số", ec=PH['rq2'][1], lw=1.5, fs=6.9, tfs=7.6)
    xg2 = p2x + p2w + 0.07
    arrow(ax, [_right(last2), (xg2, last2[1]), (xg2, H - 0.62), (c3, H - 0.62), _top(b1)])
    ax.text(W / 2, 0.2, 'Luật temporal pre-register trước khi tính trạng thái của cụm (amendment v2.5).', ha='center', fontsize=6.9,
            color=INK_2, style='italic', va='center')
    return _save(fig, path)


# ---------------------------------------------------------------- 6. RQ3 generation + evaluation
def fig_rq3(f, path):
    W, H = 8.6, 8.15
    fig, ax = _fig(W, H)
    p1x, p1w = 0.12, 2.75
    p2x, p2w = p1x + p1w + 0.13, 2.75
    p3x, p3w = p2x + p2w + 0.13, W - (p2x + p2w + 0.13) - 0.12
    py0, ph = 0.42, H - 0.54
    panel(ax, p1x, py0, p1w, ph, 'Chọn cụm và lọc evidence\n(amendment v2.6–v2.7)', PH['rq3'])
    panel(ax, p2x, py0, p2w, ph, 'Sinh Candidate Requirement\n(prompt khóa v3)', PH['rq3'])
    panel(ax, p3x, py0, p3w, ph, 'Đánh giá (GĐ15)', PH['rq3'])
    c1, bw = p1x + p1w / 2, p1w - 0.28
    top = box(ax, c1, H - 1.0, bw, 0.5, f"{f['n_topics']} cụm (đều RECURRING)", None, tfs=7.8)
    routes = [('purity < 0,25 → chuyển BA', f"cụm {f['ba_topics']}\n({num(f['n_ba_reviews'])} review, kèm review_id)", GREY_LIGHT, GREY, True),
              ('Cụm 0, 8 → sàng lọc phạm vi app', f"LLM: app_actionable / out_of_scope\nQA 40 review: đồng ý {f['qa_agree']}, κ = {f['qa_kappa']}", 'white', INK_2, False),
              ('0,25 ≤ purity < 0,60', 'lọc theo loại chủ đạo (11, 12, 14)\ntách theo loại Q14 (cụm 7)', 'white', INK_2, False),
              ('purity ≥ 0,60', 'dùng toàn bộ cụm\n(1, 3, 5, 6, 10, 15, 16)', 'white', INK_2, False)]
    y = H - 1.72
    rb = []
    for t, b, fc, ec, is_ba in routes:
        bb = box(ax, c1 + 0.12, y, bw - 0.24, 0.74, t, b, fc=fc, ec=ec, fs=6.8, tfs=7.5)
        if is_ba:
            person(ax, c1 + 0.12 + (bw - 0.24) / 2 - 0.15, y + 0.35, s=0.24)
        rb.append(bb)
        y -= 0.92
    xs_ = p1x + 0.2
    ax.plot([xs_, xs_], [top[1] - 0.25, rb[-1][1]], color=INK_2, lw=1.0, zorder=2)
    ax.plot([top[0], top[0]], [top[1] - 0.25, top[1] - 0.33], color=INK_2, lw=1.0, zorder=2)
    ax.plot([xs_, top[0]], [top[1] - 0.33, top[1] - 0.33], color=INK_2, lw=1.0, zorder=2)
    for bb in rb:
        arrow(ax, [(xs_, bb[1]), _left(bb)])
    unit = box(ax, c1, rb[-1][1] - 1.0, bw, 0.78, 'Unit = cụm hoặc cụm × loại', 'luật loại chủ đạo áp lại trên tập đã lọc\ntemporal tính lại trên tập evidence', fs=6.8, tfs=7.6)
    for bb in rb[1:]:
        ax.plot([bb[0] + bb[2] / 2, p1x + p1w - 0.07], [bb[1], bb[1]], color=INK_2, lw=1.0, zorder=2)
    ax.plot([p1x + p1w - 0.07, p1x + p1w - 0.07], [rb[1][1], unit[1] + 0.6], color=INK_2, lw=1.0, zorder=2)
    arrow(ax, [(p1x + p1w - 0.07, unit[1] + 0.6), (unit[0], unit[1] + 0.6), _top(unit)])
    u_out = box(ax, c1, unit[1] - 1.02, bw, 0.62, f"{f['n_units']} unit", f"{num(f['n_evidence_pool'])} review làm evidence", fc='white', ec=PH['rq3'][1], lw=1.5, tfs=8.2, fs=7)
    arrow(ax, [_bot(unit), _top(u_out)])
    # panel 2
    c2, bw2 = p2x + p2w / 2, p2w - 0.28
    e1 = box(ax, c2, H - 1.08, bw2, 0.66, 'Chọn review đưa vào prompt', f"{f['k_shown']} review gần tâm tập evidence\n(cosine mpnet) → nhãn E1 … E{f['k_shown']}", fs=6.8, tfs=7.6)
    e2 = box(ax, c2, H - 1.92, bw2, 0.66, 'Gọi LLM', 'gpt-4o-mini-2024-07-18\nT = 0, seed 42, prompt v3 (hash)', fs=6.8, tfs=7.6)
    arrow(ax, [_bot(e1), _top(e2)])
    ax.text(c2, H - 2.48, 'JSON schema (thứ tự sinh)', ha='center', fontsize=7.4, fontweight='bold', va='center', zorder=6,
            bbox=dict(fc=PH['rq3'][0], ec='none', pad=1.2))
    fields = [('evidence_ids', 'enum E1 … E20, 2–10 review'), ('problem_vi', 'vấn đề chung, tiếng Việt'),
              ('title_vi', 'Sửa lỗi / Bổ sung / Cải thiện trải nghiệm'), ('user_story', 'As a …, I want …, so that ….'),
              ('acceptance_criteria', '2–3 AC Given / When / Then')]
    fy = H - 2.8
    fb = []
    for i, (k, d) in enumerate(fields):
        bb = box(ax, c2, fy, bw2 - 0.1, 0.36, None, None, fc='white', ec=PH['rq3'][1])
        ax.text(c2 - (bw2 - 0.1) / 2 + 0.08, fy, k, ha='left', va='center', fontsize=6.9, fontweight='bold', zorder=6)
        ax.text(c2 + (bw2 - 0.1) / 2 - 0.08, fy, d, ha='right', va='center', fontsize=6.3, color=INK_2, zorder=6)
        fb.append(bb)
        if i:
            arrow(ax, [_bot(fb[i - 1]), _top(bb)], lw=0.8)
        fy -= 0.44
    arrow(ax, [_bot(e2), _top(fb[0])])
    dg = diamond(ax, c2, fy - 0.18, 1.25, 0.62, 'Đạt luật\nkiểm tra ?', fs=6.8)
    arrow(ax, [_bot(fb[-1]), _top(dg)])
    xr = c2 + bw2 / 2 + 0.07
    arrow(ax, [_right(dg), (xr, dg[1]), (xr, e2[1]), _right(e2)], ls='--', color=INK_2)
    ax.text(c2 + 0.86, dg[1] - 0.24, 'không: gọi lại 1 lần\nkèm danh sách vi phạm', ha='center', va='center', fontsize=6.2,
            color=INK_2, linespacing=LS, zorder=6)
    mp = box(ax, c2, dg[1] - 0.88, bw2, 0.66, 'Ánh xạ E# → review_id', 'kiểm tra backend: tồn tại, đúng cụm,\nđúng tập evidence, đúng danh sách', fs=6.7, tfs=7.6)
    arrow(ax, [_bot(dg), _top(mp)], label='có', label_at=(c2 - 0.15, (dg[1] - 0.31 + mp[1] + 0.33) / 2))
    r_out = box(ax, c2, mp[1] - 0.85, bw2, 0.62, f"{f['n_requirements']} Candidate Requirement", f"{f['n_evidence_links']} liên kết evidence", fc='white', ec=PH['rq3'][1], lw=1.5, tfs=8.2, fs=7)
    arrow(ax, [_bot(mp), _top(r_out)])
    arrow(ax, [_right(u_out), (p1x + p1w + 0.065, u_out[1]), (p1x + p1w + 0.065, H - 0.68), (c2, H - 0.68), _top(e1)])
    # panel 3
    c3, bw3 = p3x + p3w / 2, p3w - 0.28
    a1 = header_box(ax, c3, H - 1.6, bw3, 1.45, 'Tự động', 'luật QUS theo AQUSA:\nwell-formed, atomic, minimal,\nuniform, unique, dependencies\nGiven/When/Then · số AC\nevidence ID hợp lệ\ncặp User Story gần trùng', head_fc=PH['rq3'][1], fs=7.0)
    a2 = header_box(ax, c3, a1[1] - a1[3] / 2 - 0.28 - 1.1, bw3, 2.2, f"Chấm tay (toàn bộ {f['n_requirements']})", 'fidelity · actionability ·\nAC testability (thang 1–5)\nhallucination · đúng loại (0/1)\n\nmỗi review được trích:\nnêu đúng vấn đề? (0/1)\n→ evidence precision\n\n2 người độc lập\nweighted κ / Cohen κ', head_fc=PH['rq3'][1], fs=7.0)
    person(ax, c3 + bw3 / 2 - 0.16, a2[1] + a2[3] / 2 - 0.45, s=0.26)
    arrow(ax, [_right(r_out), (p2x + p2w + 0.065, r_out[1]), (p2x + p2w + 0.065, H - 0.68), (c3, H - 0.68), _top(a1)])
    arrow(ax, [_bot(a1), _top(a2)])
    d6 = db(ax, c3, a2[1] - a2[3] / 2 - 0.68, bw3, 0.78, 'D6 · structured output', 'requirements · evidence\n+ cột điểm đánh giá', fs=6.8, tfs=7.6)
    arrow(ax, [_bot(a2), _top(d6)])
    ba = box(ax, c3, d6[1] - 1.0, bw3, 0.7, 'BA duyệt Candidate', 'đối chiếu evidence ID\nrồi duyệt / sửa / loại', fs=6.8, tfs=7.6, ec=PH['rq3'][1], lw=1.5)
    person(ax, c3 + bw3 / 2 - 0.16, ba[1] + 0.33, s=0.24)
    arrow(ax, [_bot(d6), _top(ba)])
    ax.text(W / 2, 0.2, 'Không dùng LLM để kiểm tra output của LLM; chất lượng nội dung đo bằng người chấm (amendment v2.8–v2.9).',
            ha='center', fontsize=6.9, color=INK_2, style='italic', va='center')
    return _save(fig, path)


# ---------------------------------------------------------------- 7. D6 data model
D6_KEYS = {('reviews', 'review_id'): 'PK', ('reviews', 'topic_id'): 'FK', ('topics', 'topic_id'): 'PK',
           ('requirements', 'requirement_id'): 'PK', ('requirements', 'topic_id'): 'FK',
           ('requirement_evidence', 'requirement_id'): 'PK, FK', ('requirement_evidence', 'review_id'): 'PK, FK'}


def d6_tables(data_dictionary):
    """{table: [(column, key)]} from 06_output_data_model/data_dictionary.csv; human-evaluation columns (eval_*) are
    collapsed into one row when there are several."""
    out = {}
    for t, g in data_dictionary.groupby('table', sort=False):
        cols = list(g['column'])
        ev = [c for c in cols if c.startswith('eval_')]
        rows = [(c, D6_KEYS.get((t, c), '')) for c in cols if not c.startswith('eval_')]
        if len(ev) > 1:
            rows.append((f'eval_* ({len(ev)} cột chấm tay)', ''))
        rows += [(c, '') for c in ev if len(ev) == 1]
        out[t] = rows
    return out


def fig_data_model(tables, path):
    """tables: {name: [(column, 'PK' | 'FK' | 'PK, FK' | '')]} in display order (see d6_tables)."""
    W, H = 9.0, 6.0
    fig, ax = _fig(W, H)
    rh, hh, cw = 0.205, 0.34, 1.92
    geo = {}

    def draw(name, x0, ytop, dashed=False):
        cols = tables[name]
        h = hh + rh * len(cols) + 0.06
        ec = GREY if dashed else INK_2
        head = GREY if dashed else PH['data'][1]
        ax.add_patch(FancyBboxPatch((x0, ytop - h), cw, h, boxstyle='round,pad=0,rounding_size=0.05', fc='white', ec=ec,
                                    lw=1.0, ls='--' if dashed else '-', zorder=3))
        ax.add_patch(FancyBboxPatch((x0, ytop - hh), cw, hh, boxstyle='round,pad=0,rounding_size=0.05', fc=head, ec=ec,
                                    lw=1.0, zorder=3.1))
        ax.add_patch(Rectangle((x0 + 0.01, ytop - hh), cw - 0.02, hh / 2, fc=head, ec='none', zorder=3.2))
        ax.text(x0 + cw / 2, ytop - hh / 2, name, ha='center', va='center', fontsize=7.8, fontweight='bold', color='white', zorder=4)
        rows = {}
        for i, (c, key) in enumerate(cols):
            y = ytop - hh - rh * (i + 0.5) - 0.03
            ax.text(x0 + 0.08, y, c, ha='left', va='center', fontsize=6.6, zorder=4, fontweight='bold' if 'PK' in key else 'normal')
            if key:
                ax.text(x0 + cw - 0.08, y, key, ha='right', va='center', fontsize=6, color=SLOT[0] if 'FK' in key else INK_2,
                        fontweight='bold', zorder=4)
            rows[c] = y
        geo[name] = dict(x0=x0, x1=x0 + cw, top=ytop, bot=ytop - h, cx=x0 + cw / 2, rows=rows)

    top = H - 0.2
    draw('reviews', 0.15, top)
    draw('topics', 2.55, top)
    draw('requirement_evidence', 2.55, geo['topics']['bot'] - 0.5)
    draw('requirements', 4.95, top)
    draw('model_comparison', 6.95, top, dashed=True)
    R, T, E, Q = geo['reviews'], geo['topics'], geo['requirement_evidence'], geo['requirements']

    def card(x, y, s, ha):
        ax.text(x, y + 0.085, s, fontsize=6.6, color=SLOT[0], fontweight='bold', ha=ha, va='center', zorder=5)

    def line(xs, ys):
        ax.plot(xs, ys, color=SLOT[0], lw=0.95, zorder=2)

    # reviews.topic_id (N) → topics (1): straight across
    y = R['rows']['topic_id']
    line([R['x1'], T['x0']], [y, y])
    card(R['x1'] + 0.06, y, 'N', 'left')
    card(T['x0'] - 0.06, y, '1', 'right')
    # requirements.topic_id (N) → topics (1)
    y = Q['rows']['topic_id']
    line([Q['x0'], T['x1']], [y, y])
    card(Q['x0'] - 0.06, y, 'N', 'right')
    card(T['x1'] + 0.06, y, '1', 'left')
    # requirement_evidence.requirement_id (N) → requirements (1): right, then up into the bottom edge
    y = E['rows']['requirement_id']
    line([E['x1'], Q['cx'], Q['cx']], [y, y, Q['bot']])
    card(E['x1'] + 0.06, y, 'N', 'left')
    ax.text(Q['cx'] + 0.07, Q['bot'] - 0.12, '1', fontsize=6.6, color=SLOT[0], fontweight='bold', ha='left', va='center')
    # requirement_evidence.review_id (N) → reviews (1): left, then up into the bottom edge
    y = E['rows']['review_id']
    line([E['x0'], R['cx'], R['cx']], [y, y, R['bot']])
    card(E['x0'] - 0.06, y, 'N', 'right')
    ax.text(R['cx'] + 0.07, R['bot'] - 0.12, '1', fontsize=6.6, color=SLOT[0], fontweight='bold', ha='left', va='center')
    M = geo['model_comparison']
    ax.text(M['cx'], M['bot'] - 0.25, 'bảng độc lập (RQ1),\nkhông có khóa ngoài', ha='center', va='center', fontsize=6.6,
            color=INK_2, linespacing=LS)
    ax.text(0.15, 0.2, 'PK: khóa chính · FK: khóa ngoài · N / 1: quan hệ nhiều – một. reviews.text có văn bản review nên không phân phối công khai.',
            fontsize=6.8, color=INK_2, style='italic', va='center')
    return _save(fig, path)
