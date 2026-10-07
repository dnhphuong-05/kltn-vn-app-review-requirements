"""GĐ2 — minimal cleaning + metadata into 01_data/interim/ (raw stays read-only).

Steps per app (Shopee and Lazada are processed and stored separately):
  1. dedupe by review_id (re-check after merge_dedupe)
  2. cut to the study window [2025-10-06 00:00, 2026-10-06 23:59:59] (+07:00)
  3. re-apply the current PII scrub rules to `text` and `reply_content` (idempotent)
  4. text_clean = NFC + drop control/format chars + collapse whitespace. No lowercasing,
     no emoji removal, no spelling fixes (ĐC §7.1 minimal cleaning).
  5. drop rows whose text_clean is empty or has no letters (emoji/symbol/digit-only),
     keeping the dropped rows + reason in a side file for traceability
  6. flag (never delete): `near_duplicate` = identical text_clean under another review_id;
     `non_vi` / `lang_und` from a rule-based language check
"""
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

from src.collect.google_play import CUTOFF, T0_END, scrub_pii

WINDOW_START, WINDOW_END = CUTOFF, T0_END   # [2025-10-06 00:00, 2026-10-06 23:59:59] +07:00

SCHEMA = ['review_id', 'source_app', 'text', 'text_clean', 'rating', 'review_date', 'app_version',
          'thumbs_up', 'reply_content', 'replied_at', 'lang', 'flags']

# ---------------------------------------------------------------- text normalisation
_ZWJ = '‍'   # keeps multi-codepoint emoji (👨‍👩‍👧) intact
_WS = re.compile(r'\s+')


def normalize_text(s):
    if not s:
        return ''
    s = unicodedata.normalize('NFC', s)
    s = ''.join(ch for ch in s
                if ch.isspace() or ch == _ZWJ or unicodedata.category(ch) not in ('Cc', 'Cf'))
    return _WS.sub(' ', s).strip()


def has_letter(s: str) -> bool:
    return any(unicodedata.category(ch).startswith('L') for ch in s)


# ---------------------------------------------------------------- language (rule-based)
# No language-ID library is installed; short, unaccented review text also defeats most of
# them. Rules: Vietnamese-specific letters → vi; non-Latin scripts → non_vi; otherwise a
# lexicon of common unaccented Vietnamese words → vi; ≥3 Latin words with no hit → non_vi;
# anything else (e.g. "ok", "good") → und.
_VI_CHARS = re.compile('[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]',
                       re.IGNORECASE)
_OTHER_SCRIPT = re.compile('[Ѐ-ӿ฀-๿぀-ヿ㐀-鿿가-힯]')
_VI_ASCII = {
    'khong', 'ko', 'kg', 'hok', 'hong', 'dc', 'duoc', 'rat', 'qua', 'hang', 'giao', 'mua', 'tot',
    'dep', 'nhanh', 'cham', 'lam', 'roi', 'nhe', 'nha', 'cua', 'voi', 'thi', 'minh', 'toi', 'sao',
    'nao', 'nay', 'nhung', 'nhieu', 'cung', 'dang', 'loi', 'khi', 'hon', 'chua', 'tien', 'phi',
    'don', 'dung', 'xai', 'cai', 'moi', 'nua', 'luon', 'vui', 'buon', 'mat', 'ung', 'sp',
    'vs', 'nhan', 'gui', 'tra', 'hoan', 'huy', 'giam', 'gia', 'okela', 'tuyet', 'thich', 'luong',
    'san', 'pham',
    # short words / teencode seen in GĐ2 trial run ("hay", "ngon", "quang cao cc", "thua shoppe xa")
    'hay', 'ngon', 'oke', 'oki', 'okie', 'ngu', 'qc', 'quang', 'cao', 'thua', 'xa', 'bij', 'bi',
    'wa', 'hok', 'ak', 'chan', 'xau', 'te', 'nhanh', 'lun', 'mn', 'dth',
    'chuyen', 'nguoi', 'ban', 'dich', 'vu', 'khach', 'mong', 'nen', 'them', 'sua', 'cap', 'nhat',
}
_LATIN_WORD = re.compile(r'[a-z]+')


def detect_lang(s: str) -> str:
    if _VI_CHARS.search(s):
        return 'vi'
    if _OTHER_SCRIPT.search(s):
        return 'non_vi'
    words = _LATIN_WORD.findall(s.lower())
    if any(w in _VI_ASCII for w in words):
        return 'vi'
    return 'non_vi' if len(words) >= 3 else 'und'


# ---------------------------------------------------------------- PII QA
_SUSPECT = {
    'title_after_pronoun': re.compile(
        r'\b(?:bạn|anh|chị|em)\s+((?:[^\W\d_]+\s+){1,2}[^\W\d_]+)', re.IGNORECASE),
    'long_digits': re.compile(r'\d(?:[\s.\-]?\d){8,}'),
    'at_sign': re.compile(r'[\w.+-]+@[\w-]+'),
}


def pii_suspects(s):
    """Heuristic second pass for PII the scrub may have missed. Returns list of (kind, match)."""
    if not s:
        return []
    hits = []
    for m in _SUSPECT['title_after_pronoun'].finditer(s):
        toks = m.group(1).split()
        if all(t[0].isupper() and not t.isupper() for t in toks):   # Title Case run, not shouting
            hits.append(('title_after_pronoun', m.group(0)))
    hits += [('long_digits', m.group(0)) for m in _SUSPECT['long_digits'].finditer(s)]
    hits += [('at_sign', m.group(0)) for m in _SUSPECT['at_sign'].finditer(s)]
    return hits


def pii_qa(records, n_reply: int = 50, n_text: int = 30, seed: int = 42) -> dict:
    """Random sample for manual review + suspect counts over the FULL app corpus."""
    rng = random.Random(seed)
    with_reply = [r for r in records if r.get('reply_content')]
    sample_reply = rng.sample(with_reply, min(n_reply, len(with_reply)))
    sample_text = rng.sample(records, min(n_text, len(records)))
    full = Counter()
    examples = defaultdict(list)
    for r in records:
        for field in ('text', 'reply_content'):
            for kind, frag in pii_suspects(r.get(field)):
                full[(field, kind)] += 1
                if len(examples[(field, kind)]) < 8:
                    examples[(field, kind)].append((r['review_id'], frag))
    sample_hits = [(r['review_id'], f, h) for r in sample_reply + sample_text
                   for f in ('text', 'reply_content') for h in pii_suspects(r.get(f))]
    return {
        'n_sample_reply': len(sample_reply), 'n_sample_text': len(sample_text),
        'sample_reply': [(r['review_id'], r['reply_content']) for r in sample_reply],
        'sample_text': [(r['review_id'], r['text']) for r in sample_text],
        'n_suspects_in_sample': len(sample_hits), 'suspects_in_sample': sample_hits,
        'suspects_full_corpus': {f'{f}:{k}': n for (f, k), n in sorted(full.items())},
        'suspect_examples': {f'{f}:{k}': v for (f, k), v in sorted(examples.items())},
    }


# ---------------------------------------------------------------- pipeline
def _parse(ts):
    return datetime.fromisoformat(ts) if ts else None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_app(raw_path: str, app: str, out_path: str, dropped_path: str) -> dict:
    """Raw JSONL (read-only) → interim JSONL with the unified schema. Returns step counts."""
    steps = {'raw': 0}
    with open(raw_path, encoding='utf-8') as f:
        raw = [json.loads(l) for l in f]
    steps['raw'] = len(raw)

    seen, uniq = set(), []
    for r in raw:
        if r['review_id'] not in seen:
            seen.add(r['review_id'])
            uniq.append(r)
    steps['duplicate_review_id'] = len(raw) - len(uniq)

    inwin = [r for r in uniq if WINDOW_START <= _parse(r['review_date']) <= WINDOW_END]
    steps['outside_window_before'] = sum(_parse(r['review_date']) < WINDOW_START for r in uniq)
    steps['outside_window_after'] = sum(_parse(r['review_date']) > WINDOW_END for r in uniq)
    steps['after_window'] = len(inwin)

    kept, dropped, n_rescrub = [], [], 0
    for r in inwin:
        text, k1 = scrub_pii(r.get('text'), 'text')
        reply, k2 = scrub_pii(r.get('reply_content'), 'reply')
        n_rescrub += k1 + k2
        clean = normalize_text(text)
        rec = {
            'review_id': r['review_id'], 'source_app': app, 'text': text, 'text_clean': clean,
            'rating': r.get('rating'), 'review_date': r['review_date'],
            'app_version': r.get('app_version') or None, 'thumbs_up': r.get('thumbs_up', 0),
            'reply_content': reply, 'replied_at': r.get('replied_at'),
            'lang': None, 'flags': [],
        }
        if not clean:
            dropped.append({**rec, 'drop_reason': 'empty'})
        elif not has_letter(clean):
            dropped.append({**rec, 'drop_reason': 'no_letters'})   # emoji/symbol/digit-only
        else:
            kept.append(rec)
    steps['drop_empty'] = sum(d['drop_reason'] == 'empty' for d in dropped)
    steps['drop_no_letters'] = sum(d['drop_reason'] == 'no_letters' for d in dropped)
    steps['after_filter'] = len(kept)
    steps['pii_replacements_at_interim'] = n_rescrub

    by_text = Counter(r['text_clean'] for r in kept)
    for r in kept:
        r['lang'] = detect_lang(r['text_clean'])
        if by_text[r['text_clean']] > 1:
            r['flags'].append('near_duplicate')
        if r['lang'] == 'non_vi':
            r['flags'].append('non_vi')
        elif r['lang'] == 'und':
            r['flags'].append('lang_und')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in kept:
            f.write(json.dumps({k: r[k] for k in SCHEMA}, ensure_ascii=False) + '\n')
    with open(dropped_path, 'w', encoding='utf-8') as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    dates = [_parse(r['review_date']) for r in kept]
    return {
        'app': app, 'raw_path': raw_path, 'raw_sha256': _sha256(raw_path),
        'out_path': out_path, 'out_sha256': _sha256(out_path), 'dropped_path': dropped_path,
        'window': [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        'steps': steps,
        'min_review_date': min(dates).isoformat(), 'max_review_date': max(dates).isoformat(),
        'flag_counts': dict(Counter(fl for r in kept for fl in r['flags'])),
        'lang_counts': dict(Counter(r['lang'] for r in kept)),
    }


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]


def write_manifest(runs: list, path: str) -> dict:
    manifest = {
        'stage': 'GĐ2 cleaning + metadata',
        'window': [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        'cleaning': 'NFC; drop Cc/Cf except ZWJ; collapse whitespace; no lowercasing / emoji removal / '
                    'spelling fixes; PII scrub re-applied with current rules',
        'filters': 'drop empty and no-letter (emoji/symbol/digit-only) text_clean; dropped rows kept in *_dropped.jsonl',
        'flags': 'near_duplicate (identical text_clean, other review_id), non_vi / lang_und (rule-based); never deleted',
        'runs': runs,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
