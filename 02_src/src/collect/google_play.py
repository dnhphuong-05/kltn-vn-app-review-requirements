"""GĐ1 — crawl Google Play reviews (run LOCALLY; Kaggle has internet off by default).

Protocol v1.1: sort NEWEST, paginate with continuation_token, stop by DATE once the
oldest review in a batch is older than CUTOFF (T0 - 12 months). A safety cap guards
against very dense apps. Raw JSONL is immutable: an existing output file is never
overwritten. userName is stored only as a 12-char sha256 hash; userImage is dropped.

Amendment v1.4: Google Play caps NEWEST pagination at ~40.000 reviews, so Shopee is
crawled per star rating (filter_score_with = 1..5) and merged/deduplicated by
review_id. PII (names after greetings, phone numbers, emails) is scrubbed from `text`
and `reply_content` at write time.
"""
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from google_play_scraper import Sort, app as gp_app, reviews

APPS = {'shopee': 'com.shopee.vn', 'lazada': 'com.lazada.android'}
LANG, COUNTRY = 'vi', 'vn'
VN_TZ = timezone(timedelta(hours=7))
T0 = datetime(2026, 10, 6, tzinfo=VN_TZ)       # amendment v1.3: actual crawl date
CUTOFF = datetime(2025, 10, 6, tzinfo=VN_TZ)   # T0 - 12 months
# End of the T0 day: reviews newer than this are skipped so a crawl run after 06/10 still
# snapshots the corpus as of T0 (00:00 would drop reviews posted on 06/10 itself).
T0_END = datetime(2026, 10, 6, 23, 59, 59, tzinfo=VN_TZ)
SCORES = (1, 2, 3, 4, 5)

# ---------------------------------------------------------------- PII scrub (v1.4)
# Name = 1–5 tokens right after a greeting/thanks, each Capitalized (or a lowercase
# Vietnamese middle-name particle), ending at punctuation / end / a typical follow-up word.
_GREET = r'\b(?:cảm ơn|cám ơn|xin chào|chào|kính gửi|thân gửi|gửi|dear|hi|hello|hei)\b'
_HONOR = r'(?:bạn|anh|chị|em|quý khách|khách hàng)\b'
_NAME_END = r'(?=\s*(?:[,!.?;:]|$)|\s+(?:đã|vì|ạ|nhé|nha|rất|nhiều)\b)'
# Title-case name anywhere; in review `text` the honorific is required (brand typos
# like "Cảm ơn Shoppe" would otherwise be taken for names).
_GREETING = re.compile(
    rf'(?P<g>{_GREET}(?:\s+{_HONOR})?(?:\s*,\s*|\s+))'
    rf'(?P<name>[^\W\d_]+(?:\s+[^\W\d_]+){{0,4}}?){_NAME_END}', re.IGNORECASE)
_GREETING_HONOR = re.compile(
    rf'(?P<g>{_GREET}\s+{_HONOR}(?:\s*,\s*|\s+))'
    rf'(?P<name>[^\W\d_]+(?:\s+[^\W\d_]+){{0,4}}?){_NAME_END}', re.IGNORECASE)
# Seller replies open with "<greeting> [bạn] <display name>, …" — display names are free-form
# (lowercase, nicknames, emoji), so take everything up to the first comma/!/"đã".
_REPLY_OPENING = re.compile(
    rf'^(?P<g>\s*{_GREET}(?:\s+{_HONOR})?(?:\s*,\s*|\s+))'
    r'(?P<name>[^,!?\n]{1,60}?)(?=\s*[,!?]|\s+đã\b)', re.IGNORECASE)
# Some seller replies (Lazada) open with the display name itself: "<name>, cảm ơn bạn …",
# "<name>, chúng tôi …", "<name>, mình …", "<name>, we appreciate …" (optionally after a
# mistyped "Cho" for "Chào").
_REPLY_NAME_FIRST = re.compile(
    r'^(?P<g>\s*(?:cho\s+)?)(?P<name>[^,!?\n]{1,60}?)'
    r'(?=\s*,\s*(?:cảm ơn|cám ơn|chúng tôi|mình|we|thank|thanks|rất|xin|hy vọng)\b)', re.IGNORECASE)
_PARTICLES = {'van', 'văn', 'thi', 'thị', 'thanh', 'ngoc', 'ngọc', 'hoang', 'hoàng'}
_NOT_NAMES = {  # capitalized words that commonly follow a greeting but are not people
    'shopee', 'shoppee', 'shope', 'lazada', 'lazmall', 'shopeepay', 'shopeefood', 'lazpay',
    'shop', 'mall',
    'bạn', 'anh', 'chị', 'em', 'quý', 'khách', 'chúng', 'mình', 'tôi', 'rất', 'xin',
    'cảm', 'cám', 'chào', 'vui', 'thật', 'hi', 'hello', 'dear', 'team', 'admin', 'app', 'ứng',
    # function words that start a sentence, not a name ("Cảm ơn bạn đã đánh giá 5 sao!")
    'đã', 'vì', 'về', 'đến', 'cho', 'và', 'nhé', 'nha', 'ạ', 'nhiều', 'luôn', 'đã', 'rằng',
    'vì', 'khi', 'trong', 'với', 'là', 'sẽ', 'phản', 'những', 'sự', 'các',
}
# VN numbers: mobile 0[35789]xxxxxxxx, landline 02xxxxxxxxx, or +84/84 prefix.
_PHONE = re.compile(r'(?<![\d\-])(?:(?:\+?84[\s.]?|0)[35789](?:[\s.]?\d){8}|02(?:[\s.]?\d){9})(?![\d\-])')
_EMAIL = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')


def _is_title(t: str) -> bool:
    # "Lan", "Đạt" yes; "HIỆN", "NỮA" (all-caps shouting) no
    return t[0].isupper() and (len(t) == 1 or not t.isupper())


def _replace_name(m: re.Match) -> str:
    tokens = m.group('name').split()
    first = tokens[0]
    if not _is_title(first) or first.lower() in _NOT_NAMES:
        return m.group(0)
    for t in tokens[1:]:
        if not (_is_title(t) or t.lower() in _PARTICLES):
            return m.group(0)
    return m.group('g') + '[NAME]'


# In a reply opening the slot right after the greeting is the display name, so words that can
# be part of a name ("Anh", "Em", "Vui", "Nha") are allowed; only brands, greetings and
# sentence-starting function words mean "this is not a name".
_OPENING_NOT_NAMES = (_NOT_NAMES - {'anh', 'chị', 'em', 'vui', 'nha', 'mình'}) | {
    # interjections / sentence openers seen before a comma in seller replies
    'dạ', 'vâng', 'ồ', 'ôi', 'tiếc', 'đáng', 'chúc', 'ok', 'oke', 'hiện', 'hiện tại', 'tuy', 'nếu',
    'để', 'mong', 'hy', 'hi vọng', 'thật', 'lazada', 'shopee'}
_GREET_WORDS = {'chào', 'xin', 'cảm', 'cám', 'kính', 'thân', 'gửi', 'dear', 'hi', 'hello', 'hei'}


def _replace_opening(m: re.Match) -> str:
    name = m.group('name')
    first = name.split()[0].lower() if name.split() else ''
    if not first or first in _OPENING_NOT_NAMES or first in _GREET_WORDS or '[' in name:
        return m.group(0)
    if name.strip().lower() in ('anh', 'chị', 'em'):   # a bare pronoun, not a name
        return m.group(0)
    if len(name.split()) > 8:                          # a clause, not a display name
        return m.group(0)
    return m.group('g') + '[NAME]'


def scrub_pii(s, field: str = 'text'):
    """Return (scrubbed_text, n_replacements). None passes through.
    field='text' (user review) or 'reply' (seller reply, which addresses the user by name)."""
    if not s:
        return s, 0
    out, k1 = _EMAIL.subn('[EMAIL]', s)
    out, k2 = _PHONE.subn('[PHONE]', out)
    before = out.count('[NAME]')
    if field == 'reply':
        out = _REPLY_OPENING.sub(_replace_opening, out, count=1)
        out = _REPLY_NAME_FIRST.sub(_replace_opening, out, count=1)
        out = _GREETING.sub(_replace_name, out)
    else:
        out = _GREETING_HONOR.sub(_replace_name, out)
    return out, k1 + k2 + out.count('[NAME]') - before


# ---------------------------------------------------------------- records
def _aware(at: datetime) -> datetime:
    # google-play-scraper builds `at` with datetime.fromtimestamp -> naive LOCAL time
    # of the crawling machine; astimezone() interprets it as local and attaches tz.
    return at.astimezone(timezone.utc)


def _build(r: dict):
    user = r.get('userName') or ''
    text, n1 = scrub_pii(r.get('content'))
    reply, n2 = scrub_pii(r.get('replyContent'), 'reply')
    return {
        'review_id': r['reviewId'],
        'text': text,
        'rating': r.get('score'),
        'review_date': _aware(r['at']).isoformat(),
        'app_version': r.get('reviewCreatedVersion'),
        'thumbs_up': r.get('thumbsUpCount', 0),
        'user_hash': hashlib.sha256(user.encode('utf-8')).hexdigest()[:12] if user else None,
        'reply_content': reply,
        'replied_at': _aware(r['repliedAt']).isoformat() if r.get('repliedAt') else None,
    }, n1 + n2


def to_record(r: dict) -> dict:
    return _build(r)[0]


def _fetch(app_id, count, token, filter_score_with=None, retries=3, backoff=5.0):
    for attempt in range(retries + 1):
        try:
            return reviews(app_id, lang=LANG, country=COUNTRY, sort=Sort.NEWEST,
                           count=count, filter_score_with=filter_score_with,
                           continuation_token=token)
        except Exception as e:  # network / parsing hiccups from Google Play
            if attempt == retries:
                raise
            wait = backoff * (attempt + 1)
            print(f'  retry {attempt + 1}/{retries} sau {wait:.0f}s: {e!r}')
            time.sleep(wait)


def preview(app_id: str, n: int = 50, filter_score_with=None) -> dict:
    """Small test crawl — nothing is written to disk. For schema + app identity check."""
    info = gp_app(app_id, lang=LANG, country=COUNTRY)
    res, _ = _fetch(app_id, n, None, filter_score_with)
    return {
        'app_id': app_id,
        'title': info.get('title'),
        'developer': info.get('developer'),
        'installs': info.get('installs'),
        'library_fields': sorted(res[0].keys()) if res else [],
        'first_record': to_record(res[0]) if res else None,
        'n_fetched': len(res),
        'date_range': (min(_aware(r['at']) for r in res).isoformat(),
                       max(_aware(r['at']) for r in res).isoformat()) if res else None,
    }


# ---------------------------------------------------------------- crawl
def crawl(app_id: str, out_path: str, cutoff: datetime = CUTOFF, cap: int = 120000,
          batch: int = 200, sleep: float = 1.0, progress_every: int = 5000,
          filter_score_with=None, t0_end: datetime = T0_END) -> dict:
    """Crawl newest→oldest until a batch reaches past `cutoff`. Writes incrementally.
    Reviews dated after `t0_end` are skipped (not written) without stopping: NEWEST puts
    them first, and pagination continues back to `cutoff` as usual."""
    if os.path.exists(out_path):
        raise FileExistsError(f'{out_path} đã tồn tại — raw là bất biến, không ghi đè.')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    partial = out_path + '.partial'
    token, seen, kept, n_batches, n_scrubbed, n_after_t0 = None, set(), 0, 0, 0, 0
    oldest = newest = None
    stop_reason, next_report = None, progress_every
    tag = app_id if filter_score_with is None else f'{app_id}[{filter_score_with} sao]'
    started = datetime.now(timezone.utc)
    with open(partial, 'w', encoding='utf-8') as f:
        while True:
            res, token = _fetch(app_id, batch, token, filter_score_with)
            n_batches += 1
            if not res:
                stop_reason = 'empty_batch'
                break
            for r in res:
                if r['reviewId'] in seen:
                    continue
                if filter_score_with is not None and r.get('score') != filter_score_with:
                    raise ValueError(f"{r['reviewId']}: score={r.get('score')} ≠ filter {filter_score_with}")
                seen.add(r['reviewId'])
                if _aware(r['at']) > t0_end:
                    n_after_t0 += 1
                    continue
                rec, k = _build(r)
                n_scrubbed += k
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                kept += 1
            f.flush()
            batch_oldest = min(_aware(r['at']) for r in res)
            batch_newest = max(_aware(r['at']) for r in res)
            oldest = batch_oldest if oldest is None else min(oldest, batch_oldest)
            newest = batch_newest if newest is None else max(newest, batch_newest)
            if kept >= next_report:
                print(f'  {tag}: {kept} reviews, oldest={oldest.isoformat()}', flush=True)
                next_report += progress_every
            if batch_oldest < cutoff:
                stop_reason = 'reached_cutoff'
                break
            if token is None:
                stop_reason = 'no_more_pages'
                break
            if kept >= cap:
                stop_reason = 'safety_cap'
                break
            time.sleep(sleep)
    os.replace(partial, out_path)
    return {
        'app_id': app_id, 'filter_score_with': filter_score_with, 'out_path': out_path,
        'n_reviews': kept, 'n_batches': n_batches, 'n_pii_replacements': n_scrubbed,
        'n_excluded_after_t0': n_after_t0, 't0_end': t0_end.isoformat(),
        'stop_reason': stop_reason, 'started_utc': started.isoformat(),
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'cutoff': cutoff.isoformat(), 'cap': cap, 'batch': batch,
    }


def _count_valid(path: str):
    """Number of reviews if every line is a JSON record with review_id and there is ≥1;
    None if the file is empty or unreadable/corrupt."""
    n = 0
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not json.loads(line).get('review_id'):
                    return None
                n += 1
    except (OSError, ValueError, AttributeError):
        return None
    return n or None


def crawl_stratified(app_id: str, out_dir: str, prefix: str, scores=SCORES,
                     pause_between: float = 60.0, **kw) -> list:
    """One crawl per star rating into `<prefix>_score<k>.jsonl`; pause between strata
    to avoid rate limiting. Returns per-stratum stats (each with its own stop_reason).

    Resumable: a stratum whose file already exists and is valid is SKIPPED (never
    overwritten); its crawl stats are reloaded from the `.stats.json` sidecar. An existing
    but empty/corrupt file is deleted and that stratum is crawled again."""
    runs, crawled_before = [], False
    for s in scores:
        out = os.path.join(out_dir, f'{prefix}_score{s}.jsonl')
        sidecar = out + '.stats.json'
        if os.path.exists(out):
            n = _count_valid(out)
            if n:
                stats = {'filter_score_with': s, 'out_path': out}
                if os.path.exists(sidecar):
                    with open(sidecar, encoding='utf-8') as f:
                        stats = json.load(f)
                summ = summarize(out)
                print(f'Mức {s} sao: đã có {n} review, giữ nguyên', flush=True)
                if not accept(summ, kw.get('cutoff', CUTOFF)):
                    print(f'  Mức {s} sao chưa tới cutoff (min={summ["min_review_date"]}, '
                          f'stop={stats.get("stop_reason")}) — tệp được giữ nguyên, xem phần đánh giá mức độ đầy đủ.', flush=True)
                runs.append({**stats, **summ, 'resumed': True})
                continue
            print(f'  Mức {s} sao: tệp rỗng hoặc hỏng, thu thập lại', flush=True)
            os.remove(out)
            if os.path.exists(sidecar):
                os.remove(sidecar)
        if crawled_before:
            time.sleep(pause_between)
        print(f'===== {app_id} score={s} → {out}', flush=True)
        stats = crawl(app_id, out, filter_score_with=s, **kw)
        with open(sidecar, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        crawled_before = True
        runs.append({**stats, **summarize(out), 'resumed': False})
    return runs


def merge_dedupe(paths: list, out_path: str) -> dict:
    """Concatenate JSONL files, keep the first record per review_id."""
    if os.path.exists(out_path):
        raise FileExistsError(f'{out_path} đã tồn tại — không ghi đè.')
    seen, n_in = set(), 0
    with open(out_path + '.partial', 'w', encoding='utf-8') as out:
        for p in paths:
            with open(p, encoding='utf-8') as f:
                for line in f:
                    n_in += 1
                    rid = json.loads(line)['review_id']
                    if rid in seen:
                        continue
                    seen.add(rid)
                    out.write(line if line.endswith('\n') else line + '\n')
    os.replace(out_path + '.partial', out_path)
    return {'merged_from': [os.path.basename(p) for p in paths], 'n_in': n_in,
            'n_out': len(seen), 'n_duplicates': n_in - len(seen), 'out_path': out_path}


def scrub_file(in_path: str, out_path: str) -> dict:
    """Post-process an existing raw file into a NEW scrubbed file (original untouched)."""
    if os.path.exists(out_path):
        raise FileExistsError(f'{out_path} đã tồn tại — không ghi đè.')
    n, n_rep = 0, 0
    with open(in_path, encoding='utf-8') as f, open(out_path + '.partial', 'w', encoding='utf-8') as out:
        for line in f:
            rec = json.loads(line)
            for field in ('text', 'reply_content'):
                rec[field], k = scrub_pii(rec.get(field), 'reply' if field == 'reply_content' else 'text')
                n_rep += k
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n += 1
    os.replace(out_path + '.partial', out_path)
    return {'in_path': in_path, 'out_path': out_path, 'n_reviews': n, 'n_pii_replacements': n_rep}


# ---------------------------------------------------------------- manifest
def summarize(path: str) -> dict:
    """Recompute stats from the file on disk (source of truth for the manifest)."""
    h, n, dates, ratings, no_version = hashlib.sha256(), 0, [], Counter(), 0
    with open(path, 'rb') as f:
        for line in f:
            h.update(line)
            n += 1
            rec = json.loads(line)
            dates.append(datetime.fromisoformat(rec['review_date']))
            ratings[rec.get('rating')] += 1
            no_version += rec.get('app_version') is None
    return {
        'n_reviews': n,
        'min_review_date': min(dates).isoformat() if dates else None,
        'max_review_date': max(dates).isoformat() if dates else None,
        'rating_counts': {str(k): v for k, v in sorted(ratings.items(), key=lambda kv: str(kv[0]))},
        'app_version_null_rate': round(no_version / n, 4) if n else None,
        'sha256': h.hexdigest(),
    }


def accept(summary: dict, cutoff: datetime = CUTOFF) -> bool:
    """GĐ1 acceptance: min(review_date) <= cutoff."""
    return (summary['min_review_date'] is not None
            and datetime.fromisoformat(summary['min_review_date']) <= cutoff)


def write_manifest(runs: list, path: str) -> dict:
    import platform
    from importlib.metadata import version
    manifest = {
        'stage': 'GĐ1 data collection',
        'T0': T0.date().isoformat(),
        'cutoff': CUTOFF.isoformat(),
        'library': f"google-play-scraper=={version('google-play-scraper')}",
        'python': platform.python_version(),
        'lang': LANG, 'country': COUNTRY, 'sort': 'NEWEST',
        'review_date_tz': 'UTC (converted from library local-time `at`)',
        'pii': 'userName -> sha256[:12]; userImage not stored; text/reply_content scrubbed '
               '([NAME] after greetings, [PHONE], [EMAIL]) — amendment v1.4',
        't0_filter': f'reviews with review_date > {T0_END.isoformat()} (end of T0 day) skipped at crawl '
                     'time so a run after T0 still snapshots the corpus as of T0; counts in '
                     'n_excluded_after_t0 (no amendment: mechanism to enforce T0)',
        'runs': runs,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
