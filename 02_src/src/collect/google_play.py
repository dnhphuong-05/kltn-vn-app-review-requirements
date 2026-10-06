"""GĐ1 — crawl Google Play reviews (run LOCALLY; Kaggle has internet off by default).

Protocol v1.1: sort NEWEST, paginate with continuation_token, stop by DATE once the
oldest review in a batch is older than CUTOFF (T0 - 12 months). A safety cap guards
against very dense apps. Raw JSONL is immutable: an existing output file is never
overwritten. userName is stored only as a 12-char sha256 hash; userImage is dropped.
"""
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone

from google_play_scraper import Sort, app as gp_app, reviews

APPS = {'shopee': 'com.shopee.vn', 'lazada': 'com.lazada.android'}
LANG, COUNTRY = 'vi', 'vn'
VN_TZ = timezone(timedelta(hours=7))
T0 = datetime(2026, 10, 6, tzinfo=VN_TZ)       # amendment v1.3: actual crawl date
CUTOFF = datetime(2025, 10, 6, tzinfo=VN_TZ)   # T0 - 12 months


def _aware(at: datetime) -> datetime:
    # google-play-scraper builds `at` with datetime.fromtimestamp -> naive LOCAL time
    # of the crawling machine; astimezone() interprets it as local and attaches tz.
    return at.astimezone(timezone.utc)


def to_record(r: dict) -> dict:
    user = r.get('userName') or ''
    return {
        'review_id': r['reviewId'],
        'text': r.get('content'),
        'rating': r.get('score'),
        'review_date': _aware(r['at']).isoformat(),
        'app_version': r.get('reviewCreatedVersion'),
        'thumbs_up': r.get('thumbsUpCount', 0),
        'user_hash': hashlib.sha256(user.encode('utf-8')).hexdigest()[:12] if user else None,
        'reply_content': r.get('replyContent'),
        'replied_at': _aware(r['repliedAt']).isoformat() if r.get('repliedAt') else None,
    }


def _fetch(app_id, count, token, retries=3, backoff=5.0):
    for attempt in range(retries + 1):
        try:
            return reviews(app_id, lang=LANG, country=COUNTRY, sort=Sort.NEWEST,
                           count=count, continuation_token=token)
        except Exception as e:  # network / parsing hiccups from Google Play
            if attempt == retries:
                raise
            wait = backoff * (attempt + 1)
            print(f'  retry {attempt + 1}/{retries} sau {wait:.0f}s: {e!r}')
            time.sleep(wait)


def preview(app_id: str, n: int = 50) -> dict:
    """Small test crawl — nothing is written to disk. For schema + app identity check."""
    info = gp_app(app_id, lang=LANG, country=COUNTRY)
    res, _ = _fetch(app_id, n, None)
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


def crawl(app_id: str, out_path: str, cutoff: datetime = CUTOFF, cap: int = 120000,
          batch: int = 200, sleep: float = 1.0) -> dict:
    """Crawl newest→oldest until a batch reaches past `cutoff`. Writes incrementally."""
    if os.path.exists(out_path):
        raise FileExistsError(f'{out_path} đã tồn tại — raw là bất biến, không ghi đè.')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    partial = out_path + '.partial'
    token, seen, kept, n_batches = None, set(), 0, 0
    oldest = newest = None
    stop_reason = None
    started = datetime.now(timezone.utc)
    with open(partial, 'w', encoding='utf-8') as f:
        while True:
            res, token = _fetch(app_id, batch, token)
            n_batches += 1
            if not res:
                stop_reason = 'empty_batch'
                break
            for r in res:
                if r['reviewId'] in seen:
                    continue
                seen.add(r['reviewId'])
                f.write(json.dumps(to_record(r), ensure_ascii=False) + '\n')
                kept += 1
            f.flush()
            batch_oldest = min(_aware(r['at']) for r in res)
            batch_newest = max(_aware(r['at']) for r in res)
            oldest = batch_oldest if oldest is None else min(oldest, batch_oldest)
            newest = batch_newest if newest is None else max(newest, batch_newest)
            if n_batches % 10 == 0:
                print(f'  {app_id}: {kept} reviews, oldest={oldest.date()}')
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
        'app_id': app_id, 'out_path': out_path, 'n_reviews': kept, 'n_batches': n_batches,
        'stop_reason': stop_reason, 'started_utc': started.isoformat(),
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'cutoff': cutoff.isoformat(), 'cap': cap, 'batch': batch,
    }


def summarize(path: str) -> dict:
    """Recompute stats from the file on disk (source of truth for the manifest)."""
    h, n, dates = hashlib.sha256(), 0, []
    with open(path, 'rb') as f:
        for line in f:
            h.update(line)
            n += 1
            dates.append(datetime.fromisoformat(json.loads(line)['review_date']))
    return {
        'n_reviews': n,
        'min_review_date': min(dates).isoformat() if dates else None,
        'max_review_date': max(dates).isoformat() if dates else None,
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
        'pii': 'userName -> sha256[:12]; userImage not stored',
        'runs': runs,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
