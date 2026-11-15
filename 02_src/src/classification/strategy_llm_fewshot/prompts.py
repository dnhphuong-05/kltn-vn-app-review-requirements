"""Strategy C — few-shot prompts (protocol: LLM few-shot, Q4; ĐC §7.3.3).

Exemplars come ONLY from Train, are drawn once with seed 42 and stay fixed for every review. Repository
policy keeps review text out of git, so `04_prompts/classification_fewshot/` stores the system prompt and an
exemplar manifest (review_id, label, sha256 of the text); the texts are read back from Train at run time and
checked against the manifest. `prompt_hash` covers the fully rendered prompt (system + exemplar texts and
labels + JSON schema + model + temperature) and is locked in `04_prompts/prompts_registry.md`.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

from src.utils.kaggle_paths import _repo_root

PROMPT_DIR = os.path.join(_repo_root(), '04_prompts', 'classification_fewshot')
REGISTRY = os.path.join(_repo_root(), '04_prompts', 'prompts_registry.md')
LABELS = {1: ['NOT_ACTIONABLE', 'ACTIONABLE'], 2: ['BUG', 'FEATURE', 'UX'],
          'flat': ['BUG', 'FEATURE', 'UX', 'OTHER']}         # 'flat': paper, amendment v3.3
# Thứ tự ví dụ trong prompt (xen kẽ lớp). Tầng 1: 4 ACTIONABLE (BUG, FEATURE, UX, BUG — BUG phổ biến nhất
# trong actionable theo pilot) + 4 NOT_ACTIONABLE (OTHER). Tầng 2: 3 ví dụ mỗi lớp.
PLAN = {1: ['BUG', 'OTHER', 'FEATURE', 'OTHER', 'UX', 'OTHER', 'BUG', 'OTHER'],
        2: ['BUG', 'FEATURE', 'UX'] * 3}
SELECTION_RULE = ('Train only; non-ambiguous; 15 <= len(text) <= 250 chars; per gold class, rows sorted by review_id '
                  'then permuted with numpy default_rng(seed); taken in PLAN order.')


def _sha(text):
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def tier_label(gold, tier):
    if tier == 1:
        return 'NOT_ACTIONABLE' if gold == 'OTHER' else 'ACTIONABLE'
    return gold


def select_exemplars(train_df, tier, seed=42, min_chars=15, max_chars=250):
    """Fixed exemplar set drawn from Train (never Validation / Hold-out)."""
    pool = train_df[(train_df['ambiguous_flag'].astype(str) != '1')
                    & train_df['text'].astype(str).str.len().between(min_chars, max_chars)]
    rng = np.random.default_rng(seed)
    queues = {}
    for gold in sorted(set(PLAN[tier])):
        g = pool[pool['gold_label'] == gold].sort_values('review_id')
        queues[gold] = list(g.iloc[rng.permutation(len(g))].itertuples(index=False))
    rows = []
    for i, gold in enumerate(PLAN[tier], 1):
        r = queues[gold].pop(0)
        rows.append({'order': i, 'review_id': r.review_id, 'gold_label': gold, 'label': tier_label(gold, tier),
                     'text': str(r.text), 'text_sha256': _sha(r.text)})
    return pd.DataFrame(rows)


def _stem(tier):
    """File/schema stem: 'tier1', 'tier2' (unchanged since v1.1) or 'flat'."""
    return 'flat' if tier == 'flat' else f'tier{tier}'


def manifest_path(tier, version):
    return os.path.join(PROMPT_DIR, f'{_stem(tier)}_exemplars_{version}.json')


def system_path(tier, version):
    return os.path.join(PROMPT_DIR, f'{_stem(tier)}_system_{version}.txt')


def write_manifest(ex, tier, version, seed=42, rule=SELECTION_RULE):
    """Exemplar ids + labels + text hashes (no text). Refuses to overwrite a different existing manifest."""
    content = {'tier': tier, 'version': version, 'seed': seed, 'source': 'shopee_train.csv', 'rule': rule,
               'exemplars': ex.drop(columns=['text']).to_dict('records')}
    path = manifest_path(tier, version)
    if os.path.exists(path):
        old = json.load(open(path, encoding='utf-8'))
        if old != json.loads(json.dumps(content, ensure_ascii=False)):
            raise RuntimeError(f'{path} đã tồn tại với nội dung khác — prompt đã khóa, muốn đổi phải tạo version mới.')
        return path
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=1)
    return path


def schema(tier):
    return {'type': 'json_schema',
            'json_schema': {'name': f'{_stem(tier)}_label', 'strict': True,
                            'schema': {'type': 'object', 'additionalProperties': False, 'required': ['label'],
                                       'properties': {'label': {'type': 'string', 'enum': LABELS[tier]}}}}}


class FewShotPrompt:
    def __init__(self, tier, version, system, exemplars):
        self.tier, self.version, self.system, self.exemplars = tier, version, system, exemplars

    def messages(self, review_text):
        msgs = [{'role': 'system', 'content': self.system}]
        for e in self.exemplars:
            msgs.append({'role': 'user', 'content': f"Review: {e['text']}"})
            msgs.append({'role': 'assistant', 'content': json.dumps({'label': e['label']}, ensure_ascii=False)})
        msgs.append({'role': 'user', 'content': f'Review: {review_text}'})
        return msgs

    def hash(self, settings):
        payload = {'tier': self.tier, 'version': self.version, 'system': self.system,
                   'exemplars': [[e['text'], e['label']] for e in self.exemplars], 'schema': schema(self.tier),
                   'model': settings.snapshot, 'temperature': settings.temperature}
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def load_prompt(tier, version, train_df):
    """System prompt + exemplars rebuilt from Train by review_id; each text must match its manifest hash."""
    system = open(system_path(tier, version), encoding='utf-8').read().strip()
    man = json.load(open(manifest_path(tier, version), encoding='utf-8'))
    texts = train_df.set_index('review_id')['text'].astype(str)
    exemplars = []
    for e in sorted(man['exemplars'], key=lambda d: d['order']):
        t = texts[e['review_id']]
        if _sha(t) != e['text_sha256']:
            raise RuntimeError(f"Text của ví dụ {e['review_id']} khác manifest — Train đã đổi?")
        exemplars.append({**e, 'text': t})
    return FewShotPrompt(tier, version, system, exemplars)


def register(prompt, settings, date):
    """Lock: add the hash to prompts_registry.md, or verify it equals the locked one. Returns (hash, status)."""
    h = prompt.hash(settings)
    key = (f'classification_fewshot/{_stem(prompt.tier)}_system_{prompt.version}.txt + '
           f'{_stem(prompt.tier)}_exemplars_{prompt.version}.json')
    lines = open(REGISTRY, encoding='utf-8').read().splitlines()
    for line in lines:
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) >= 3 and cells[0] == key and cells[1] == prompt.version:
            if cells[2] != h:
                raise RuntimeError(f'prompt_hash {h[:12]}… khác bản đã khóa {cells[2][:12]}… cho {key} — '
                                   'prompt đã khóa, muốn đổi phải tạo version mới.')
            return h, 'khớp bản đã khóa'
    with open(REGISTRY, 'a', encoding='utf-8') as f:
        f.write(f'| {key} | {prompt.version} | {h} | {date} |\n')
    return h, 'đã đăng ký (khóa)'


# ---------------------------------------------------------------- flat 4-class prompt (paper, amendment v3.3)
FLAT_CLASS_ORDER = ['BUG', 'FEATURE', 'UX', 'OTHER']
FLAT_RULE = ('Union of the locked tier1_exemplars_v1 and tier2_exemplars_v1 (Train only, no new draw); label = 4-class '
             'gold; classes interleaved BUG, FEATURE, UX, OTHER, each class in order of first appearance (tier 1 then tier 2).')
_HARD_CASES = 'Quy tắc cho ca khó:'
_ANSWER = 'Trả lời bằng đúng một đối tượng JSON'


def _block(text, start, end):
    i, j = text.index(start) + len(start), text.index(end)
    return text[i:j].strip()


def build_flat_system(tier1_system, tier2_system):
    """Mechanical merge of the two locked tier prompts (amendment v3.3). Class definitions and both hard-case
    blocks are copied verbatim, except NOT_ACTIONABLE → OTHER and ACTIONABLE → BUG/FEATURE/UX in the Tier 1 block."""
    defs = _block(tier2_system, 'Hãy phân review vào đúng một lớp:', 'Thứ tự xét:')
    other = _block(tier1_system, 'NOT_ACTIONABLE:', 'Cây quyết định')
    actionable = _block(tier1_system, 'ACTIONABLE:', 'thuộc một trong ba loại:').rstrip(',').strip()
    intro = tier1_system.split(' và quyết định review đó')[0].strip()
    t1_rules = _block(tier1_system, _HARD_CASES, _ANSWER).replace('NOT_ACTIONABLE', 'OTHER').replace('ACTIONABLE', 'BUG/FEATURE/UX')
    t2_rules = _block(tier2_system, _HARD_CASES, _ANSWER)
    return '\n'.join([
        f'{intro} và phân review vào đúng một trong bốn lớp:',
        defs,
        f'- OTHER: {other}',
        f'BUG, FEATURE và UX chỉ dùng khi {actionable}.',
        '',
        'Cây quyết định (đi theo thứ tự):',
        '1. Review có nói về chính app/phần mềm không? Không → OTHER.',
        '2. Review có chỉ ra điều cụ thể mà đội phát triển xử lý được không? Không → OTHER.',
        '3. Có biểu hiện lỗi/malfunction → BUG.',
        '4. Có yêu cầu thêm/đổi chức năng chưa có → FEATURE.',
        '5. Chức năng có rồi nhưng khó dùng/rối/gây phiền cụ thể → UX.',
        '6. Còn lại → OTHER.',
        '',
        'Quy tắc cho ca khó giữa OTHER và BUG/FEATURE/UX:',
        t1_rules,
        '',
        'Quy tắc cho ca khó giữa BUG, FEATURE và UX:',
        t2_rules,
        '',
        f'{_ANSWER} theo schema {{"label": "BUG" | "FEATURE" | "UX" | "OTHER"}}, không giải thích.'])


def flat_exemplars(train_df, version='v1'):
    """Exemplar rows for the flat prompt from the two locked manifests (texts checked against their hashes)."""
    seen, by_class = set(), {c: [] for c in FLAT_CLASS_ORDER}
    for tier in (1, 2):
        for e in load_prompt(tier, version, train_df).exemplars:
            if e['review_id'] not in seen:
                seen.add(e['review_id'])
                by_class[e['gold_label']].append(e)
    rows, k = [], 0
    while any(by_class.values()):
        for c in FLAT_CLASS_ORDER:
            if by_class[c]:
                e = by_class[c].pop(0)
                k += 1
                rows.append({'order': k, 'review_id': e['review_id'], 'gold_label': e['gold_label'],
                             'label': e['gold_label'], 'text': e['text'], 'text_sha256': e['text_sha256']})
    return pd.DataFrame(rows)
