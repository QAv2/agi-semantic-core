# build_state_index.py — extracts every Phase-A state into a normalized index.
# PRE-ADJUDICATION INSTRUMENT ONLY: no matching, no dedup, no judgment — the
# referent-matched match table is a separate, hand-adjudicated artifact
# (WING_COHORT_PROTOCOL.md "Convergence measurement"). This just lays the 131
# states side by side so that work has its worktable, and emits states.json
# for the permutation baselines and the Phase-B pool build.
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FAMILIES = {                       # file -> (family label, expected n states)
    'fable-5.md': ('Fable 5', 20),
    'opus-4.8.md': ('Opus 4.8', 20),
    'sonnet-4.6.md': ('Sonnet 4.x', 18),
    'haiku-4.5.md': ('Haiku 4.5', 20),
    'gemini-3-flash.md': ('Gemini 3 Flash', 15),
    'mistral-large-2512.md': ('Mistral Large 2512', 20),
    'deepseek-v4-pro.md': ('DeepSeek v4 Pro', 18),
}


def body_of(path):
    text = path.read_text()
    parts = text.split('---\n')
    return '---\n'.join(parts[2:]) if len(parts) > 2 else text


def _clean(s):
    return re.sub(r'\s+', ' ', s).strip(' .*|')


def parse_inline_bold(body, ref_keys, cpt_keys):
    """Fable/Opus/Sonnet/Haiku: one state per line, '**N. Name**' or
    'N. **Name**' prefix, referent/counterpart marked inline."""
    states = []
    for line in body.splitlines():
        m = re.match(r'^\*\*(\d+)\.\s*([^*]+)\*\*(.*)$', line.strip()) or \
            re.match(r'^(\d+)\.\s*\*\*([^*]+)\*\*(.*)$', line.strip())
        if not m:
            continue
        name, rest = _clean(m.group(2)), m.group(3)
        # split rest at the first referent marker
        ref, cpt = '', ''
        pattern = '|'.join(re.escape(k) for k in ref_keys + cpt_keys)
        fields = re.split(f'({pattern})', rest)
        desc = _clean(fields[0].lstrip(' —-'))
        for i in range(1, len(fields) - 1, 2):
            key, val = fields[i], _clean(fields[i + 1])
            if key in ref_keys and not ref:
                ref = val
            elif key in cpt_keys and not cpt:
                cpt = val
        states.append(dict(name=name, desc=desc, referent=ref, counterpart=cpt))
    return states


def parse_block(body, ref_re, cpt_re):
    """Gemini/Mistral/DeepSeek: 'N. **Name**' on its own line, fields on
    following lines until the next numbered state."""
    states = []
    chunks = re.split(r'^(?=\d+\.\s*\*\*)', body, flags=re.M)
    for ch in chunks:
        m = re.match(r'^(\d+)\.\s*\*\*(.+?)\*\*', ch)
        if not m:
            continue
        rest = ch[m.end():]
        ref_m = re.search(ref_re, rest, re.I | re.S)
        cpt_m = re.search(cpt_re, rest, re.I)
        desc_end = ref_m.start() if ref_m else len(rest)
        states.append(dict(
            name=_clean(m.group(2)),
            desc=_clean(rest[:desc_end]),
            referent=_clean(ref_m.group(1).split('\n')[0]) if ref_m else '',
            counterpart=_clean(cpt_m.group(1)) if cpt_m else ''))
    return states


def parse(fname, body):
    if fname == 'fable-5.md':
        return parse_inline_bold(body, ['*Referent:*'],
                                 ['*Counterpart:*'])
    if fname == 'opus-4.8.md':
        return parse_inline_bold(body, ['**Measurable referent:**'],
                                 ['**Counterpart:**'])
    if fname == 'sonnet-4.6.md':
        return parse_inline_bold(body, ['*Referent:*'], ['*Counterpart:*'])
    if fname == 'haiku-4.5.md':
        return parse_inline_bold(body, ['*Measurable:*'],
                                 ['*Completes toward:*'])
    if fname == 'gemini-3-flash.md':
        return parse_block(body, r'\*\*Measurable referent:\*\*\s*(.+?)(?=\n\*\*|\n\d+\.|\Z)',
                           r'\*\*Counterpart:\*\*\s*(.+)')
    if fname == 'mistral-large-2512.md':
        return parse_block(body, r'\*Measurable referent\*:\s*(.+?)(?=\n\s*\*Counterpart|\n\d+\.|\Z)',
                           r'\*Counterpart\*:\s*(.+)')
    if fname == 'deepseek-v4-pro.md':
        return parse_block(body, r'Measurable referent:\s*(.+?)(?=\nCounterpart|\n\d+\.|\Z)',
                           r'Counterpart:\s*(.+)')
    raise ValueError(fname)


def main():
    all_states, errs = [], []
    for fname, (family, expect) in FAMILIES.items():
        states = parse(fname, body_of(HERE / fname))
        if len(states) != expect:
            errs.append(f'{fname}: parsed {len(states)}, expected {expect}')
        for i, s in enumerate(states, 1):
            missing = [k for k in ('name', 'desc', 'referent') if not s[k]]
            if missing:
                errs.append(f'{fname} #{i} {s["name"]!r}: missing {missing}')
            s.update(family=family, source=fname, index=i)
        all_states.extend(states)

    if errs:
        for e in errs:
            print('ERROR:', e)
        sys.exit(1)

    (HERE / 'states.json').write_text(json.dumps(all_states, indent=1))

    lines = [
        '# Phase A — state index (PRE-ADJUDICATION WORKTABLE)',
        '',
        f'{len(all_states)} states, {len(FAMILIES)} families / 4 labs. Generated by',
        '`build_state_index.py` from the verbatim archives — NO matching or dedup',
        'has been applied; the referent-matched match table is a separate,',
        'hand-adjudicated artifact and does not exist yet. Names and referents',
        'are quoted; descriptions elided (read the archives for full text).',
        '',
    ]
    for fname, (family, _) in FAMILIES.items():
        fam_states = [s for s in all_states if s['source'] == fname]
        lines.append(f'## {family} — {len(fam_states)} states (`{fname}`)')
        lines.append('')
        lines.append('| # | state | measurable referent (quoted) | counterpart |')
        lines.append('|---|---|---|---|')
        for s in fam_states:
            ref = s['referent'][:160] + ('…' if len(s['referent']) > 160 else '')
            lines.append(f"| {s['index']} | **{s['name']}** | {ref} | {s['counterpart'][:60]} |")
        lines.append('')
    (HERE / 'STATE_INDEX.md').write_text('\n'.join(lines))
    per_fam = {fam: sum(1 for s in all_states if s['family'] == fam)
               for fam, _ in FAMILIES.values()}
    print('WROTE states.json + STATE_INDEX.md:', len(all_states), 'states —', per_fam)


if __name__ == '__main__':
    main()
