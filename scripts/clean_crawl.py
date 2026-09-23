"""Triage a raw crawl into a usable corpus: drop excluded hosts, strip boilerplate, stamp capture date.

Boilerplate = any line appearing on >=BOILERPLATE_RATIO of pages (site nav, footer, social links).
On this site that is ~71% of the bytes, and it carries the Juno/ERP login link on every page.

  python3 scripts/clean_crawl.py mahindra_complete.json --captured-at 2026-09-22
"""
import argparse
import collections
import json
import re
import sys
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

# Hard rule 6 (CLAUDE.md §12): Juno never enters the project. The ERP is Juno-powered,
# and its login URL also carries a live jsessionid. Never collect these hosts.
BLOCKED_HOSTS = {'muerp.mahindrauniversity.edu.in'}
SESSION_TOKEN = re.compile(r'jsessionid|sessionid=|[?&]token=', re.I)
BOILERPLATE_RATIO = 0.8
MIN_CONTENT_CHARS = 200


def strip_boilerplate(pages):
    """Return {url: cleaned_markdown}, dropping lines common to most pages."""
    seen = collections.Counter()
    for p in pages:
        for line in {l.strip() for l in p['markdown'].split('\n') if l.strip()}:
            seen[line] += 1
    cutoff = len(pages) * BOILERPLATE_RATIO
    boiler = {l for l, c in seen.items() if c >= cutoff}
    return {p['url']: '\n'.join(l for l in (x.strip() for x in p['markdown'].split('\n'))
                                if l and l not in boiler) for p in pages}, boiler


def triage(pages):
    kept, dropped = [], []
    for p in pages:
        host = urlparse(p['url']).netloc
        if host in BLOCKED_HOSTS:
            dropped.append((p['url'], 'blocked host (Juno-powered ERP, §12 rule 6)'))
        elif SESSION_TOKEN.search(p['url']):
            dropped.append((p['url'], 'URL carries a session token'))
        elif len(p['markdown']) < MIN_CONTENT_CHARS:
            dropped.append((p['url'], f'under {MIN_CONTENT_CHARS} chars, no extractable content'))
        else:
            kept.append(p)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('crawl')
    ap.add_argument('--captured-at', default=date.today().isoformat())
    ap.add_argument('--out', default='data/raw/crawl_clean.json')
    ap.add_argument('--pdf-manifest', default='data/raw/crawl_pdfs.csv')
    a = ap.parse_args()

    pages = json.loads(Path(a.crawl).read_text(encoding='utf-8'))
    kept, dropped = triage(pages)
    if not kept:
        sys.exit('ERROR: every page was dropped; check the crawl file')

    # PDFs are found in the RAW text: the calendar/admission links sit in page bodies,
    # and the brochure links sit in the footer that boilerplate-stripping removes.
    pdfs = collections.defaultdict(set)
    for p in kept:
        for u in re.findall(r'https://[^\s\)\]"\']+\.pdf', p['markdown']):
            pdfs[u].add(p['url'])

    cleaned, boiler = strip_boilerplate(kept)
    short = [u for u, m in cleaned.items() if len(m) < MIN_CONTENT_CHARS]

    out = [{'url': p['url'], 'title': p['title'], 'markdown': cleaned[p['url']],
            'captured_at': a.captured_at} for p in kept if p['url'] not in short]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    with Path(a.pdf_manifest).open('w', newline='', encoding='utf-8') as f:
        import csv
        w = csv.writer(f)
        w.writerow(['pdf_url', 'filename', 'linked_from', 'captured_at'])
        for u in sorted(pdfs):
            w.writerow([u, u.rsplit('/', 1)[-1], ' | '.join(sorted(pdfs[u])), a.captured_at])

    raw_chars = sum(len(p['markdown']) for p in pages)
    new_chars = sum(len(p['markdown']) for p in out)
    print(f'pages in: {len(pages)}   kept: {len(out)}   dropped: {len(dropped) + len(short)}')
    for url, why in dropped:
        print(f'  DROP {why}: {url[:90]}')
    for u in short:
        print(f'  DROP nothing left after boilerplate removal: {u[:90]}')
    print(f'boilerplate lines removed: {len(boiler)}')
    print(f'chars {raw_chars:,} -> {new_chars:,} ({new_chars / raw_chars * 100:.1f}% kept)')
    print(f'wrote {a.out} and {a.pdf_manifest} ({len(pdfs)} PDFs)')


def demo():
    """Self-check: blocked host dropped, boilerplate stripped, real content kept."""
    pages = [{'url': 'https://muerp.mahindrauniversity.edu.in/login.htm;jsessionid=ABC',
              'title': 'x', 'markdown': 'Powered by JUNO Campus ' * 20},
             {'url': 'https://www.x.edu/a', 'title': 'a', 'markdown': 'NAV\nFOOTER\nreal fact A ' + 'x' * 300},
             {'url': 'https://www.x.edu/b', 'title': 'b', 'markdown': 'NAV\nFOOTER\nreal fact B ' + 'y' * 300}]
    kept, dropped = triage(pages)
    assert len(kept) == 2 and len(dropped) == 1, (kept, dropped)
    assert 'jsessionid' in dropped[0][0]
    cleaned, boiler = strip_boilerplate(kept)
    assert 'NAV' in boiler and 'FOOTER' in boiler, boiler
    assert 'real fact A' in cleaned['https://www.x.edu/a']
    assert 'NAV' not in cleaned['https://www.x.edu/a']
    print('self-check OK')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo()
    else:
        main()
