"""Sumana's source/pair register, validation, held-out split, and TERMLY snapshots. Stdlib only.

Field names follow CLAUDE.md section 6: instruction / response / tier (STATIC | TERMLY).
A pair counts as verified when verified_by and verified_at are both filled.
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
DATA = ROOT / 'data'
SOURCES = ['source_id','title','url','source_type','captured_at','public_access','notes']
PAIRS = ['id','group_id','instruction','response','tier','source_id','source_url','captured_at','verified_by','verified_at','legacy_id','notes']
EXPORT = ['id','group_id','instruction','response','tier','source_id','source_url','captured_at','verified_by','verified_at']
TIERS = ('STATIC','TERMLY')  # PRIVATE exists in the tier scheme but never enters the dataset (D4)
PRIVATE_HINTS = re.compile(r'\b(student id|roll number|attendance percentage|my grades|my marks|personal timetable|fee balance)\b', re.I)
MOJIBAKE = re.compile(r'Ã|â€')
VERSION = re.compile(r'^v\d+(\.\d+)*$')

def read_csv(path):
    if not path.exists():
        raise ValueError(f'Missing {path}. Run init first.')
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def init():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, cols in [('sources.csv',SOURCES),('pairs.csv',PAIRS)]:
        path=RAW/name
        if not path.exists(): write_csv(path,cols,[])
        print(f'{path}: ready (existing content preserved)')

def date_ok(value):
    try: date.fromisoformat(value); return True
    except (ValueError, TypeError): return False

def is_verified(p):
    return bool(p.get('verified_by','').strip() and p.get('verified_at','').strip())

def source_ids(p):
    return [s.strip() for s in p.get('source_id','').split(';') if s.strip()]

def problems(sources, pairs):
    """Every rule the dataset must satisfy; returns a list of error strings."""
    errors=[]; ids=set(); known=set(); group_tier={}
    for n,s in enumerate(sources,2):
        sid=s.get('source_id','').strip()
        if not sid or sid in known: errors.append(f'sources row {n}: missing/duplicate source_id')
        known.add(sid)
        if s.get('public_access','').lower() != 'yes': errors.append(f'sources row {n}: public_access must be yes')
        if not s.get('url','').startswith(('https://','http://','assets/')): errors.append(f'sources row {n}: url must be http(s) or an assets/ path')
        if s.get('captured_at') and not date_ok(s['captured_at']): errors.append(f'sources row {n}: invalid captured_at')
    for n,p in enumerate(pairs,2):
        pid=p.get('id','').strip()
        if not pid or pid in ids: errors.append(f'pairs row {n}: missing/duplicate id')
        ids.add(pid)
        tier=p.get('tier')
        if tier not in TIERS: errors.append(f'pairs row {n}: tier must be one of {"/".join(TIERS)} (PRIVATE never enters the dataset)')
        g=p.get('group_id','').strip()
        if not g: errors.append(f'pairs row {n}: missing group_id')
        elif group_tier.setdefault(g,tier) != tier: errors.append(f'pairs row {n}: group {g} mixes tiers')
        sids=source_ids(p)
        if not sids or any(s not in known for s in sids): errors.append(f'pairs row {n}: unregistered source_id')
        if bool(p.get('verified_by','').strip()) != bool(p.get('verified_at','').strip()):
            errors.append(f'pairs row {n}: verified_by and verified_at must be filled together')
        if is_verified(p):
            for field in ('instruction','response','source_url','captured_at'):
                if not p.get(field,'').strip(): errors.append(f'pairs row {n}: verified pair missing {field}')
        for field in ('captured_at','verified_at'):
            if p.get(field) and not date_ok(p[field]): errors.append(f'pairs row {n}: invalid {field}')
        text=' '.join(p.get(k,'') for k in ('instruction','response'))
        if PRIVATE_HINTS.search(text): errors.append(f'pairs row {n}: potential private data; manual review required')
        if MOJIBAKE.search(text): errors.append(f'pairs row {n}: mojibake (Ã or â€); re-read the source as UTF-8')
    return errors

def checked():
    pairs=read_csv(RAW/'pairs.csv')
    return pairs, problems(read_csv(RAW/'sources.csv'), pairs)

def valid_pairs():
    pairs,errors=checked()
    if errors: raise ValueError('\n'.join(errors))
    result=[p for p in pairs if is_verified(p)]
    if not result: raise ValueError('No verified pairs yet. Verify pairs against their source first.')
    return result

def version_dir(version):
    if not VERSION.match(version): raise ValueError(f'version must look like v0 or v0.3, got {version}')
    return DATA/version

def dump_jsonl(path,rows,split_name):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as out:
        for p in rows: out.write(json.dumps({**{k:p[k] for k in EXPORT},'split':split_name},ensure_ascii=False)+'\n')

def test_groups(pairs, test_size, seed):
    """Pick whole paraphrase groups for the test set, each tier getting its proportional share."""
    groups=defaultdict(list)
    for p in pairs: groups[p['group_id']].append(p)
    by_tier=defaultdict(list)
    for g,members in groups.items(): by_tier[members[0]['tier']].append(g)
    rng=random.Random(seed); chosen=set()
    for tier in sorted(by_tier):
        keys=sorted(by_tier[tier]); rng.shuffle(keys)
        quota=round(test_size*sum(len(groups[k]) for k in keys)/len(pairs)); count=0
        for k in keys:
            if count >= quota: break
            chosen.add(k); count+=len(groups[k])
    return chosen

def split(version,seed,test_size):
    if test_size < 1: raise ValueError('test-size must be positive')
    out=version_dir(version)
    if (out/'test.jsonl').exists(): raise ValueError(f'{out} already has a split: versions are immutable. Use a new version.')
    pairs=valid_pairs(); chosen=test_groups(pairs,test_size,seed)
    train=[p for p in pairs if p['group_id'] not in chosen]
    test=[p for p in pairs if p['group_id'] in chosen]
    if not train or not test:
        raise ValueError(f'Cannot split: verified={len(pairs)}, test-size={test_size} leaves train or test empty')
    dump_jsonl(out/'train.jsonl',train,'train'); dump_jsonl(out/'test.jsonl',test,'test')
    per_tier={t:{'train':sum(p['tier']==t for p in train),'test':sum(p['tier']==t for p in test)} for t in TIERS}
    manifest={'version':version,'created_at':date.today().isoformat(),'seed':seed,'requested_test_size':test_size,'train_size':len(train),'test_size':len(test),'per_tier':per_tier,'warning':'Do not train/tune on test.jsonl. Manually audit semantic near-duplicates.'}
    (out/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

def snapshot(version):
    rows=[p for p in valid_pairs() if p['tier']=='TERMLY']
    if not rows: raise ValueError('No verified TERMLY pairs to snapshot')
    target=version_dir(version)/'termly_snapshot.json'
    if target.exists(): raise ValueError(f'{target} already exists: snapshots are immutable.')
    target.parent.mkdir(parents=True,exist_ok=True)
    payload={'version':version,'created_at':date.today().isoformat(),'pairs':{p['id']:{k:p[k] for k in EXPORT} for p in rows}}
    target.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Wrote {target} ({len(rows)} TERMLY pairs)')

def compare(before_version,after_version):
    load=lambda v: json.loads((version_dir(v)/'termly_snapshot.json').read_text(encoding='utf-8'))['pairs']
    before=load(before_version); after=load(after_version); rows=[]
    for pid in sorted(before.keys()|after.keys()):
        old=before.get(pid); new=after.get(pid)
        change='added' if old is None else 'removed' if new is None else 'changed' if old['response']!=new['response'] else 'unchanged'
        rows.append({'id':pid,'change':change,'before_response':old['response'] if old else '','after_response':new['response'] if new else ''})
    target=version_dir(after_version)/f'changes_since_{before_version}.csv'
    write_csv(target,['id','change','before_response','after_response'],rows)
    print(f'Wrote {target}: {len(rows)} pair IDs compared')

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('init'); sub.add_parser('validate')
    s=sub.add_parser('split'); s.add_argument('--version',required=True); s.add_argument('--seed',type=int,default=42); s.add_argument('--test-size',type=int,default=300)
    sn=sub.add_parser('snapshot'); sn.add_argument('--version',required=True)
    c=sub.add_parser('compare'); c.add_argument('--before',required=True); c.add_argument('--after',required=True)
    args=p.parse_args()
    try:
        if args.cmd=='init': init()
        elif args.cmd=='validate':
            pairs,errors=checked()
            tiers={t:sum(x.get('tier')==t for x in pairs) for t in TIERS}
            print(f'{len(pairs)} pairs {tiers}; {sum(map(is_verified,pairs))} verified; {len(errors)} errors')
            if errors: raise ValueError('\n'.join(errors))
        elif args.cmd=='split': split(args.version,args.seed,args.test_size)
        elif args.cmd=='snapshot': snapshot(args.version)
        else: compare(args.before,args.after)
    except (ValueError,FileNotFoundError) as e: print(f'ERROR: {e}',file=sys.stderr); sys.exit(1)
if __name__=='__main__': main()
