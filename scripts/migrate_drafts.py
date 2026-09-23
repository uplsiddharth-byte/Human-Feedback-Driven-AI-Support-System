"""One-shot: move the 22 Sept draft CSVs onto the CLAUDE.md section 6 schema.

Reads data/mu_qa_dataset_draft.csv and data/combined_sources.csv, writes data/raw/pairs.csv and
data/raw/sources.csv. Refuses to run once pairs.csv has rows, so it can never clobber verification work.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline

DATA = pipeline.DATA
# 'Changes each term' was applied to anything that can drift. D2 reserves TERMLY for facts that expire
# within a term or admission cycle; these drift annually at most, so they are STATIC.
TO_STATIC = {'QA004','QA006','QA007','QA008','QA009','QA010','QA013','QA014','QA016','QA022','QA066','QA067',
             'QA068','QA069','QA070','QA071','QA072','QA073','QA074','QA078','QA080','QA099','QA148',
             'QA158','QA159','QA160','QA161','QA162'}
MAP_ASSET = {'SRC-M-01':'assets/campus_map_closeup.jpg','SRC-M-02':'assets/campus_map_full_sign.jpg'}

def tier(row):
    tag=row['stability_tag']
    if tag=='Changes each term' and row['qa_id'] not in TO_STATIC: return 'TERMLY'
    return 'STATIC'  # Stable, Needs review (campus map, one news item) and the reassigned rows above

def sources():
    rows=[]
    for s in pipeline.read_csv(DATA/'combined_sources.csv'):
        sid=s['source_id']
        rows.append({'source_id':sid,'title':s['title'],'url':MAP_ASSET.get(sid,s['url_or_reference']),
                     'source_type':s['source_type'],'captured_at':'','public_access':'yes','notes':s['notes']})
    return rows

def pairs(src):
    url={s['source_id']:s['url'] for s in src}; rows=[]
    for n,q in enumerate(pipeline.read_csv(DATA/'mu_qa_dataset_draft.csv'),1):
        t=tier(q); notes=[q['notes']] if q['notes'] else []
        if q['stability_tag']!='Stable': notes.append(f"draft stability_tag: {q['stability_tag']}")
        sids=pipeline.source_ids(q)
        rows.append({'id':f'mu-{n:06d}','group_id':f'mu-{n:06d}','instruction':q['question'],'response':q['answer'],
                     'tier':t,'source_id':q['source_id'],'source_url':q['source_url'] or '; '.join(url[s] for s in sids),
                     'captured_at':'','verified_by':'','verified_at':'','legacy_id':q['qa_id'],'notes':' | '.join(notes)})
    return rows

def main():
    target=pipeline.RAW/'pairs.csv'
    if target.exists() and pipeline.read_csv(target):
        sys.exit(f'{target} already has rows; migration is one-shot.')
    src=sources(); prs=pairs(src)
    pipeline.write_csv(pipeline.RAW/'sources.csv',pipeline.SOURCES,src)
    pipeline.write_csv(target,pipeline.PAIRS,prs)
    print(f'{len(src)} sources, {len(prs)} pairs written')

if __name__=='__main__': main()
