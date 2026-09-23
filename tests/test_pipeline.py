import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pipeline

SRC=[{'source_id':'S1','url':'https://x','public_access':'yes','captured_at':'2026-09-23'}]
def pair(i,tier='STATIC',group=None,**kw):
    return {'id':i,'group_id':group or i,'instruction':'q','response':'a','tier':tier,'source_id':'S1',
            'source_url':'https://x','captured_at':'2026-09-23','verified_by':'','verified_at':'',**kw}

class TestPipeline(unittest.TestCase):
    def test_date(self):
        self.assertTrue(pipeline.date_ok('2026-09-21'))
        self.assertFalse(pipeline.date_ok('2026-02-30'))
    def test_privacy_flag(self):
        self.assertTrue(pipeline.PRIVATE_HINTS.search('my grades'))
    def test_clean_rows_pass(self):
        self.assertEqual(pipeline.problems(SRC,[pair('a'),pair('b','TERMLY')]),[])
    def test_rejects(self):
        for bad in (pair('a','PRIVATE'), pair('a',response='â€™'), pair('a',source_id='S9'),
                    pair('a',verified_by='sumana'), pair('a',verified_by='s',verified_at='2026-09-24',captured_at='')):
            self.assertTrue(pipeline.problems(SRC,[bad]),bad)
        self.assertTrue(pipeline.problems(SRC,[pair('a',group='g'),pair('b','TERMLY',group='g')]))
    def test_split_is_stratified_and_keeps_groups_whole(self):
        rows=[pair(f's{i}',group=f'gs{i//2}') for i in range(80)]+[pair(f't{i}','TERMLY') for i in range(20)]
        chosen=pipeline.test_groups(rows,10,seed=1)
        test=[r for r in rows if r['group_id'] in chosen]
        self.assertEqual(sum(r['tier']=='TERMLY' for r in test),2)
        self.assertEqual(sum(r['tier']=='STATIC' for r in test),8)

if __name__=='__main__': unittest.main()
