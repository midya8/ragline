import copy,json,subprocess,sys,unittest
from pathlib import Path
from ragline.postings_audit import SCHEMA,report,load
from tools.build_postings import build

class AuditTests(unittest.TestCase):
    def setUp(self):self.p=dict(schema=SCHEMA,vectors=[[-1,0],[0,1],[1,-1],[0,0],[1,0]],query=[1,1],k=3)
    def test_fixture_regeneration(self):self.assertEqual(Path('docs/postings-cases.json').read_text(),build())
    def test_full_witness_exact(self):
        r=report(self.p);self.assertEqual(r['order'],[1,4,2,3,0]);self.assertEqual(r['selected'],[1,4,2]);self.assertEqual(r['rows'][2]['zero_kind'],'cancellation');self.assertEqual(r['rows'][3]['zero_kind'],'empty_norm');self.assertEqual(r['counts'],dict(query_nonzero=2,posting_visits=5,candidate_count=4,implicit_zero_count=1,scan_dot_products=10,indexed_dot_products=5))
    def test_never_visited_zero_is_not_dropped(self):
        r=report(dict(schema=SCHEMA,vectors=[[-1,0],[0,1]],query=[1,0],k=1));self.assertEqual(r['selected'],[1]);self.assertEqual(r['rows'][1]['zero_kind'],'implicit')
    def test_strict_schema_and_bounds(self):
        for field,value in [('schema','wrong'),('vectors',[[0]*3]),('query',[]),('query',[1]*65),('vectors',[[1,2]]*33),('k',True),('k',-1),('k',33),('k',1.0),('vectors',[[False,0]]),('vectors',[[0.,0]]),('vectors',[[1001,0]]),('vectors',[[float('nan'),0]])]:
            p=copy.deepcopy(self.p);p[field]=value
            with self.assertRaises(ValueError):report(p)
        p=copy.deepcopy(self.p);p['extra']=1
        with self.assertRaises(ValueError):report(p)
    def test_duplicate_nonfinite_and_oversize_raw_json(self):
        for s in ['{"k":1,"k":2}','{"x":NaN}','{}x',' '*131073]:
            with self.assertRaises(ValueError):load(s)
    def test_cli_stdin_file_and_errors(self):
        c=subprocess.run([sys.executable,'-m','ragline.postings_audit','-'],input=json.dumps(self.p),capture_output=True,text=True);self.assertEqual(c.returncode,0,c.stderr);self.assertEqual(json.loads(c.stdout),report(self.p))
        for s in ['{}','{"x":NaN}','['*10000+']'*10000]:
            c=subprocess.run([sys.executable,'-m','ragline.postings_audit','-'],input=s,capture_output=True,text=True);self.assertEqual(c.returncode,2);self.assertEqual(c.stdout,'');self.assertNotIn('Traceback',c.stderr)
    def test_zero_normalization_and_no_mutation(self):
        before=copy.deepcopy(self.p);r=report(self.p);self.assertEqual(self.p,before);self.assertEqual(r,report(self.p));self.assertEqual(r['rows'][2]['cosine'],0)

if __name__=='__main__':unittest.main()
