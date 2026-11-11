"""Independent dense arithmetic and original-pipeline controls, no self-oracle."""
import copy
import itertools
import math
import random
import subprocess
import sys
import unittest
from ragline.postings import SignedPostingsIndex, PostingsRetriever
from ragline.cli import _pipeline
from ragline.embedder import Embedder
from ragline.evaluate import evaluate


def dense(vectors, query):
    scores=[]
    for row in vectors:
        dot=sum(query[j]*row[j] for j in range(len(row)))
        d=math.sqrt(sum(x*x for x in row))*math.sqrt(sum(x*x for x in query))
        scores.append(dot/d if d else 0.0)
    return scores,sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)


class PostingsTests(unittest.TestCase):
    def check(self, vectors, query):
        index=SignedPostingsIndex(vectors,dim=len(query))
        r=index.score(query,witness=True)
        expected,order=dense(vectors,query)
        self.assertEqual(r['scores'],expected)
        self.assertEqual(r['order'],order)
        expected_products=[[{'bin':j,'product':query[j]*row[j]} for j in range(len(query)) if query[j] and row[j]] for row in vectors]
        self.assertEqual(r['products'],expected_products)
        self.assertEqual(r['posting_visits'],sum(map(len,expected_products)))
        self.assertEqual(r['candidate_count'],sum(bool(x) for x in expected_products))
        self.assertEqual(r['touched'],list(map(bool,expected_products)))
        self.assertEqual(r['scan_dot_products'],len(vectors)*len(query))
        self.assertEqual(r['indexed_dot_products'],r['posting_visits'])
        self.assertEqual(r['document_norm_squared'],[sum(x*x for x in row) for row in vectors])
        return r

    def test_exhaustive_signed_cube(self):
        rows=list(itertools.product((-1,0,1),repeat=3))
        for query in rows:
            for shift in range(len(rows)):
                self.check(rows[shift:]+rows[:shift],query)

    def test_seeded_real_dense_original_arithmetic(self):
        rng=random.Random(924)
        for _ in range(400):
            dim=rng.choice([1,3,8,17,256]);n=rng.randrange(20)
            rows=[[rng.choice([0.,0.,rng.uniform(-100,100)]) for _ in range(dim)] for _ in range(n)]
            query=[rng.choice([0.,0.,rng.uniform(-100,100)]) for _ in range(dim)]
            self.check(rows,query)

    def test_implicit_zero_before_negative_and_cancellation(self):
        r=self.check([[-1,0],[0,1],[1,-1],[0,0],[1,0]],[1,1])
        self.assertEqual(r['order'],[1,4,2,3,0])
        self.assertEqual(r['touched'],[True,True,True,False,True])
        self.assertEqual(r['dots'][2:4],[0,0])

    def test_display_tie_does_not_control_ranking(self):
        r=self.check([[1000000,2],[1000000,1]],[1,0])
        self.assertEqual([round(x,6) for x in r['scores']],[1,1])
        self.assertEqual(r['order'],[1,0])

    def test_exact_tie_input_order_and_all_zero(self):
        self.assertEqual(self.check([[1,0],[2,0],[0,0]],[1,0])['order'],[0,1,2])
        self.assertEqual(self.check([[1,0],[-1,0],[0,0]],[0,0])['order'],[0,1,2])

    def test_snapshot_and_metadata(self):
        chunks=[dict(chunk_id='x',topic='x',doc_ids=['d'],text='alpha beta.')]
        e=Embedder(dim=8).fit([chunks[0]['text']]);p=PostingsRetriever(chunks,e);before=p.query('alpha')
        chunks[0]['text']='other';chunks[0]['doc_ids'].clear();e.fit(['other']);e.dim=16
        p.chunks[0]['text']='again';p.query('alpha')[0]['doc_ids'].clear()
        self.assertEqual(p.query('alpha'),before)
        self.assertIsInstance(p.vectors,tuple)

    def test_actual_hashed_pipeline_and_all_slice_values(self):
        count=0
        for seed,dim,window in itertools.product(range(5),(8,32,256),(1,2,4)):
            scan=_pipeline(seed,2,dim,window);post=_pipeline(seed,2,dim,window,'postings')
            for query in ['TCP reliable delivery','chlorophyll blue light','the an of','unknownxyz','DNS certificates','glucose carbon dioxide']:
                for k in [-3,0,1,3,100]:
                    self.assertEqual(scan.query(query,k),post.query(query,k));count+=1
            self.assertEqual(evaluate(scan),evaluate(post))
        self.assertEqual(count,1350)

    def test_cli_ask_evaluate_json_and_gate_both_sides(self):
        for args in [['ask','--question','TCP reliable delivery','--require-grounded'],['evaluate','--min-recall','1','--min-mrr','.9','--min-grounded','1'],['json'],['evaluate','--min-recall','2'],['ask','--question','the an of','--require-grounded']]:
            a=subprocess.run([sys.executable,'-m','ragline.cli',*args],capture_output=True,text=True)
            b=subprocess.run([sys.executable,'-m','ragline.cli',*args,'--backend','postings'],capture_output=True,text=True)
            self.assertEqual((a.returncode,a.stdout,a.stderr),(b.returncode,b.stdout,b.stderr))

    def test_invalid_vectors_and_queries_fail(self):
        for rows in [[[True]],[[float('nan')]],[[float('inf')]],[[1000001]],[[10**1000]],[[1],[1,2]],[[]]]:
            with self.assertRaises(ValueError):SignedPostingsIndex(rows)
        for q in [[1,2],[True],[float('nan')],[]]:
            with self.assertRaises(ValueError):SignedPostingsIndex([[1]]).score(q)
        for d in [None,0,4097,True]:
            with self.assertRaises(ValueError):SignedPostingsIndex([],dim=d)

    def test_generator_rows_and_dimension_contract(self):
        idx=SignedPostingsIndex((iter(r) for r in [[1,0],[0,1]]))
        self.assertEqual(idx.vectors,((1,0),(0,1)))
        for dim in [True,1.0,0,4097]:
            with self.assertRaises(ValueError):SignedPostingsIndex([[1]],dim=dim)

    def test_index_copy_and_empty(self):
        rows=[[1,0]];idx=SignedPostingsIndex(rows);rows[0][0]=2
        self.assertEqual(idx.vectors,((1,0),))
        self.assertEqual(SignedPostingsIndex([],dim=2).score([1,2])['order'],[])

if __name__=='__main__':unittest.main()
