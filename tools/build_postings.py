"""Deterministic synthetic integer-engine controls. No relevance benchmark."""
import json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ragline.postings_audit import SCHEMA,report


def cases():
    c=[]
    def add(name,v,q,k=3):c.append({'name':name,'input':dict(schema=SCHEMA,vectors=v,query=q,k=k)})
    add('Zeros must beat negative matches',[[-1,0],[0,1],[1,-1],[0,0],[1,0]],[1,1])
    add('No common bin is a real zero',[[-2,0],[0,3],[0,0]],[1,0])
    add('Touched, but exactly cancelled',[[1,-1],[2,-2],[1,0],[0,0]],[1,1])
    add('All-negative results are still results',[[-1,0],[-2,1],[-1,1]],[1,0])
    add('Input order owns exact ties',[[1,0],[2,0],[3,0],[0,0]],[1,0])
    add('Zero query keeps every input tie',[[1,2],[-2,1],[0,0]],[0,0])
    add('Empty corpus',[],[1,-1])
    add('Empty vectors are zero norm',[[0,0],[0,0]],[2,1])
    add('Request zero results',[[1,2],[2,1]],[1,0],0)
    add('Request beyond the corpus',[[1,2],[-2,-1]],[1,0],32)
    add('Same bin, opposite sign',[[1],[-1],[0]],[1])
    add('Dense query touches every posting',[[1,-2,3],[-1,2,3],[0,0,0]],[3,-2,1])
    add('Only one active bin',[[0,0,1],[0,1,0],[1,0,0]],[0,0,2])
    add('Duplicate vectors remain distinct',[[1,2],[1,2],[-1,-2]],[1,2])
    # Display rounding is deliberately not used to rank these close scores.
    add('Displayed ties can have different ranks',[[999,1],[1000,1],[998,1]],[1,0])
    add('Norm and length matter',[[1,1],[10,1],[1,10]],[1,0])
    rng=random.Random(20261111)
    for t in range(8):
        d=[3,8,16,64][t%4];n=[6,12,20,32][t%4]
        q=[rng.choice([0,0,-2,-1,1,2]) for _ in range(d)]
        v=[[rng.choice([0,0,0,-3,-1,1,3]) for _ in range(d)] for _ in range(n)]
        if t == 7:
            v=[[(-1 if i == 0 else 1) if b == i else 0 for b in range(d)] for i in range(n)]
            q=[1]+[0]*(d-1)
            add('One product, but all zero rows still rank',v,q,4)
        else:add('Seeded signed matrix '+str(t+1),v,q,min(4,n))
    return c

def build():
    c=cases();out=[dict(**x,report=report(x['input'])) for x in c]
    return json.dumps(out,ensure_ascii=True,separators=(',',':'))+'\n'
if __name__=='__main__':
    p=Path(__file__).resolve().parents[1]/'docs/postings-cases.json'
    if '--check' in sys.argv:
        assert p.read_text()==build(),'postings artifacts stale'
    else:p.write_text(build())
