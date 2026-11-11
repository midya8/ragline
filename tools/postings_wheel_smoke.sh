#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps --wheel-dir "$tmp/wheels" "$root"
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-deps "$tmp"/wheels/ragline-*.whl
cp "$root/docs/postings-cases.json" "$tmp/cases.json"
cp "$root/examples/demo.sh" "$tmp/demo.sh"
cd "$tmp"
"$tmp/venv/bin/python" - <<'PY'
import importlib,json,subprocess,sys
from pathlib import Path
from ragline.postings_audit import report
from ragline.cli import _pipeline
from ragline.evaluate import evaluate
for name in ['','answerer','cli','corpus','embedder','evaluate','retriever','postings','postings_audit']:
    m=importlib.import_module('ragline'+('.'+name if name else ''))
    assert Path(m.__file__).is_relative_to(Path(sys.prefix)),m.__file__
commands=[[sys.executable,'-m','ragline.postings_audit','-'],[str(Path(sys.prefix)/'bin/ragline-postings-audit'),'-']]
cases=json.loads(Path('cases.json').read_text())
for c in cases:
    assert report(c['input'])==c['report']
    for cmd in commands:
        r=subprocess.run(cmd,input=json.dumps(c['input']),capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        assert json.loads(r.stdout)==c['report']
for seed in range(5):
    for dim in [8,32,256]:
        for window in [1,2,4]:
            a=_pipeline(seed,2,dim,window);b=_pipeline(seed,2,dim,window,'postings')
            assert evaluate(a)==evaluate(b)
for cmd in [[sys.executable,'-m','ragline.cli'],[str(Path(sys.prefix)/'bin/ragline')]]:
    for args in [['ask','--question','TCP reliable delivery','--require-grounded'],['json'],['evaluate','--min-recall','1','--min-mrr','.9','--min-grounded','1'],['evaluate','--min-recall','2'],['ask','--question','the of an','--require-grounded']]:
        a=subprocess.run(cmd+args,capture_output=True,text=True);b=subprocess.run(cmd+args+['--backend','postings'],capture_output=True,text=True)
        assert (a.returncode,a.stdout,a.stderr)==(b.returncode,b.stdout,b.stderr)
print('Installed API/module/console:24 full reports,45 real fixture pipeline comparisons,both gate sides,9 module origins PASS')
PY
PATH="$tmp/venv/bin:$PATH" bash demo.sh
