'use strict';
(function(root){
 const SCHEMA='ragline-signed-postings/v1';
 function load(text){
  if(new TextEncoder().encode(text).length>131072)throw Error('Input exceeds 128 KiB');
  let i=0;const ws=()=>{while(/[ \t\r\n]/.test(text[i]||'')&&i<text.length)i++;};
  function string(){const start=i++;while(i<text.length){const c=text[i++];if(c==='"')return JSON.parse(text.slice(start,i));if(c==='\\')i++;}throw Error('Unclosed string');}
  function value(){ws();const c=text[i];if(c==='"')return string();if(c==='['){i++;const a=[];ws();if(text[i]===']'){i++;return a;}for(;;){a.push(value());ws();if(text[i]===']'){i++;return a;}if(text[i++]!==',')throw Error('Expected comma');}}
   if(c==='{'){i++;const a=Object.create(null),keys=new Set;ws();if(text[i]==='}'){i++;return a;}for(;;){ws();if(text[i]!=='"')throw Error('Expected field');const k=string();if(keys.has(k))throw Error('Duplicate JSON field');keys.add(k);ws();if(text[i++]!==':')throw Error('Expected colon');a[k]=value();ws();if(text[i]==='}'){i++;return a;}if(text[i++]!==',')throw Error('Expected comma');}}
   for(const [s,v] of [['true',true],['false',false],['null',null]])if(text.slice(i,i+s.length)===s){i+=s.length;return v;}
   const m=/-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(i));if(!m||m.index!==0)throw Error('Invalid JSON');i+=m[0].length;if(/[.eE]/.test(m[0]))throw Error('Use JSON integer coordinates and k, not decimal/exponent syntax');return Number(m[0]);
  }
  const r=value();ws();if(i!==text.length)throw Error('Trailing JSON data');return r;
 }
 function validate(p){
  if(!p||Array.isArray(p)||typeof p!=='object'||Object.keys(p).sort().join(',')!=='k,query,schema,vectors'||p.schema!==SCHEMA)throw Error('Expected exact schema, vectors, query and k fields');
  const q=p.query,v=p.vectors;if(!Array.isArray(q)||q.length<1||q.length>64)throw Error('Query dimension must be 1..64');
  if(!Array.isArray(v)||v.length>32||v.some(r=>!Array.isArray(r)||r.length!==q.length))throw Error('At most 32 vectors of the query dimension');
  if([q,...v].some(r=>r.some(x=>typeof x!=='number'||!Number.isInteger(x)||Math.abs(x)>1000)))throw Error('Coordinates must be integers -1000..1000');
  if(!Number.isInteger(p.k)||p.k<0||p.k>32)throw Error('k must be an integer 0..32');return p;
 }
 // Reproduce Python round(binary64, 6), including exact half-even ties, by
 // rounding its exact binary rational. Ordinary Math.round has another rule.
 function shown(x){
  if(x===0)return 0;const neg=x<0,a=Math.abs(x),dv=new DataView(new ArrayBuffer(8));dv.setFloat64(0,a);const bits=dv.getBigUint64(0),exp=Number((bits>>52n)&2047n),frac=bits&((1n<<52n)-1n);let num=(exp?frac+(1n<<52n):frac)*1000000n,den=1n;const shift=(exp?exp-1023:1-1023)-52;if(shift>=0)num<<=BigInt(shift);else den<<=BigInt(-shift);let n=num/den;const r=num%den;if(2n*r>den||(2n*r===den&&n%2n))n++;const out=Number(n)/1000000;return out===0?0:neg?-out:out;
 }
 function report(p){
  validate(p);p=JSON.parse(JSON.stringify(p));const q=p.query,v=p.vectors,n=v.length,d=q.length;
  // Independent dense row-first reference, NOT the native postings algorithm.
  const q2=q.reduce((s,x)=>s+x*x,0),norm2=v.map(r=>r.reduce((s,x)=>s+x*x,0));
  const products=v.map(r=>r.flatMap((x,b)=>x!==0&&q[b]!==0?[{bin:b,product:x*q[b]}]:[]));
  const dots=v.map(r=>r.reduce((s,x,b)=>s+x*q[b],0)),scores=dots.map((x,i)=>q2&&norm2[i]?x/(Math.sqrt(q2)*Math.sqrt(norm2[i])):0);
  const order=v.map((_,i)=>i).sort((a,b)=>scores[b]-scores[a]||a-b),selected=order.slice(0,p.k);
  const rows=v.map((r,i)=>({index:i,vector:r.slice(),touched:products[i].length>0,products:products[i],dot:dots[i],norm_squared:norm2[i],cosine:shown(scores[i]),rank:order.indexOf(i)+1,selected:selected.includes(i),zero_kind:dots[i]===0?(q2===0||norm2[i]===0?'empty_norm':products[i].length?'cancellation':'implicit'):null}));
  const visits=products.reduce((s,r)=>s+r.length,0),candidates=products.filter(r=>r.length).length;
  return {schema:SCHEMA,input:p,query_norm_squared:q2,postings:Array.from({length:d},(_,b)=>({bin:b,entries:v.flatMap((r,i)=>r[b]!==0?[{index:i,value:r[b]}]:[])})),rows,order,selected,counts:{query_nonzero:q.filter(x=>x!==0).length,posting_visits:visits,candidate_count:candidates,implicit_zero_count:n-candidates,scan_dot_products:d*n,indexed_dot_products:visits},construction:{dense_coordinates:n*d,stored_nonzero:v.reduce((s,r)=>s+r.filter(x=>x!==0).length,0),cached_norms:n},ranking:'descending full binary64 cosine; exact-score ties keep input order; rounding is display-only'};
 }
 const api={SCHEMA,load,validate,shown,report};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.RAGLINE_POSTINGS=api;
})(typeof globalThis!=='undefined'?globalThis:this);
