#!/usr/bin/env python3
"""Leaderboard Eval: SynapseFlow vs Baselines on 4 Benchmarks"""
import sys,os,json,time,argparse
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.brain import call

COST={m:0.001 for m in['ds-pro','ds-think','glm','qwen','kimi']}
COST['groq']=0.0002; COST['ds-pro']=0.002; COST['kimi']=0.0015

def route_synapseflow(q):
    from engine.brainstem_wrapper import load; from engine.pareto_fep import ParetoFEP
    bs=load(); r=ParetoFEP(bs)
    d=r.route(q); return [m for ml in d['selected_models'].values() for m in ml][:1],d

def route_heuristic(q):
    ql=q.lower()
    if any(w in ql for w in['code','python','sql','function','class ','def ']): return ['ds-pro'],{}
    if any(w in ql for w in['math','prove','theorem','equation','solve','integral']): return ['ds-pro'],{}
    if any(w in ql for w in['write','essay','poem','story','creative']): return ['kimi'],{}
    if any(w in ql for w in['what','why','how','explain','define']): return ['glm'],{}
    return ['groq'],{}

ROUTERS={
    'ds-pro': (lambda q: (['ds-pro'],{})),
    'heuristic': route_heuristic,
    'synapseflow': route_synapseflow,
}

def load_benchmark(name,n):
    from datasets import load_dataset
    if name=='mmlu-pro':
        ds=load_dataset('TIGER-Lab/MMLU-Pro','main',split='test',streaming=True)
        qs=[]; cat_filter=['math']
        for d in ds:
            if d.get('category','') in cat_filter: qs.append({'q':d['question'],'a':d['answer'],'choices':[d[f'options'][i] for i in range(10) if d[f'options'][i].strip()]})
            if len(qs)>=n: break
        return qs
    elif name=='gpqa':
        ds=load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train',streaming=True)
        qs=[]
        for d in ds:
            qs.append({'q':d['Question'],'a':d['Correct Answer'],'choices':[d['Correct Answer'],d['Incorrect Answer 1'],d['Incorrect Answer 2'],d['Incorrect Answer 3']]})
            if len(qs)>=n: break
        return qs
    elif name=='bbh':
        ds=load_dataset('lukaemon/bbh','all',split='test',streaming=True)
        qs=[]
        for d in ds:
            qs.append({'q':d['input'],'a':d['target']})
            if len(qs)>=n: break
        return qs
    elif name=='ifeval':
        ds=load_dataset('google/IFEval','plain_text',split='train',streaming=True)
        qs=[]
        for d in ds:
            qs.append({'q':d['prompt'],'a':''})
            if len(qs)>=n: break
        return qs
    return []

def evaluate(resp,gt,choices=None):
    if choices:
        for i,c in enumerate(choices):
            if str(c).strip().upper()[:20]==str(gt).strip().upper()[:20]: return str(resp).upper()[:50].count(chr(65+i))>0 or str(gt).upper()[:5] in str(resp).upper()[:60]
    return str(gt).strip().upper()[:20] in str(resp).upper()[:200]

def run_one(bench,router_name,n,verbose=True):
    route_fn=ROUTERS[router_name]
    questions=load_benchmark(bench,n)
    if not questions: return None
    ok=0; cost=0; lat=0
    for i,d in enumerate(questions):
        q=d['q']; gt=d.get('a',''); choices=d.get('choices',None)
        models,meta=route_fn(q); m=models[0] if models else 'groq'
        cost+=COST.get(m,0.001)*0.3
        try:
            t0=time.time(); resp=call(m,q,max_tok=200); lat+=time.time()-t0
            ok+=1 if evaluate(resp,gt,choices) else 0
        except: pass
        if verbose and (i+1)%max(1,n//5)==0: print(f'  [{i+1}/{n}] acc={ok/(i+1):.0%}')
    return {'bench':bench,'router':router_name,'n':n,'ok':ok,'acc':ok/n,'cost':cost,'lat':lat/n}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bench',default='all')
    p.add_argument('--n',type=int,default=10)
    p.add_argument('--router',default='all')
    args=p.parse_args()

    benches=['mmlu-pro','gpqa','bbh','ifeval'] if args.bench=='all' else [args.bench]
    routers=['ds-pro','heuristic','synapseflow'] if args.router=='all' else [args.router]

    all_results=[]
    for bench in benches:
        print(f'\n{'='*60}\n{bench.upper()} ({args.n} questions)\n{'='*60}')
        for rname in routers:
            print(f'\n--- {rname} ---')
            res=run_one(bench,rname,args.n)
            if res:
                all_results.append(res)
                print(f'  {res[\"ok\"]}/{res[\"n\"]} ({res[\"acc\"]:.0%}) \${res[\"cost\"]:.4f} {res[\"lat\"]:.1f}s/q')

    print(f'\n{'='*70}')
    print(f'LEADERBOARD RESULTS')
    print(f'{\"Benchmark\":<12} {\"Router\":<15} {\"Acc\":<8} {\"Cost\":<10} {\"Lat/q\":<8}')
    print('-'*55)
    for r in all_results:
        print(f'{r[\"bench\"]:<12} {r[\"router\"]:<15} {r[\"acc\"]:.0%}     \${r[\"cost\"]:.4f}    {r[\"lat\"]:.1f}s')

    total_cost=sum(r['cost'] for r in all_results)
    print(f'\nTotal spend: \${total_cost:.4f} (' + f'within \100 budget)' if total_cost<14 else 'OVER BUDGET!')

if __name__=='__main__': main()
