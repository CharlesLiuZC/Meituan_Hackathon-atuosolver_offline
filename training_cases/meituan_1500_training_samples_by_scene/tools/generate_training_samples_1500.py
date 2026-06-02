#!/usr/bin/env python3
from __future__ import annotations
import csv, json, statistics, zipfile
from pathlib import Path
import numpy as np

SOURCE = Path('/mnt/data/large_seed301(2).txt')
OUT_ZIP = Path('/mnt/data/meituan_1500_training_samples_by_scene.zip')
TARGETS = {
    'high_noise_seed601': 497.06,
    'large_seed301': 680.13,
    'large_seed302': 640.43,
    'low_willingness_seed501': 1806.96,
    'medium_seed201': 494.86,
    'medium_seed202': 527.59,
    'medium_seed203': 508.65,
    'scarce_couriers_seed401': 1554.38,
    'small_seed100': 306.91,
    'tiny_seed42': 158.65,
}
# 150 files per scene. Candidate counts are intentionally lighter than the hidden
# evaluation instances so the 1500-sample bank remains practical to store/train on.
SCENES = [
    ('high_noise_seed601', 601, 30, 70, 8000, 'high_noise'),
    ('large_seed301', 301, 40, 80, 20000, 'large301'),
    ('large_seed302', 302, 40, 80, 20000, 'large302'),
    ('low_willingness_seed501', 501, 30, 70, 8000, 'low_willingness'),
    ('medium_seed201', 201, 30, 60, 8000, 'medium201'),
    ('medium_seed202', 202, 30, 60, 8000, 'medium202'),
    ('medium_seed203', 203, 30, 60, 8000, 'medium203'),
    ('scarce_couriers_seed401', 401, 40, 34, 12000, 'scarce'),
    ('small_seed100', 100, 15, 30, 1200, 'small'),
    ('tiny_seed42', 42, 6, 12, 120, 'tiny'),
]
HEADER = 'task_id_list\tcourier_id\ttotal_score\twillingness\n'
SAMPLES_PER_SCENE = 150

def read_base(path):
    rows=[]; scores={1:[],2:[]}; wills={1:[],2:[]}
    with path.open('r', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            n=1+r['task_id_list'].count(',')
            s=float(r['total_score']); w=float(r['willingness'])
            rows.append((r['task_id_list'], r['courier_id'], s, w, n))
            scores[n].append(s); wills[n].append(w)
    return rows, {k:np.asarray(v,dtype=np.float32) for k,v in scores.items()}, {k:np.asarray(v,dtype=np.float32) for k,v in wills.items()}
BASE_ROWS, BASE_SCORES, BASE_WILLS = read_base(SOURCE)
BASE_TEXT = SOURCE.read_text(encoding='utf-8')

def bundles_for(n):
    tasks=[f'T{i:04d}' for i in range(n)]
    b=[]; lens=[]
    for t in tasks:
        b.append(t); lens.append(1)
    for i in range(n):
        for j in range(i+1,n):
            b.append(f'{tasks[i]},{tasks[j]}'); lens.append(2)
    return b, np.asarray(lens,dtype=np.int8)
BUNDLE_CACHE={}
def get_bundles(n):
    if n not in BUNDLE_CACHE: BUNDLE_CACHE[n]=bundles_for(n)
    return BUNDLE_CACHE[n]

def params(profile, rng):
    if profile=='high_noise': return 1.06, .95, 1.10, rng.uniform(10,22), 'high_noise', 0
    if profile=='low_willingness': return 1.08, 1.00, 1.08, rng.uniform(5,11), 'low', 0
    if profile=='scarce': return .99, .93, 1.01, rng.uniform(5,12), 'medium', 0
    if profile=='small': return 1.00, .96, 1.05, rng.uniform(4,12), 'normal', 0
    if profile=='tiny': return .95, .90, 1.00, rng.uniform(3,9), 'normal_high', 0
    drift={'large301':(1.0,0),'large302':(.985,.02),'medium201':(.99,.02),'medium202':(1.015,0),'medium203':(1,-.01)}.get(profile,(1,0))
    return rng.normal(drift[0],.035), rng.normal(.97,.035), rng.normal(1.02,.035), rng.uniform(3,9), 'normal_shift', drift[1]

def make_w(mode, size, rng, base, shift=0):
    if mode=='low':
        x=rng.beta(1.35,10.2,size=size)*.74+.01
        m=rng.random(size)<.08; x[m]=rng.uniform(.18,.42,int(m.sum()))
        return np.clip(x,.01,.55)
    if mode=='high_noise':
        a=rng.beta(1.2,1.25,size=size)*.89+.03
        b=np.clip(base*rng.normal(1.35,.35,size=size)+rng.normal(.06,.10,size=size),.01,.95)
        return np.where(rng.random(size)<.55,a,b)
    if mode=='normal_high':
        return np.clip(base*rng.normal(1.22,.22,size=size)+rng.normal(.07,.055,size=size),.01,.95)
    if mode=='medium':
        return np.clip(base*rng.normal(1.08,.18,size=size)+rng.normal(.04,.05,size=size),.01,.95)
    if mode=='normal_shift':
        return np.clip(base*rng.normal(1.02,.15,size=size)+shift+rng.normal(.015,.045,size=size),.01,.95)
    return np.clip(base*rng.normal(1.10,.20,size=size)+rng.normal(.03,.05,size=size),.01,.95)

def generate(scene, seed, nt, nc, cand, profile, idx):
    if scene=='large_seed301' and idx==0:
        return (BASE_TEXT if BASE_TEXT.endswith('\n') else BASE_TEXT+'\n'), {
            'num_tasks':40,'num_couriers':80,'num_candidates':len(BASE_ROWS),
            'avg_score':round(float(np.mean([r[2] for r in BASE_ROWS])),6),
            'avg_willingness':round(float(np.mean([r[3] for r in BASE_ROWS])),6),
            'single_ratio':round(float(sum(r[4]==1 for r in BASE_ROWS)/len(BASE_ROWS)),6),
            'source':'official_large_seed301_exact'
        }
    rng=np.random.default_rng(seed*100000+idx)
    bundles,lens=get_bundles(nt); nb=len(bundles); total=nb*nc; cand=min(cand,total)
    combo=rng.choice(total,size=cand,replace=False)
    bi=combo//nc; ci=combo%nc; sl=lens[bi]
    score=np.empty(cand,dtype=np.float32); wb=np.empty(cand,dtype=np.float32)
    for L in (1,2):
        m=(sl==L); cnt=int(m.sum())
        if cnt:
            score[m]=rng.choice(BASE_SCORES[L],size=cnt,replace=True)
            wb[m]=rng.choice(BASE_WILLS[L],size=cnt,replace=True)
    ss,sg,pg,noise,wm,shift=params(profile,rng)
    single=(sl==1); pair=~single
    score=score*np.where(single,sg,pg)*ss+rng.normal(0,noise,size=cand)
    if profile=='high_noise':
        out=rng.random(cand)<.11; score[out]+=rng.normal(0,28,int(out.sum()))
    if profile=='low_willingness': score[pair]+=rng.normal(5.5,3.0,int(pair.sum()))
    if profile=='scarce':
        score[pair]+=rng.normal(1.0,6.0,int(pair.sum())); score[single]+=rng.normal(-2.5,4.0,int(single.sum()))
    score=np.clip(score,10,100)
    will=make_w(wm,cand,rng,wb,shift)
    will=np.clip(will+np.where(single,rng.normal(.035,.025,cand),rng.normal(-.005,.025,cand)),.01,.95)
    courier=[f'C{i:03d}' for i in range(nc)]
    order=rng.permutation(cand)
    lines=['task_id_list\tcourier_id\ttotal_score\twillingness']
    # Slightly faster local variable lookup.
    bun=bundles; co=courier; bidx=bi; cidx=ci; sc=score; wi=will
    for k in order:
        lines.append(f'{bun[int(bidx[k])]}\t{co[int(cidx[k])]}\t{float(sc[k]):.3f}\t{float(wi[k]):.4f}')
    txt='\n'.join(lines)+'\n'
    return txt, {
        'num_tasks':nt,'num_couriers':nc,'num_candidates':cand,
        'avg_score':round(float(score.mean()),6),'avg_willingness':round(float(will.mean()),6),
        'single_ratio':round(float(single.mean()),6),'source':'augmented_from_official_large_seed301'
    }

def main():
    if OUT_ZIP.exists(): OUT_ZIP.unlink()
    manifest=[]
    readme=f'''# 1500 Meituan AutoSolver Training Samples\n\n150 synthetic-augmentation instances are provided for each requested scene.\n\nAnchor file: `case_bank/train/large_seed301/large_seed301_aug000.txt` is an exact copy of the uploaded official `large_seed301(2).txt`.\n\nSchema: `task_id_list\\tcourier_id\\ttotal_score\\twillingness`. Each row is a single-order or two-order candidate, with no duplicated `(task_id_list, courier_id)` within a file.\n\nTeacher-score targets transcribed from the feedback screenshot:\n```json\n{json.dumps(TARGETS, ensure_ascii=False, indent=2)}\n```\n\nRecommended training:\n```bash\npython training/collect_experiments.py case_bank/train --memory memory/experiments.sqlite --budget-ms 5000\npython training/tune_params.py case_bank/train/scarce_couriers_seed401 --rounds 500 --budget-ms 7000 --memory memory/experiments.sqlite\npython training/tune_params.py case_bank/train/low_willingness_seed501 --rounds 500 --budget-ms 7000 --memory memory/experiments.sqlite\npython training/train_selector.py --memory memory/experiments.sqlite --out models/strategy_selector.json\n```\n\nThese files are training augmentations, not hidden official benchmark data.\n'''
    with zipfile.ZipFile(OUT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as zf:
        zf.writestr('README_TRAINING_SAMPLES.md',readme)
        zf.writestr('tools/generate_training_samples_1500.py',Path(__file__).read_text(encoding='utf-8'))
        for scene,seed,nt,nc,cand,profile in SCENES:
            for i in range(SAMPLES_PER_SCENE):
                txt,meta=generate(scene,seed,nt,nc,cand,profile,i)
                rel=f'case_bank/train/{scene}/{scene}_aug{i:03d}.txt'
                zf.writestr(rel,txt)
                manifest.append({'path':rel,'scene':scene,'variant':i,'base_seed':seed,'profile':profile,'teacher_target_score':TARGETS[scene],**meta})
            print('generated',scene,flush=True)
        fields=['path','scene','variant','base_seed','profile','teacher_target_score','num_tasks','num_couriers','num_candidates','avg_score','avg_willingness','single_ratio','source']
        zf.writestr('manifest.csv','\n'.join([','.join(fields)]+[','.join(str(r.get(f,'')) for f in fields) for r in manifest])+'\n')
        summary={}
        for scene,*_ in SCENES:
            sr=[r for r in manifest if r['scene']==scene]
            summary[scene]={
                'files':len(sr),'teacher_target_score':TARGETS[scene],
                'num_tasks':sr[0]['num_tasks'],'num_couriers':sr[0]['num_couriers'],
                'candidate_count_min':min(r['num_candidates'] for r in sr),
                'candidate_count_max':max(r['num_candidates'] for r in sr),
                'avg_score_mean':round(statistics.mean(float(r['avg_score']) for r in sr),6),
                'avg_willingness_mean':round(statistics.mean(float(r['avg_willingness']) for r in sr),6),
                'single_ratio_mean':round(statistics.mean(float(r['single_ratio']) for r in sr),6),
            }
        zf.writestr('scene_summary.json',json.dumps(summary,ensure_ascii=False,indent=2))
    print('Wrote',OUT_ZIP,OUT_ZIP.stat().st_size/1024/1024,'MB')
if __name__=='__main__': main()
