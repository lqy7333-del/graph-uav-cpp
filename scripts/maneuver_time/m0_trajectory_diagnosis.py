"""M0: read-only trajectory/mechanism diagnosis for frozen framework results."""
import argparse, hashlib, json, pickle, time, traceback
from pathlib import Path
import sys
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.analysis.framework_mechanism import replay_result, initial_features, load_frozen
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework')
METHODS=('cc','divide_then_plan','plan_then_divide')

def write_csv(path, rows):
    if not rows: path.write_text(''); return
    keys=sorted({k for row in rows for k in row})
    import csv
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(row[k]) if isinstance(row.get(k),(list,dict)) else row.get(k) for k in keys})

def load_scale(scale):
    manifest=json.loads((BASE/f'formal{scale}_manifest.json').read_text())
    panel=[]
    for ent in manifest['instances']:
        raw=(BASE/ent['snapshot']).read_bytes(); env=pickle.loads(raw)
        panel.append((ent,env,hashlib.sha256(raw).hexdigest()))
    return manifest,panel

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--sample',action='store_true'); a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    manifest_hashes={}
    for scale in (10,20):
        manifest=json.loads((BASE/f'formal{scale}_manifest.json').read_text()); manifest_hashes[str(scale)]=hashlib.sha256((BASE/f'formal{scale}_manifest.json').read_bytes()).hexdigest()
    all_vis=[]; all_ev=[]; all_trip=[]; all_field=[]; all_replay=[]; features=[]; field_features=[]; errors=[]; timings=[]
    t_all=time.time()
    for scale in (10,20):
        manifest,panel=load_scale(scale)
        for idx,(ent,env,sha) in enumerate(panel):
            if a.sample and idx!=0: continue
            if sha!=ent['sha256']: raise ValueError(f'snapshot hash mismatch {scale} {ent["id"]}')
            feats=initial_features(env)
            features.append({'instance_id':ent['id'],'scale':scale,'snapshot_sha256':sha,**{k:v for k,v in feats.items() if k!='fields'}})
            for ff in feats['fields']: field_features.append({'instance_id':ent['id'],'scale':scale,**ff})
            for method in METHODS:
                rf=BASE/f'formal{scale}_run'/f'{idx:04d}_{method}.json'
                if not rf.exists(): raise FileNotFoundError(rf)
                d=json.loads(rf.read_text()); identity=d['identity']; result=d['result']
                if identity['id']!=ent['id'] or identity['method']!=method: raise ValueError(f'result identity mismatch {rf}')
                start=time.time()
                try:
                    led=replay_result(env,result,method,{'id':ent['id'],'fields':scale})
                except Exception as exc:
                    errors.append({'instance_id':ent['id'],'scale':scale,'method':method,'result':str(rf),'error':repr(exc),'traceback':traceback.format_exc()})
                    continue
                elapsed=time.time()-start; timings.append({'instance_id':ent['id'],'scale':scale,'method':method,'seconds':elapsed})
                all_vis.extend(led['visits']); all_ev.extend(led['events']); all_trip.extend(led['trips']); all_field.extend(led['fields']); all_replay.append(led['replay'])
    write_csv(out/'visits.csv',all_vis); write_csv(out/'events.csv',all_ev); write_csv(out/'trips.csv',all_trip); write_csv(out/'fields.csv',all_field); write_csv(out/'replay.csv',all_replay); write_csv(out/'features.csv',features); write_csv(out/'field_features.csv',field_features)
    (out/'errors.json').write_text(json.dumps(errors,indent=2))
    (out/'timings.json').write_text(json.dumps({'sample':a.sample,'replays':len(timings),'total_seconds':time.time()-t_all,'per_replay':timings},indent=2))
    (out/'source_manifest.json').write_text(json.dumps({'formal10_manifest_sha256':manifest_hashes['10'],'formal20_manifest_sha256':manifest_hashes['20'],'code_sha':None},indent=2))
    print(json.dumps({'sample':a.sample,'visits':len(all_vis),'events':len(all_ev),'trips':len(all_trip),'replays':len(all_replay),'errors':len(errors),'seconds':sum(x['seconds'] for x in timings)},indent=2))
    if errors:
        print(json.dumps(errors[:3],indent=2)); raise SystemExit(1)

if __name__=='__main__': main()
