#!/usr/bin/env python3
"""Crossed external balanced-minus-hash effects from released score files."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; CROSS=ROOT/'data'/'crosscorpus'
FAMILIES={
 'asvspoof5_wavlm': dict(corpus='asvspoof5',backend='wavlm_frozen_backend',folds=[f'A{i}' for i in range(17,26)],reference='A17',nseed=5,offset=0),
 'mlaad_xlsr': dict(corpus='mlaad',backend='xlsr_peft_adapter',folds=['FASTPITCH','GLOWTTS','JENNY','NEURALHMM','OVERFLOW','SPEEDYSPEECH','TACOTRON2','TORTOISE','VITS'],reference='JENNY',nseed=15,offset=1000),
}

def read(path):
 out={}
 for line in path.open():
  r=json.loads(line); out[str(r.get('utterance_id',''))]=(1 if r.get('label')=='spoof' else 0,float(r['score']))
 return out

def kernel(p,n): return (p[:,None]>n[None,:]).astype(float)+.5*(p[:,None]==n[None,:])
def old_auc(y,s):
 o=np.argsort(-s); y=y[o]; p=int(y.sum()); n=len(y)-p
 tp=np.cumsum(y==1); fp=np.cumsum(y==0)
 return float(np.trapezoid(np.r_[0,tp/p,1],np.r_[0,fp/n,1]))

def seeds_for(corpus,backend,folds):
 sets=[]
 for block in ['samp_hash','samp_source-balanced']:
  base=CROSS/corpus/block/backend
  for fold in folds:
   paths=list(base.glob(f'seed_*/{fold}.jsonl'))
   if paths: sets.append({int(p.parent.name.removeprefix('seed_')) for p in paths})
 return sorted(set.intersection(*sets))

def load_fold(corpus,backend,fold,seeds):
 maps={}; blocks=['samp_hash','samp_source-balanced']; missing=[]
 for ci,block in enumerate(blocks):
  for si,seed in enumerate(seeds):
   path=CROSS/corpus/block/backend/f'seed_{seed}'/f'{fold}.jsonl'
   if not path.exists(): missing.append(str(path.relative_to(ROOT)))
   else: maps[ci,si]=read(path)
 if missing: return None,missing
 sets=[set(x) for x in maps.values()]; common=set.intersection(*sets); union=set.union(*sets)
 if common!=union: raise RuntimeError(f'{corpus}/{fold}: ID mismatch intersection={len(common)} union={len(union)}')
 ref=maps[0,0]
 for key,m in maps.items():
  if any(m[u][0]!=ref[u][0] for u in common): raise RuntimeError(f'{corpus}/{fold}/{key}: label mismatch')
 pos=sorted(u for u in common if ref[u][0]); neg=sorted(u for u in common if not ref[u][0])
 K=np.empty((2,len(seeds),len(pos),len(neg))); ties=[]
 for ci,block in enumerate(blocks):
  for si,seed in enumerate(seeds):
   m=maps[ci,si]; ps=np.array([m[u][1] for u in pos]); ns=np.array([m[u][1] for u in neg]); k=kernel(ps,ns); K[ci,si]=k
   y=np.r_[np.ones(len(pos),int),np.zeros(len(neg),int)]; s=np.r_[ps,ns]; old=old_auc(y,s); new=float(k.mean())
   if not np.isclose(old,new,rtol=0,atol=1e-15): ties.append(dict(block=block,seed=seed,legacy=old,tie_correct=new,difference=new-old))
 return dict(kernels=K,contrast=K[1]-K[0],meta=dict(seeds=seeds,n_cells=len(maps),union_ids=len(union),common_ids=len(common),ids_exactly_aligned=True,spoof_items=len(pos),bonafide_items=len(neg)),ties=ties),[]

def maxt(point,boot):
 se=boot.std(0,ddof=1); centered=boot-boot.mean(0); mt=np.max(np.abs(centered/se),axis=1); crit=float(np.quantile(mt,.95)); ci=np.column_stack([point-crit*se,point+crit*se]); obs=np.abs(point/se)
 padj=np.array([(1+np.sum(mt>=v))/(len(mt)+1) for v in obs]); gp=float((1+np.sum(mt>=obs.max()))/(len(mt)+1))
 return se,crit,ci,padj,gp

def run(name,cfg,reps,batch,base_seed):
 folds=cfg['folds']; seeds=seeds_for(cfg['corpus'],cfg['backend'],folds)
 if len(seeds)!=cfg['nseed']: raise RuntimeError(f'{name}: expected {cfg["nseed"]} seeds, got {seeds}')
 loaded={}; missing={}; tie_changes=[]
 for f in folds:
  x,miss=load_fold(cfg['corpus'],cfg['backend'],f,seeds)
  if x is None: missing[f]=dict(status='not_estimable',reason='required sampler cells absent',is_reference_fold=f==cfg['reference'],missing_files=miss)
  else: loaded[f]=x; tie_changes.extend([dict(fold=f,**z) for z in x['ties']])
 estimable=[f for f in folds if f in loaded]
 H=np.array([loaded[f]['kernels'][0].mean() for f in estimable]); B=np.array([loaded[f]['kernels'][1].mean() for f in estimable]); D=B-H
 boot=np.empty((reps,len(estimable))); rng=np.random.default_rng(base_seed+cfg['offset'])
 for start in range(0,reps,batch):
  stop=min(start+batch,reps); size=stop-start
  sw=rng.multinomial(len(seeds),np.full(len(seeds),1/len(seeds)),size=size)/len(seeds)
  for j,f in enumerate(estimable):
   C=loaded[f]['contrast']; np_,nn=C.shape[1:]
   pw=rng.multinomial(np_,np.full(np_,1/np_),size=size)/np_; nw=rng.multinomial(nn,np.full(nn,1/nn),size=size)/nn
   sa=np.einsum('bs,spn->bpn',sw,C,optimize=True); pa=np.einsum('bp,bpn->bn',pw,sa,optimize=True); boot[start:stop,j]=np.einsum('bn,bn->b',pa,nw)
 ordinary=np.quantile(boot,[.025,.975],axis=0); q=.05/(2*9); bonf=np.quantile(boot,[q,1-q],axis=0); se,crit,mci,padj,gp=maxt(D,boot)
 rows={}
 for j,f in enumerate(estimable):
  rows[f]=dict(status='estimable',is_reference_fold=f==cfg['reference'],**loaded[f]['meta'],hash_auc=float(H[j]),balanced_auc=float(B[j]),balanced_minus_hash=float(D[j]),bootstrap_se=float(se[j]),ordinary_percentile_ci95=ordinary[:,j].tolist(),ordinary_excludes_zero=bool(ordinary[0,j]>0 or ordinary[1,j]<0),bonferroni_m9_percentile_ci95=bonf[:,j].tolist(),bonferroni_m9_excludes_zero=bool(bonf[0,j]>0 or bonf[1,j]<0),max_t_ci95=mci[j].tolist(),max_t_excludes_zero=bool(mci[j,0]>0 or mci[j,1]<0),max_t_adjusted_p=float(padj[j]),bootstrap_probability_positive=float(np.mean(boot[:,j]>0)+.5*np.mean(boot[:,j]==0)))
 rows.update(missing)
 return dict(corpus=cfg['corpus'],backend=cfg['backend'],reference_fold=cfg['reference'],planned_folds=folds,estimable_folds=estimable,paired_seeds=seeds,planned_bonferroni_family_size=9,max_t_estimable_family_size=len(estimable),max_t_critical=crit,global_max_t_p_any_effect_nonzero=gp,tie_audit=dict(files_checked=2*len(seeds)*len(estimable),files_changed_by_tie_correction=len(tie_changes),changes=tie_changes),folds={f:rows[f] for f in folds})

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=200000); ap.add_argument('--batch',type=int,default=2000); ap.add_argument('--seed',type=int,default=20260712); ap.add_argument('--output',type=Path,default=ROOT/'outputs'/'external_crossed_effects_200k.json'); a=ap.parse_args()
 result=dict(method=dict(bootstrap_replicates=a.reps,batch_size=a.batch,rng_seed=a.seed,auroc='tie-correct Mann-Whitney; 0.5 credit for ties',seed_draw='shared across folds and arms within family',item_draw='class-stratified within fold; shared across seeds and arms',ordinary_ci='percentile crossed bootstrap',bonferroni_ci='percentile crossed bootstrap alpha=.05/9',max_t_ci='single-step centered bootstrap max-t; bootstrap-SD studentizer'),families={name:run(name,cfg,a.reps,a.batch,a.seed) for name,cfg in FAMILIES.items()})
 a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print('wrote',a.output)
 for name,fam in result['families'].items():
  print(f'\n{name}: max-t critical={fam["max_t_critical"]:.6f}, global p={fam["global_max_t_p_any_effect_nonzero"]:.6g}')
  print('fold          hash      balanced  delta     ordinary CI          Bonf m=9 CI          max-T CI')
  for fold,row in fam['folds'].items():
   if row['status']!='estimable': print(f'{fold:13s} NOT ESTIMABLE ({row["reason"]})'); continue
   o,b,m=row['ordinary_percentile_ci95'],row['bonferroni_m9_percentile_ci95'],row['max_t_ci95']
   print(f'{fold:13s} {row["hash_auc"]:.6f}  {row["balanced_auc"]:.6f}  {row["balanced_minus_hash"]:+.6f}  [{o[0]:+.6f},{o[1]:+.6f}]  [{b[0]:+.6f},{b[1]:+.6f}]  [{m[0]:+.6f},{m[1]:+.6f}]')
 return 0
if __name__=='__main__': raise SystemExit(main())
