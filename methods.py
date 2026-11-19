"""Shared fitting and calibration for established split-conformal constructions.

Every fit, residual, tuning choice and numerical floor uses proper-training
observations only. Query covariates are read only for prediction.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor
from forests import fit_forests
from qr import fit_path,predict_qr

K_GRID=(10,20,40,80,120)
LEAF_GRID=(5,10,20,40)
LAMBDA_GRID=(0.,.01,.1,1.)
METHODS=('Split','MA','RMS','RCP','CQR')
INNER_REPETITIONS=5

def arrays_hash(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a)
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def feature_x(x):
    z=2*np.asarray(x)-1
    return np.column_stack((z,z*z))

def point_fit(x,y,kind,random_state):
    if kind=='ols':return ('ols',np.linalg.lstsq(np.column_stack((np.ones(len(x)),feature_x(x))),y,rcond=None)[0])
    if kind=='rf':
        model=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,max_features=1.,bootstrap=True,n_jobs=1,random_state=random_state)
        return ('rf',model.fit(x,y))
    raise ValueError('point_model must be ols or rf')

def point_predict(fit,x):
    return np.column_stack((np.ones(len(x)),feature_x(x)))@fit[1] if fit[0]=='ols' else fit[1].predict(x)

def partition_groups(n,seed,groups=None,fractions=None,n_folds=5):
    """Randomly assign entire groups; fractions apply to group counts."""
    groups=np.arange(n) if groups is None else np.asarray(groups)
    if groups.shape!=(n,):raise ValueError('groups must have one value per row')
    unique,inverse=np.unique(groups,return_inverse=True)
    permutation=np.random.default_rng(seed).permutation(len(unique))
    if fractions is None:
        chunks=np.array_split(permutation,n_folds)
    else:
        cumulative=np.cumsum(fractions)
        if not np.isclose(cumulative[-1],1):raise ValueError('fractions must sum to one')
        cuts=(cumulative[:-1]*len(unique)).astype(int)
        chunks=np.split(permutation,cuts)
    parts=[np.flatnonzero(np.isin(inverse,c)) for c in chunks]
    if any(len(p)==0 for p in parts):raise ValueError('empty group partition')
    assert sum(map(len,parts))==n
    return parts

def crossfit_training(x,y,seed,point_model='ols',groups=None):
    part_seed,*model_seeds=seed.spawn(6)
    folds=partition_groups(len(y),part_seed,groups=groups)
    residuals=np.empty(len(y));fold_id=np.empty(len(y),int);records=[]
    for j,(held,mseed) in enumerate(zip(folds,model_seeds)):
        mask=np.ones(len(y),bool);mask[held]=False
        fit=point_fit(x[mask],y[mask],point_model,int(mseed.generate_state(1)[0]))
        residuals[held]=y[held]-point_predict(fit,x[held]);fold_id[held]=j
        if groups is not None:assert not set(np.asarray(groups)[held])&set(np.asarray(groups)[mask])
        records.append({'fold':j,'heldout_size':len(held),'heldout_indices':held.tolist(),'model_spawn_key':list(mseed.spawn_key)})
    return residuals,{'partition_spawn_key':list(part_seed.spawn_key),'folds':records,
                      'fold_id_sha256':arrays_hash(fold_id),'residual_sha256':arrays_hash(residuals)},fold_id

def local_scales(x,residuals,query,k_values=K_GRID):
    if max(k_values)>len(x):raise ValueError('k exceeds proper-training residual pool')
    center,spread=x.mean(axis=0),x.std(axis=0,ddof=0)
    spread=np.where(spread>0,spread,1.)
    distances=cdist((query-center)/spread,(x-center)/spread,metric='sqeuclidean')
    # Stable full sorting fixes nearest-neighbor ties by proper-training row order.
    nearest=np.argsort(distances,axis=1,kind='stable')[:,:max(k_values)]
    magnitudes=np.abs(residuals)[nearest]
    floor=max(.01*float(np.median(np.abs(residuals))),np.finfo(float).tiny)
    sums,squares=np.cumsum(magnitudes,axis=1),np.cumsum(magnitudes*magnitudes,axis=1)
    values={}
    for k in k_values:
        for method,raw in [('MA',sums[:,k-1]/k),('RMS',np.sqrt(squares[:,k-1]/k))]:
            values[method,'knn',k]={'scale':np.maximum(raw,floor),'floor_active':raw<floor}
    return values,floor

def fit_candidates(x,y,seed,query,alpha=.1,point_model='ols',groups=None,include_linear=True,
                   k_values=K_GRID,leaf_grid=LEAF_GRID,lambdas=LAMBDA_GRID):
    x,y,query=np.asarray(x,float),np.asarray(y,float),np.asarray(query,float)
    cf_seed,point_seed,forest_seed=seed.spawn(3)
    residuals,signature,fold_id=crossfit_training(x,y,cf_seed,point_model,groups)
    mu=point_predict(point_fit(x,y,point_model,int(point_seed.generate_state(1)[0])),query)
    candidates,floor=local_scales(x,residuals,query,k_values)
    candidates['Split','fixed',0]={'scale':np.ones(len(query))}
    forests,forest_diagnostics=fit_forests(x,y,np.abs(residuals),query,(alpha,),int(forest_seed.generate_state(1)[0]),leaf_grid=leaf_grid)
    candidates.update(forests[alpha]);solvers=[]
    if include_linear:
        models=fit_path(feature_x(x),np.abs(residuals),1-alpha,lambdas)
        lower=fit_path(feature_x(x),y,alpha/2,lambdas)
        upper=fit_path(feature_x(x),y,1-alpha/2,lambdas)
        for lam in lambdas:
            raw=predict_qr(models[lam],feature_x(query))
            candidates['RCP','quadratic',lam]={'scale':np.maximum(raw,floor),'floor_active':raw<floor}
            l,u=predict_qr(lower[lam],feature_x(query)),predict_qr(upper[lam],feature_x(query))
            candidates['CQR','quadratic',lam]={'lower':np.minimum(l,u),'upper':np.maximum(l,u),'crossed':l>u}
            for method,model in [('RCP',models[lam]),('CQR-lower',lower[lam]),('CQR-upper',upper[lam])]:
                solvers.append({'method':method,'learner':'quadratic','param':lam,**model['diagnostics']})
    signature.update({'floor':floor,'absolute_oof_residuals':np.abs(residuals).tolist(),'oof_fold_ids':fold_id.tolist(),'point_model':point_model,'point_spawn_key':list(point_seed.spawn_key),
                      'forest_spawn_key':list(forest_seed.spawn_key),'training_size':len(y)})
    return candidates,mu,signature,solvers,forest_diagnostics

def conformal_quantile(scores,alpha):
    rank=math.ceil((len(scores)+1)*(1-alpha))
    if rank>len(scores):return math.inf
    if rank<1:raise ValueError('alpha must lie strictly between zero and one')
    return float(np.partition(scores,rank-1)[rank-1])

def calibrate_predict(candidate,mu,cal_y,n_cal,alpha=.1):
    if 'scale' in candidate:
        s=candidate['scale'];q=conformal_quantile(np.abs(cal_y-mu[:n_cal])/s[:n_cal],alpha)
        lower,upper=mu[n_cal:]-q*s[n_cal:],mu[n_cal:]+q*s[n_cal:]
        active=candidate.get('floor_active',np.zeros(len(mu),bool))
        diag={'calibration_q':q,'test_floor_fraction':float(np.mean(active[n_cal:])),
              'test_crossing_fraction':0.,'test_inversion_fraction':0.}
    else:
        l,u=candidate['lower'],candidate['upper']
        q=conformal_quantile(np.maximum(l[:n_cal]-cal_y,cal_y-u[:n_cal]),alpha)
        lower,upper=l[n_cal:]-q,u[n_cal:]+q
        inverted=lower>upper
        # An empty calibrated set is enlarged to the initial midpoint singleton.
        # Signed correction is retained; it is not truncated to zero.
        midpoint=(l[n_cal:]+u[n_cal:])/2
        lower=np.where(inverted,midpoint,lower);upper=np.where(inverted,midpoint,upper)
        diag={'calibration_q':q,'test_floor_fraction':0.,
              'test_crossing_fraction':float(np.mean(candidate['crossed'][n_cal:])),
              'test_inversion_fraction':float(np.mean(inverted))}
    return lower,upper,diag

def interval_score(lower,upper,y,alpha):
    return upper-lower+2/alpha*np.maximum(lower-y,0)+2/alpha*np.maximum(y-upper,0)

def evaluate(lower,upper,y,alpha,groups=None):
    covered=(lower<=y)&(y<=upper);width=upper-lower;score=interval_score(lower,upper,y,alpha)
    out={'coverage':float(covered.mean()),'width':float(width.mean()),'interval_score':float(score.mean())}
    local=[]
    if groups is not None:
        for g in np.unique(groups):
            ix=groups==g
            local.append({'group':int(g),'n_test':int(ix.sum()),'coverage':float(covered[ix].mean()),
                          'width':float(width[ix].mean()),'interval_score':float(score[ix].mean())})
        cov=np.array([r['coverage'] for r in local])
        out.update(worst_group_undercoverage=float(np.maximum(1-alpha-cov,0).max()),
                   max_group_absolute_error=float(np.abs(cov-(1-alpha)).max()))
    return out,local

def select_parameters(x,y,seed,alpha=.1,point_model='ols',groups=None,include_linear=True,
                      k_values=K_GRID,leaf_grid=LEAF_GRID,lambdas=LAMBDA_GRID):
    raw=[];signatures=[];solvers=[];forestlogs=[]
    for repeat,inner_seed in enumerate(seed.spawn(INNER_REPETITIONS),1):
        partition_seed,fit_seed=inner_seed.spawn(2)
        fit,cal,val=partition_groups(len(y),partition_seed,groups=groups,fractions=(.6,.2,.2))
        candidates,mu,signature,solver,forest=fit_candidates(x[fit],y[fit],fit_seed,np.vstack((x[cal],x[val])),
                    alpha,point_model,None if groups is None else np.asarray(groups)[fit],include_linear,k_values,leaf_grid,lambdas)
        for (method,learner,param),candidate in candidates.items():
            lower,upper,_=calibrate_predict(candidate,mu,y[cal],len(cal),alpha)
            raw.append({'inner_repeat':repeat,'method':method,'learner':learner,'param':param,
                        'loss':float(interval_score(lower,upper,y[val],alpha).mean())})
        signatures.append({'inner_repeat':repeat,'partition_spawn_key':list(partition_seed.spawn_key),
                          'fit_indices':fit.tolist(),'cal_indices':cal.tolist(),'validation_indices':val.tolist(),
                          'crossfit':signature})
        solvers.extend({'stage':'inner','inner_repeat':repeat,**r} for r in solver)
        forestlogs.append({'stage':'inner','inner_repeat':repeat,**forest})
    averaged=[];choices={}
    for method,learner in sorted({(r['method'],r['learner']) for r in raw}):
        records=[]
        for param in sorted({r['param'] for r in raw if (r['method'],r['learner'])==(method,learner)}):
            losses=[r['loss'] for r in raw if (r['method'],r['learner'],r['param'])==(method,learner,param)]
            assert len(losses)==INNER_REPETITIONS
            record={'method':method,'learner':learner,'param':param,'loss':float(np.mean(losses))}
            records.append(record);averaged.append(record)
        choices[method,learner]=min(records,key=lambda r:(r['loss'],-r['param']))
    return choices,raw,averaged,signatures,solvers,forestlogs

def summarize_frame(frame,by,metrics):
    import pandas as pd
    records=[]
    for key,group in frame.groupby(by,sort=False):
        rec=dict(zip(by,key if isinstance(key,tuple) else (key,)));n=len(group);rec['repetitions']=n
        critical=student_t.ppf(.975,n-1) if n>1 else math.nan
        for metric in metrics:
            a=group[metric].to_numpy();mean=float(a.mean());se=float(a.std(ddof=1)/np.sqrt(n)) if n>1 else math.nan
            rec.update({metric:mean,metric+'_mcse':se,metric+'_low':mean-critical*se,metric+'_high':mean+critical*se})
        records.append(rec)
    return pd.DataFrame(records)
