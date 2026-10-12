"""Patient-level DeLong, AP, calibration, classification and exploratory DCA."""
import numpy as np
from scipy.special import logit
from scipy.stats import norm
from sklearn.metrics import average_precision_score

def validate_binary(y, p=None):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Outcome must be a nonempty one-dimensional 0/1 array")
    if p is not None:
        p = np.asarray(p, dtype=float)
        if p.shape[-1] != len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilities must be finite, in [0,1], and align with outcomes")
    return y.astype(int)

def delong(y, probabilities):
    """Placement-value covariance, handling ties as 0.5. Models are rows."""
    ps = np.atleast_2d(probabilities).astype(float)
    y = validate_binary(y, ps)
    pos, neg = ps[:, y == 1], ps[:, y == 0]
    if min(pos.shape[1], neg.shape[1]) < 2:
        raise ValueError("DeLong covariance requires at least two observations per outcome class")
    v = (pos[:, :, None] > neg[:, None, :]).astype(float) + .5 * (pos[:, :, None] == neg[:, None, :])
    covariance = np.atleast_2d(np.cov(v.mean(axis=2))) / pos.shape[1] + np.atleast_2d(np.cov(v.mean(axis=1))) / neg.shape[1]
    return v.mean(axis=(1, 2)), covariance

def paired_delong(y, first, second):
    auc, cov = delong(y, [first, second])
    delta = float(auc[0] - auc[1]); variance = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    p = float(2 * norm.sf(abs(delta) / np.sqrt(variance))) if variance > 1e-15 else (1.0 if abs(delta) < 1e-15 else None)
    return {"difference": delta, "P": p, "variance": variance}

def wilson(k, n):
    if n == 0:
        return {"count": int(k), "denominator": 0, "estimate": None, "low": None, "high": None}
    if not 0 <= k <= n:
        raise ValueError("Invalid binomial numerator/denominator")
    z = norm.ppf(.975); value = k / n; denominator = 1 + z*z/n
    center = (value + z*z/(2*n))/denominator
    half = z*np.sqrt(value*(1-value)/n + z*z/(4*n*n))/denominator
    return {"count": int(k), "denominator": int(n), "estimate": value, "low": max(0., center-half), "high": min(1., center+half)}

def classification(y, p, threshold):
    y = validate_binary(y, p)
    if not 0 < threshold < 1:
        raise ValueError("Threshold must lie in (0,1)")
    z = np.asarray(p) >= threshold
    tp = int(((y == 1) & z).sum()); fn = int(((y == 1) & ~z).sum())
    fp = int(((y == 0) & z).sum()); tn = int(((y == 0) & ~z).sum())
    return {"TP":tp,"FN":fn,"FP":fp,"TN":tn,
            **{name:wilson(k,n) for name,k,n in [("Sensitivity",tp,tp+fn),("Specificity",tn,tn+fp),("PPV",tp,tp+fp),("NPV",tn,tn+fn),("Accuracy",tp+tn,len(y)),("ResidualRisk",fn,fn+tn)]}}

def performance(y, p, threshold=None, bootstrap=1000, seed=978):
    y = validate_binary(y, p); p = np.asarray(p)
    auc, cov = delong(y, p); se = np.sqrt(cov[0,0]); z = norm.ppf(.975)
    rng = np.random.RandomState(seed); aps = []
    for _ in range(bootstrap):
        ix = rng.choice(len(y),len(y),True)
        if np.unique(y[ix]).size == 2:
            aps.append(average_precision_score(y[ix],p[ix]))
    ap_ci = np.percentile(aps,[2.5,97.5]).tolist() if aps else [None,None]
    result = {"N":len(y),"events":int(y.sum()),"AUC":float(auc[0]),"AUC_low":max(0.,float(auc[0]-z*se)),"AUC_high":min(1.,float(auc[0]+z*se)),"AP":float(average_precision_score(y,p)),"AP_low":ap_ci[0],"AP_high":ap_ci[1],"Brier":float(np.mean((p-y)**2)),"valid_AP_bootstraps":len(aps)}
    if threshold is not None:result.update(classification(y,p,threshold))
    return result

def net_benefit(y, p, thresholds):
    y = validate_binary(y,p); p = np.atleast_2d(p); ts = np.asarray(thresholds)
    if ts.ndim != 1 or np.any((ts<=0)|(ts>=1)):
        raise ValueError("DCA thresholds must be in (0,1)")
    odds = ts/(1-ts)
    contributions = (p[:,:,None]>=ts)*(y[None,:,None]-(1-y[None,:,None])*odds)
    return contributions.mean(axis=1), contributions

def paired_dca(y, first, second, thresholds, bootstrap=1000, seed=978):
    nb, contributions = net_benefit(y,[first,second],thresholds)
    delta = contributions[0]-contributions[1];n=len(y)
    weights = np.random.RandomState(seed).multinomial(n,np.full(n,1/n),size=bootstrap)/n
    low,high = np.percentile(weights@delta,[2.5,97.5],axis=0)
    return {"threshold":np.asarray(thresholds),"first":nb[0],"second":nb[1],"difference":delta.mean(axis=0),"low":low,"high":high}

def paired_average_precision(y,first,second,bootstrap=1000,seed=978):
    y=validate_binary(y,[first,second]);first,second=np.asarray(first),np.asarray(second)
    rng=np.random.RandomState(seed);values=[]
    for _ in range(bootstrap):
        ix=rng.choice(len(y),len(y),True)
        if np.unique(y[ix]).size==2:values.append(average_precision_score(y[ix],first[ix])-average_precision_score(y[ix],second[ix]))
    low,high=np.percentile(values,[2.5,97.5])
    return {"difference":float(average_precision_score(y,first)-average_precision_score(y,second)),"low":float(low),"high":float(high)}

def calibration(y,p):
    import statsmodels.api as sm
    y = validate_binary(y,p); lp=logit(np.clip(p,1e-12,1-1e-12))
    joint=sm.GLM(y,sm.add_constant(lp),family=sm.families.Binomial()).fit()
    citl=sm.GLM(y,np.ones((len(y),1)),offset=lp,family=sm.families.Binomial()).fit()
    ci=joint.conf_int();ci2=citl.conf_int()
    return {"joint_intercept":float(joint.params[0]),"joint_intercept_low":float(ci[0,0]),"joint_intercept_high":float(ci[0,1]),"slope":float(joint.params[1]),"slope_low":float(ci[1,0]),"slope_high":float(ci[1,1]),"CITL":float(citl.params[0]),"CITL_low":float(ci2[0,0]),"CITL_high":float(ci2[0,1])}

def calibration_curve(y,p,bootstrap=1000,seed=978):
    from sklearn.preprocessing import SplineTransformer
    from sklearn.linear_model import LogisticRegression
    y=validate_binary(y,p);p=np.asarray(p);lp=logit(np.clip(p,1e-6,1-1e-6))
    trans=SplineTransformer(n_knots=3,degree=3,knots="quantile",include_bias=False)
    x=trans.fit_transform(lp[:,None]);grid=np.linspace(p.min(),p.max(),200)
    xx=trans.transform(logit(np.clip(grid,1e-6,1-1e-6))[:,None])
    def fit(ix):return LogisticRegression(C=1e6,solver="lbfgs",max_iter=10000).fit(x[ix],y[ix]).predict_proba(xx)[:,1]
    estimated=fit(np.arange(len(y)));rng=np.random.RandomState(seed);bp=[]
    for _ in range(bootstrap):
        ix=rng.choice(len(y),len(y),True)
        if np.unique(y[ix]).size==2:bp.append(fit(ix))
    low,high=np.percentile(bp,[2.5,97.5],axis=0)
    return {"probability":grid,"estimated":estimated,"low":low,"high":high,"valid_bootstraps":len(bp)}
