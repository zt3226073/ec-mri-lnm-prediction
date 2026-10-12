"""Reference train-partition transforms. Not the original historical training program."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from .imaging import area_weighted_mean,dl_patient_logit

def standardized_smote(training_features,training_labels,heldout_features,seed=42):
    """Fit standardization to original training observations, then oversample training only."""
    from imblearn.over_sampling import BorderlineSMOTE
    x=np.asarray(training_features,dtype=float);y=np.asarray(training_labels)
    held=np.asarray(heldout_features,dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(held).all():raise ValueError("Finite feature matrices required")
    scale=StandardScaler().fit(x);z=scale.transform(x);zheld=scale.transform(held)
    sampler=BorderlineSMOTE(kind="borderline-1",sampling_strategy=.5,k_neighbors=5,m_neighbors=10,random_state=seed)
    resampled,labels=sampler.fit_resample(z,y)
    return {"scaler":scale,"original_training":z,"resampled_training":resampled,"resampled_labels":labels,"heldout":zheld}

def pcc_keep(standardized_training,original_training,original_labels,cutoff=.9):
    """Remove correlated features using outcome association in original training observations."""
    x=np.asarray(standardized_training,float);original=np.asarray(original_training,float);y=np.asarray(original_labels,float)
    if original.shape[0]!=len(y) or x.shape[1]!=original.shape[1]:raise ValueError("Feature/outcome dimensions differ")
    nonconstant=np.std(x,axis=0)>0;keep=list(np.flatnonzero(nonconstant))
    if len(keep)<2:return np.asarray(keep,int)
    corr=np.nan_to_num(np.corrcoef(x[:,keep],rowvar=False));association=np.array([abs(np.corrcoef(original[:,j],y)[0,1]) if np.std(original[:,j])>0 else 0 for j in keep])
    active=list(range(len(keep)))
    while True:
        pairs=[(i,j) for n,i in enumerate(active) for j in active[n+1:] if abs(corr[i,j])>cutoff]
        if not pairs:break
        i,j=pairs[0];remove=j if association[i]>=association[j] else i;active.remove(remove)
    return np.asarray([keep[i] for i in active],int)

def selected_classifier_probability(features,parameters):
    """Explicit selected-input interface in Table S3; caller must supply the verified column order and units."""
    from scipy.special import expit
    x=np.asarray(features,float);mean=np.asarray(parameters["scaler_mean"]);scale=np.asarray(parameters["scaler_scale"])
    if x.shape[-1]!=len(mean) or not np.isfinite(x).all() or np.any(scale<=0):raise ValueError("Selected-feature interface mismatch")
    return expit(parameters["standardized_intercept"]+((x-mean)/scale)@np.asarray(parameters["standardized_coefficients"]))

def patient_signature_standardization(training_patient_scores,heldout_patient_scores):
    """Exactly one unstandardized score per original training patient; no synthetic observations."""
    scale=StandardScaler().fit(np.asarray(training_patient_scores,float).reshape(-1,1))
    return scale,scale.transform(np.asarray(heldout_patient_scores,float).reshape(-1,1)).ravel()
