"""Exploratory preoperative subgroup; only pooled internal/external observations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .analysis import read_table
from .predict import load_specifications,predict
from .statistics import delong,classification,net_benefit,paired_delong,performance

EARLY_FIELDS=["MR_reported_LNM","Preop_cervical_stromal_invasion","Preop_adnexal_involvement","Preop_uterine_serosal_involvement","Preop_extrauterine_local_extension","Preop_peritoneal_metastasis","Preop_distant_metastasis"]
MODELS=["Clinical","Radiomics","DL","Clinical+Radiomics","Clinical+DL","Nomogram"]

def early_status(d):
    missing=set(EARLY_FIELDS)-set(d.columns)
    if missing:raise ValueError("Missing preoperative fields: "+", ".join(sorted(missing)))
    if not d[EARLY_FIELDS].stack().isin([0,1,2]).all():raise ValueError("Preoperative indicators must use 0/1/2 or missing")
    positive=d[EARLY_FIELDS].eq(1).any(axis=1);unknown=d[EARLY_FIELDS].eq(2).any(axis=1)|d[EARLY_FIELDS].isna().any(axis=1)
    return pd.Series(np.where(positive,"excluded",np.where(unknown,"unclassifiable","eligible")),index=d.index)

def pooled_early(internal,external,output,bootstrap=20000,seed=978):
    output=Path(output)
    if output.exists():raise FileExistsError("Use a new output directory")
    specs=load_specifications();selected=[];flow=[]
    for co,path in [("internal",internal),("external",external)]:
        d=read_table(path);status=early_status(d);flow.append({"cohort":co,"N":len(d),**status.value_counts().to_dict()})
        d=d.loc[status.eq("eligible")].copy();d["cohort_key"]=co
        for m in MODELS:d[m+"_probability"]=[predict(row,specs[m]) for row in d.to_dict("records")]
        selected.append(d)
    d=pd.concat(selected,ignore_index=True);y=d.label.to_numpy(int);n=len(y);events=int(y.sum())
    if min(events,n-events)<2:raise ValueError("Too few events/non-events for this subgroup evaluation")
    p=d[[m+"_probability" for m in MODELS]].to_numpy().T;av,cv=delong(y,p)
    counts=np.zeros((bootstrap,n),dtype=np.int16);rng=np.random.default_rng(seed)
    for _,ix in d.groupby(["cohort_key","label"],sort=False).indices.items():counts[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=bootstrap)
    weights=np.zeros((bootstrap,n));rng=np.random.default_rng(seed+100)
    for _,ix in d.groupby("cohort_key",sort=False).indices.items():weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=bootstrap)/n
    pos,neg=p[:,y==1],p[:,y==0];kernel=(pos[:,:,None]>neg[:,None,:]).astype(float)+.5*(pos[:,:,None]==neg[:,None,:])
    metrics=[];aucboots=[];apboots=[]
    for j,m in enumerate(MODELS):
        ab=np.einsum('bi,ij,bj->b',counts[:,y==1],kernel[j],counts[:,y==0],optimize=True)/(events*(n-events));aucboots.append(ab)
        order=np.argsort(-p[j],kind="stable");yc=y[order];cc=counts[:,order].astype(float);tp=np.cumsum(cc*yc,axis=1);tot=np.cumsum(cc,axis=1)
        ends=np.r_[np.flatnonzero(np.diff(p[j,order])!=0),n-1];tpe=tp[:,ends];tote=tot[:,ends];increment=np.diff(np.c_[np.zeros(bootstrap),tpe],axis=1)
        precision=np.divide(tpe,tote,out=np.zeros_like(tpe),where=tote>0);apb=(precision*increment).sum(axis=1)/events;apboots.append(apb)
        lo,hi=np.quantile(ab,[.025,.975]);al,ah=np.quantile(apb,[.025,.975]);bl,bh=np.quantile(weights@((p[j]-y)**2),[.025,.975])
        row={"model":m,"N":n,"events":events,"AUC":float(av[j]),"AUC_low":float(lo),"AUC_high":float(hi),"AP":float(average_precision_score(y,p[j])),"AP_low":float(al),"AP_high":float(ah),"Brier":float(np.mean((p[j]-y)**2)),"Brier_low":float(bl),"Brier_high":float(bh)}
        if specs[m].get("threshold") is not None:row.update(classification(y,p[j],specs[m]["threshold"]))
        metrics.append(row)
    ni=MODELS.index("Nomogram");pairs=[];ts=np.arange(.01,.401,.01);nb,contributions=net_benefit(y,p,ts);dca=[]
    for j,m in enumerate(MODELS):
        if m!= "Nomogram":
            lo,hi=np.quantile(aucboots[ni]-aucboots[j],[.025,.975]);al,ah=np.quantile(apboots[ni]-apboots[j],[.025,.975])
            pairs.append({"comparator":m,**paired_delong(y,p[ni],p[j]),"AUC_difference_low":float(lo),"AUC_difference_high":float(hi),"AP_difference":float(average_precision_score(y,p[ni])-average_precision_score(y,p[j])),"AP_difference_low":float(al),"AP_difference_high":float(ah)})
        if m=="Nomogram":continue
        for threshold in [.10,.20,.30]:
            q=int(round(threshold*100))-1;diff=contributions[ni,:,q]-contributions[j,:,q];lo,hi=np.quantile(weights@diff,[.025,.975]);dca.append({"comparator":m,"threshold":threshold,"NB_nomogram":float(nb[ni,q]),"NB_comparator":float(nb[j,q]),"difference":float(diff.mean()),"low":float(lo),"high":float(hi)})
    for comparator in ["TreatNone","TreatAll"]:
        for threshold in [.10,.20,.30]:
            q=int(round(threshold*100))-1
            reference=np.zeros(n) if comparator=="TreatNone" else y-(1-y)*threshold/(1-threshold)
            diff=contributions[ni,:,q]-reference;lo,hi=np.quantile(weights@diff,[.025,.975])
            dca.append({"comparator":comparator,"threshold":threshold,"NB_nomogram":float(nb[ni,q]),"NB_comparator":float(reference.mean()),"difference":float(diff.mean()),"low":float(lo),"high":float(hi)})
    output.mkdir(parents=True)
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding="utf-8")
    for name,rows in [("flow",flow),("paired_comparisons",pairs),("DCA_selected_thresholds",dca)]:pd.DataFrame(rows).to_csv(output/(name+".csv"),index=False)
    pd.DataFrame({"threshold":ts,**{m:nb[j] for j,m in enumerate(MODELS)},"TreatAll":y.mean()-(1-y.mean())*ts/(1-ts),"TreatNone":0}).to_csv(output/"pooled_DCA_curve.csv",index=False)
    d[["ID","cohort_key","label"]+[m+"_probability" for m in MODELS]].to_csv(output/"pooled_predictions.csv",index=False,float_format="%.17g")
    # Do not fit a calibration regression with five events.
    (output/"calibration_descriptive.json").write_text(json.dumps({"N":n,"events":events,"expected_events":float(p[ni].sum()),"mean_predicted_risk":float(p[ni].mean()),"observed_rate":events/n,"E_over_O":float(p[ni].sum()/events),"calibration_regression":"Not fitted: sparse events"},indent=2),encoding="utf-8")
    return metrics

def nodal_pathway_masks(d):
    needed={"Nodal_procedure","SLN_mapping_attempted","Pelvic_nodes_assessed","Para_aortic_nodes_assessed","SLN_initial_HE_status","Bilateral_SLN_mapping","Ultrastaging_performed"}
    if not needed.issubset(d):raise ValueError("Missing nodal pathway fields")
    return {"LND_pelvic_paraaortic":d.Nodal_procedure.eq("Lymphadenectomy alone")&d.SLN_mapping_attempted.eq("No")&d.Pelvic_nodes_assessed.eq("Yes")&d.Para_aortic_nodes_assessed.eq("Yes")&d.SLN_initial_HE_status.eq("Not applicable"),
            "bilateral_SLNB":d.Nodal_procedure.eq("SLNB alone")&d.Bilateral_SLN_mapping.eq("Yes")&d.Pelvic_nodes_assessed.eq("Yes")&(d.SLN_initial_HE_status.eq("Positive")|(d.SLN_initial_HE_status.eq("Negative")&d.Ultrastaging_performed.eq("Yes")))}
