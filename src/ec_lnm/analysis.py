"""Evaluation of frozen patient-level models; optional train-only final fitting."""
import itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from .predict import load_specifications,predict
from .statistics import performance,paired_delong,paired_average_precision,net_benefit,paired_dca,calibration,calibration_curve

CLINICAL=["HE4","MR_reported_DMI","MR_reported_diameter","MR_reported_LNM"]
COMBINED={"Clinical":CLINICAL,"Clinical+Radiomics":CLINICAL+["radiomics"],"Clinical+DL":CLINICAL+["DL"],"Nomogram":CLINICAL+["radiomics","DL"]}

def read_table(path):
    path=Path(path);d=pd.read_excel(path) if path.suffix.lower()==".xlsx" else pd.read_csv(path)
    if "ID" not in d or d.ID.isna().any() or not d.ID.is_unique:raise ValueError("Unique nonmissing patient ID required")
    if "label" not in d or not d.label.isin([0,1]).all():raise ValueError("Binary label column required")
    if "Pathological_LNM" in d and not d.label.eq(d.Pathological_LNM).all():raise ValueError("label and Pathological_LNM disagree")
    return d

def fit_training(train,specifications=None):
    import statsmodels.api as sm
    specs=json.loads(json.dumps(specifications or load_specifications()))
    for name,cols in COMBINED.items():
        x=train[cols].astype(float)
        if not np.isfinite(x).all().all():raise ValueError("Missing/nonfinite training input")
        for row in train.to_dict("records"):predict(row,specs[name])
        fit=sm.GLM(train.label,sm.add_constant(x,has_constant="add"),family=sm.families.Binomial()).fit()
        if not fit.converged:raise RuntimeError("Logistic regression failed to converge")
        specs[name].update(intercept=float(fit.params["const"]),coefficients=[float(fit.params[k]) for k in cols],AIC=float(fit.aic))
    return specs

def evaluate(cohorts,output,refit=False,bootstrap=1000,curve_bands=False):
    output=Path(output)
    if output.exists():raise FileExistsError("Use a new output directory to preserve previous analyses")
    data={name:read_table(p) for name,p in cohorts.items()};specs=fit_training(data["train"]) if refit else load_specifications()
    output.mkdir(parents=True);(output/"model_specifications.json").write_text(json.dumps(specs,indent=2),encoding="utf-8")
    metrics=[];pairs=[];ap_pairs=[];cal=[];dca=[];subgroups=[];tables={};ts=np.arange(.01,.801,.01)
    for co,d in data.items():
        out=d[["ID","label"]].copy(); y=d.label.to_numpy()
        for name,spec in specs.items():
            out[name+"_probability"]=[predict(row,spec) for row in d.to_dict("records")]
            if spec.get("threshold") is not None:out[name+"_prediction"]=(out[name+"_probability"]>=spec["threshold"]).astype(int)
            row={"cohort":co,"model":name,**performance(y,out[name+"_probability"].to_numpy(),spec.get("threshold"),bootstrap)}
            # Keep Wilson counts and intervals as explicit JSON, avoiding ambiguous string-only rates.
            metrics.append(row)
        out.to_csv(output/f"{co}_predictions.csv",index=False,float_format="%.17g");tables[co]=out
        for first,second in itertools.combinations(specs,2):
            pairs.append({"cohort":co,"model1":first,"model2":second,**paired_delong(y,out[first+"_probability"],out[second+"_probability"])})
            ap_pairs.append({"cohort":co,"model1":first,"model2":second,**paired_average_precision(y,out[first+"_probability"],out[second+"_probability"],bootstrap)})
        pn=out.Nomogram_probability.to_numpy();cal.append({"cohort":co,**calibration(y,pn)})
        if co!="train":
            from .subgroups import nodal_pathway_masks
            masks={"MRI_negative":d.MR_reported_LNM.eq(0)}
            if "Nodal_procedure" in d:masks.update(nodal_pathway_masks(d))
            for name,mask in masks.items():
                yy=y[mask];pp=pn[mask]
                if min(sum(yy),len(yy)-sum(yy))<2:continue
                row={"cohort":co,"subgroup":name,**performance(yy,pp,.161,bootstrap)}
                row["paired_vs_Clinical_Radiomics"]=paired_delong(yy,pp,out.loc[mask,"Clinical+Radiomics_probability"].to_numpy());subgroups.append(row)
        nb,_=net_benefit(y,out[[m+"_probability" for m in specs]].to_numpy().T,ts)
        pd.DataFrame({"threshold":ts,**{m:nb[j] for j,m in enumerate(specs)},"All":y.mean()-(1-y.mean())*ts/(1-ts),"None":0}).to_csv(output/f"{co}_dca.csv",index=False)
        for m in specs:
            if m=="Nomogram":continue
            z=paired_dca(y,pn,out[m+"_probability"],ts,bootstrap)
            dca.extend({"cohort":co,"comparator":m,**{k:float(v[i]) for k,v in z.items()}} for i in range(len(ts)))
        if curve_bands:pd.DataFrame(calibration_curve(y,pn,bootstrap)).to_csv(output/f"{co}_calibration_curve.csv",index=False)
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding="utf-8")
    (output/"restricted_subgroups.json").write_text(json.dumps(subgroups,indent=2,allow_nan=False),encoding="utf-8")
    for name,rows in [("paired_delong",pairs),("paired_AP",ap_pairs),("calibration",cal),("paired_dca",dca)]:pd.DataFrame(rows).to_csv(output/(name+".csv"),index=False)
    return tables
