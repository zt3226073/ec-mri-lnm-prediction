"""Readable ROC/PR, calibration and DCA figures from exported patient scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve

MODELS=["Clinical","Radiomics","DL","Clinical+Radiomics","Nomogram"]
COLORS=["#3b7d23","#3265a8","#009caa","#d68b12","#b5164b"]
COHORTS=["train","internal","external"]
NAMES=["Training apparent fit","Internal validation","External validation"]

def plot_results(results,output,font="Arial"):
    results=Path(results);output=Path(output)
    if output.exists():raise FileExistsError("Use a new figure output directory")
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family":font,"font.size":9,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42})
    data={co:pd.read_csv(results/f"{co}_predictions.csv") for co in COHORTS};rows=json.loads((results/"metrics.json").read_text())
    fig,axs=plt.subplots(2,3,figsize=(5.67,6.8));fig.subplots_adjust(left=.10,right=.987,top=.92,bottom=.215,wspace=.53,hspace=1.52)
    for i,co in enumerate(COHORTS):
        y=data[co].label.to_numpy()
        for j,m in enumerate(MODELS):
            p=data[co][m+"_probability"];r=next(x for x in rows if x["cohort"]==co and x["model"]==m)
            f,t,_=roc_curve(y,p);pr,re,_=precision_recall_curve(y,p)
            axs[0,i].plot(f,t,color=COLORS[j],lw=1,label=f'{m}: {r["AUC"]:.3f}');axs[1,i].step(re,pr,where="post",color=COLORS[j],lw=1,label=f'{m}: {r["AP"]:.3f}')
        axs[0,i].plot([0,1],[0,1],"--",color="#999",lw=.7);axs[1,i].axhline(y.mean(),ls="--",color="#777",lw=.8,label=f'Prevalence: {y.mean():.3f}')
        for row in [0,1]:
            ax=axs[row,i];ax.set(xlim=(0,1),ylim=(0,1.02),xlabel="False-positive rate" if row==0 else "Recall",ylabel="Sensitivity" if row==0 else "Precision")
            ax.set_title(chr(65+row*3+i)+"  "+NAMES[i].replace(" ","\n",1),loc="left",fontsize=9)
            ax.legend(loc="upper left",bbox_to_anchor=(-.27,-.29),frameon=False,borderaxespad=0,handlelength=1.15,handletextpad=.3,labelspacing=.3,fontsize=8)
    for ext in ["png","svg","pdf"]:fig.savefig(output/("Fig2."+ext),dpi=600,facecolor="white")
    plt.close(fig)

    # Calibration bands are optional expensive output of evaluate --calibration-curves.
    if not all((results/f"{co}_calibration_curve.csv").exists() for co in COHORTS):return
    fig,axs=plt.subplots(2,3,figsize=(5.67,5.95));fig.subplots_adjust(left=.11,right=.986,top=.93,bottom=.21,wspace=.58,hspace=.73)
    for i,co in enumerate(COHORTS):
        y=data[co].label.to_numpy();p=data[co].Nomogram_probability.to_numpy();b=pd.read_csv(results/f"{co}_calibration_curve.csv");ax=axs[0,i]
        ax.fill_between(b.probability,b.low,b.high,color=COLORS[-1],alpha=.12);ax.plot(b.probability,b.estimated,color=COLORS[-1],lw=1.2);ax.plot([0,1],[0,1],"--",color="#777",lw=.7)
        ax.plot(p[y==0],np.full(sum(y==0),.015),"|",color="#50718a",alpha=.5);ax.plot(p[y==1],np.full(sum(y==1),.985),"|",color=COLORS[-1],alpha=.6)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="Predicted probability",ylabel="Estimated event probability");ax.set_title(chr(65+i)+"  "+NAMES[i].replace(" ","\n",1),loc="left");ax.text(.04,.86,f'Brier = {np.mean((p-y)**2):.3f}',transform=ax.transAxes,fontsize=8)
        dc=pd.read_csv(results/f"{co}_dca.csv");ax=axs[1,i]
        for j,m in enumerate(MODELS):ax.plot(dc.threshold,dc[m],color=COLORS[j],lw=1,label=m)
        ax.plot(dc.threshold,dc.All,ls="--",color="#666",lw=.8,label="Treat all");ax.axhline(0,ls=":",color="black",lw=.8,label="Treat none")
        ax.set(xlim=(.01,.8),ylim=(-.04,.22),xlabel="Exploratory threshold\nprobability",ylabel="Net benefit");ax.set_title(chr(68+i)+"  "+NAMES[i].replace(" ","\n",1),loc="left")
    handles,labels=axs[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.52,.025),ncol=3,frameon=False,fontsize=9,handlelength=1.5,columnspacing=1)
    for ext in ["png","svg","pdf"]:fig.savefig(output/("Fig5."+ext),dpi=600,facecolor="white")
    plt.close(fig)


def plot_nomogram(output,font="Arial"):
    """Exact point rescaling of the bundled equation; native SVG text and paths."""
    from .predict import load_specifications,nomogram_points,expit
    output=Path(output)
    if output.exists():raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True)
    spec=load_specifications()["Nomogram"];lower=[0,0,0,0,-3,-4];upper=[1,1,10,1,4,4]
    dummy={k:max(lo,.1) if k=="MR_reported_diameter" else lo for k,lo in zip(spec["input_features"],lower)}
    info=nomogram_points(dummy,spec);unit=info["logit_per_point"];baseline=info["zero_point_eta"]
    maximum=sum(b*(hi-lo)/unit for b,lo,hi in zip(spec["coefficients"],lower,upper))
    plt.rcParams.update({"font.family":font,"font.size":10,"svg.fonttype":"none","pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(7,5.4));fig.subplots_adjust(left=.25,right=.97,top=.96,bottom=.04)
    labels=["HE4 abnormal","MR-DMI","MTD (cm)","MR-LNM","Radiomics signature","DL signature"]
    def axis(label,y,positions,ticklabels):
        ax.text(-8,y,label,ha="right",va="center",fontsize=10)
        ax.plot([min(positions),max(positions)],[y,y],color="#333",lw=.9)
        for x,s in zip(positions,ticklabels):
            ax.plot([x,x],[y,y-.09],color="#333",lw=.7);ax.text(x,y-.13,s,ha="center",va="top",fontsize=9)
    axis("Points",9,np.arange(0,101,20),[str(x) for x in range(0,101,20)])
    for i,(label,lo,hi,b) in enumerate(zip(labels,lower,upper,spec["coefficients"])):
        ticks=np.arange(lo,hi+1,1 if hi<=1 else 2 if i==2 else 1)
        axis(label,8-i,(ticks-lo)*b/unit,[f"{v:g}" for v in ticks])
    axis("Total points",1.7,np.arange(0,maximum+1,25),[f"{v:g}" for v in np.arange(0,maximum+1,25)])
    probabilities=np.array([.001,.01,.05,.10,.25,.5,.75,.9,.99])
    positions=(np.log(probabilities/(1-probabilities))-baseline)/unit
    axis("LNM probability",.65,positions,[f"{v:g}" for v in probabilities])
    ax.set(xlim=(-5,maximum+3),ylim=(0,9.5));ax.axis("off")
    for ext in ["png","svg","pdf"]:fig.savefig(output/("Fig4."+ext),dpi=600,facecolor="white")
    plt.close(fig)
