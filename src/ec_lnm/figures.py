"""Square ROC/PR, calibration and DCA figures from patient-level exports."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import roc_curve,precision_recall_curve

MODELS=["Clinical","Radiomics","DL","Clinical+Radiomics","Nomogram"]
COLORS=["#3b7d23","#3265a8","#009caa","#d68b12","#b5164b"]
COHORTS=["train","internal","external"]
NAMES=["Training apparent fit","Internal validation","External validation"]

def _square_axes(fig,left,bottom,size=1.60):
    width,height=fig.get_size_inches()
    ax=fig.add_axes([left/width,bottom/height,size/width,size/height]);ax.set_box_aspect(1)
    return ax

def _ticks01(ax):
    values=np.linspace(0,1,6);ax.set_xticks(values);ax.set_yticks(values)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"));ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

def _save(fig,output,name):
    for ext in ["png","svg","pdf"]:fig.savefig(output/(name+"."+ext),dpi=600,facecolor="white")
    plt.close(fig)

def plot_results(results,output,font="Arial"):
    results=Path(results);output=Path(output)
    if output.exists():raise FileExistsError("Use a new figure output directory")
    output.mkdir(parents=True)
    plt.rcParams.update({"font.family":font,"font.size":9,"axes.labelsize":9,"axes.titlesize":9,"xtick.labelsize":8,"ytick.labelsize":8,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","pdf.fonttype":42})
    data={co:pd.read_csv(results/f"{co}_predictions.csv") for co in COHORTS};rows=json.loads((results/"metrics.json").read_text())
    fig=plt.figure(figsize=(6.5,6.25))
    for row,bottom in [(0,4.19),(1,1.15)]:
        for i,co in enumerate(COHORTS):
            ax=_square_axes(fig,.54+i*2.10,bottom);y=data[co].label.to_numpy()
            for j,m in enumerate(MODELS):
                p=data[co][m+"_probability"];r=next(x for x in rows if x["cohort"]==co and x["model"]==m)
                if row==0:
                    x,z,_=roc_curve(y,p);ax.plot(x,z,color=COLORS[j],lw=1.1,label=f'{m}: {r["AUC"]:.3f}')
                else:
                    z,x,_=precision_recall_curve(y,p);ax.step(x,z,where="post",color=COLORS[j],lw=1.1,label=f'{m}: {r["AP"]:.3f}')
            if row==0:ax.plot([0,1],[0,1],"--",color="#999",lw=.7)
            else:ax.axhline(y.mean(),ls="--",color="#777",lw=.8,label=f'Prevalence: {y.mean():.3f}')
            ax.set(xlim=(0,1),ylim=(0,1),xlabel="1-Specificity" if row==0 else "Recall",ylabel="Sensitivity" if row==0 else "Precision");_ticks01(ax)
            ax.set_title(chr(65+row*3+i)+"  "+NAMES[i].replace(" ","\n",1),loc="left",fontsize=9,pad=7)
            ax.legend(loc="upper left",bbox_to_anchor=(-.27,-.23),frameon=False,borderaxespad=0,handlelength=1.10,handletextpad=.3,labelspacing=.18,fontsize=7.8)
    _save(fig,output,"Fig4")
    if not all((results/f"{co}_calibration_curve.csv").exists() for co in COHORTS):return
    fig=plt.figure(figsize=(6.5,5.40));axes=[]
    for row,bottom in [(0,3.35),(1,.95)]:
        for i,co in enumerate(COHORTS):
            ax=_square_axes(fig,.54+i*2.10,bottom);axes.append(ax);y=data[co].label.to_numpy();p=data[co].Nomogram_probability.to_numpy()
            if row==0:
                b=pd.read_csv(results/f"{co}_calibration_curve.csv");ax.fill_between(b.probability,b.low,b.high,color=COLORS[-1],alpha=.12);ax.plot(b.probability,b.estimated,color=COLORS[-1],lw=1.2);ax.plot([0,1],[0,1],"--",color="#777",lw=.7)
                ax.plot(p[y==0],np.full(sum(y==0),.015),"|",color="#50718a",alpha=.5);ax.plot(p[y==1],np.full(sum(y==1),.985),"|",color=COLORS[-1],alpha=.6)
                ax.set(xlim=(0,1),ylim=(0,1),xlabel="Predicted probability",ylabel="Estimated event probability");_ticks01(ax);ax.text(.04,.86,f'Brier = {np.mean((p-y)**2):.3f}',transform=ax.transAxes,fontsize=8)
            else:
                dc=pd.read_csv(results/f"{co}_dca.csv")
                for j,m in enumerate(MODELS):ax.plot(dc.threshold,dc[m],color=COLORS[j],lw=1.15 if m=="Nomogram" else .9,label=m)
                ax.plot(dc.threshold,dc.All,ls="--",color="#666",lw=.8,label="Treat all");ax.axhline(0,ls=":",color="black",lw=.8,label="Treat none")
                ax.set(xlim=(.01,.8),ylim=(-.04,.22),xlabel="Threshold probability",ylabel="Net benefit")
            ax.set_title(chr(65+row*3+i)+"  "+NAMES[i].replace(" ","\n",1),loc="left",pad=7)
    handles,labels=axes[3].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.54,.002),ncol=4,frameon=False,fontsize=8.5,handlelength=1.3,columnspacing=.8,labelspacing=.28)
    _save(fig,output,"Fig5")

def plot_nomogram(output,font="Arial"):
    """Exact point rescaling with seven probability labels on one baseline."""
    from .predict import load_specifications,nomogram_points
    output=Path(output)
    if output.exists():raise FileExistsError("Use a new output directory")
    output.mkdir(parents=True)
    spec=load_specifications()["Nomogram"];lower=[0,0,0,0,-3,-4];upper=[1,1,10,1,4,4]
    dummy={k:max(lo,.1) if k=="MR_reported_diameter" else lo for k,lo in zip(spec["input_features"],lower)}
    info=nomogram_points(dummy,spec);unit=info["logit_per_point"];baseline=info["zero_point_eta"]
    maximum=sum(b*(hi-lo)/unit for b,lo,hi in zip(spec["coefficients"],lower,upper))
    plt.rcParams.update({"font.family":font,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(6.5,4.0));fig.subplots_adjust(left=.008,right=.992,top=.992,bottom=.015)
    labels=["HE4","MR report DMI","MR report MTD (cm)","MR report LNM","Radiomics signature","DL signature"]
    def axis(label,y,positions,ticklabels,probability=False):
        ax.text(-30,y,label,va="center",fontsize=9 if label in ["Points","Total points","LNM probability"] else 8)
        positions=np.asarray(positions);ax.plot([min(positions),max(positions)],[y,y],color="#333",lw=.65)
        for x,s in zip(positions,ticklabels):ax.plot([x,x],[y-.06,y+.06],color="#333",lw=.5);ax.text(x,y-.17,s,ha="center",va="top",fontsize=7.5 if probability else 8)
    axis("Points",8,np.arange(0,101,10),[str(x) for x in range(0,101,10)])
    for i,(label,lo,hi,b) in enumerate(zip(labels,lower,upper,spec["coefficients"])):
        ticks=np.arange(lo,hi+1,1 if hi<=1 else 2 if i==2 else 1);axis(label,7-i,(ticks-lo)*b/unit,[f"{v:g}" for v in ticks])
    total_ticks=np.arange(0,maximum+1,25);axis("Total points",.75,total_ticks/maximum*100,[f"{v:g}" for v in total_ticks])
    probabilities=np.array([.001,.01,.1,.5,.9,.99,.999]);positions=(np.log(probabilities/(1-probabilities))-baseline)/unit/maximum*100
    axis("LNM probability",-.12,positions,[f"{v:g}" for v in probabilities],True)
    ax.set(xlim=(-31,105),ylim=(-1.08,8.72));ax.axis("off");_save(fig,output,"Fig3")
