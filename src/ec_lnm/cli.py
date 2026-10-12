import argparse,csv
from pathlib import Path
from .predict import load_specifications,predict

def main():
    p=argparse.ArgumentParser(description="EC LNM research tools; no clinical decision threshold is validated")
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("predict");q.add_argument("input",type=Path);q.add_argument("output",type=Path);q.add_argument("--model",default="Nomogram");q.add_argument("--spec",type=Path)
    q=sub.add_parser("evaluate");q.add_argument("--train",required=True);q.add_argument("--internal",required=True);q.add_argument("--external",required=True);q.add_argument("--output",required=True);q.add_argument("--refit-training",action="store_true");q.add_argument("--bootstrap",type=int,default=1000);q.add_argument("--calibration-curves",action="store_true")
    q=sub.add_parser("early-subgroup");q.add_argument("--internal",required=True);q.add_argument("--external",required=True);q.add_argument("--output",required=True);q.add_argument("--bootstrap",type=int,default=20000)
    q=sub.add_parser("plot");q.add_argument("results",type=Path);q.add_argument("output",type=Path)
    q=sub.add_parser("nomogram");q.add_argument("output",type=Path)
    args=p.parse_args()
    if args.command=="predict":
        specs=load_specifications(args.spec)
        if args.model not in specs:p.error("Unknown model")
        if args.output.exists() or args.input.resolve()==args.output.resolve():p.error("Use a new output file")
        spec=specs[args.model]
        with args.input.open(encoding="utf-8-sig",newline="") as f:
            reader=csv.DictReader(f);fields=list(reader.fieldnames or []);rows=list(reader)
        if set(spec["input_features"])-set(fields):p.error("Required columns missing")
        added=[args.model+"_probability"]+([args.model+"_prediction"] if spec.get("threshold") is not None else [])
        if set(added)&set(fields):p.error("Output columns already present")
        for row in rows:
            probability=predict(row,spec);row[added[0]]=format(probability,".17g")
            if len(added)>1:row[added[1]]=int(probability>=spec["threshold"])
        with args.output.open("x",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=fields+added);writer.writeheader();writer.writerows(rows)
        print(f"{len(rows)} rows predicted")
    elif args.command=="evaluate":
        from .analysis import evaluate
        evaluate({co:getattr(args,co) for co in ["train","internal","external"]},args.output,args.refit_training,args.bootstrap,args.calibration_curves)
    elif args.command=="early-subgroup":
        from .subgroups import pooled_early
        pooled_early(args.internal,args.external,args.output,args.bootstrap)
    elif args.command=="plot":
        from .figures import plot_results
        plot_results(args.results,args.output)
    else:
        from .figures import plot_nomogram
        plot_nomogram(args.output)
