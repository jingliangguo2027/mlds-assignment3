"""Build plots and the report's data from recorded runs, without inventing results."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from federated import load_clients, model, evaluate, transform
from run_experiments import ROOT, RUNS, restore

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 180})


def read(name):
    return json.loads((ROOT/name).read_text())


def run_record(name):
    return json.loads((RUNS/name).read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG/name, bbox_inches="tight")
    plt.close(fig)


def trajectory(result, title, name):
    fig, axes = plt.subplots(1,2,figsize=(10,3.2))
    for split,label in [("train","Training"),("validation","Validation")]:
        for ax,metric in zip(axes,["loss","accuracy"]):
            ax.plot([r["round"] for r in result["history"]],
                    [r[split][metric] for r in result["history"]],label=label,lw=1.2)
    for ax,label in zip(axes,["Cross-entropy loss","Accuracy"]):
        ax.set(xlabel="Communication round",ylabel=label)
        ax.legend()
    fig.suptitle(title)
    save(fig,name)


@torch.no_grad()
def client_details(net,clients,inputs):
    confusion = np.zeros((62,62),dtype=int)
    points = []
    for i,c in enumerate(clients):
        pred = net(transform(c["vx"],inputs)).argmax(1).numpy()
        true = c["vy"].numpy()
        np.add.at(confusion,(true,pred),1)
        counts = torch.bincount(c["y"],minlength=62)
        baseline = float((c["vy"] == counts.argmax()).float().mean())
        points.append({"client":i,"train_samples":len(c["y"]),"validation_samples":len(true),
                       "accuracy":float(np.mean(pred==true)),"baseline":baseline})
    return points,confusion


def main():
    torch.set_num_threads(1)
    clients = load_clients(ROOT/"Assignment3-data.zip")
    tuning,selection,ray,noise,validation,test = [read(n) for n in [
        "tuning_results.json","selected_config.json","ray_results.json",
        "noise_results.json","validation_results.json","test_results.json"]]
    counts = np.stack([c["counts"] for c in clients])
    sizes = np.array([len(c["y"])+len(c["vy"]) for c in clients])
    all_x = torch.cat([torch.cat([c["x"],c["vx"]]) for c in clients])
    data = {"clients":len(clients),"total":int(sizes.sum()),"train":sum(len(c["y"]) for c in clients),
            "validation":sum(len(c["vy"]) for c in clients),"test":test["samples"],
            "sizes":sizes.tolist(),"coverage":(counts>0).sum(1).tolist(),
            "class_counts":counts.sum(0).tolist(),"pixel_mean":float(all_x.mean()),
            "pixel_std":float(all_x.std()),"pixel_min":float(all_x.min()),"pixel_max":float(all_x.max())}
    fig,axes=plt.subplots(3,2,figsize=(10,5.8),sharex=True,sharey=True)
    for ax,title,row in zip(axes.flat,["All clients"]+[f"Client {i}" for i in range(5)],
                            [counts.sum(0)]+list(counts[:5])):
        ax.bar(range(62),100*row/row.sum(),width=0.8)
        ax.set_title(f"{title} (n={row.sum()})",fontsize=10)
    fig.supxlabel("Class ID")
    fig.supylabel("Images (%)")
    save(fig,"class_distribution.png")

    fig,axes=plt.subplots(1,2,figsize=(10,3.2))
    for inputs,label in [("raw","Original pixels"),("ink","1 - pixels")]:
        rows=sorted([r for r in tuning["rows"] if r["config"]["inputs"]==inputs],key=lambda r:r["config"]["lr"])
        axes[0].plot([r["config"]["lr"] for r in rows],
                     [100*r["validation"]["accuracy"] for r in rows],"o-",label=label)
    axes[0].set(xscale="log",xlabel="Learning rate",ylabel="Validation accuracy (%)",title="100 rounds, C=10%, E=3")
    axes[0].legend()
    base=next(r for r in tuning["rows"] if r["config"]["inputs"]=="raw" and r["config"]["lr"]==0.01)
    best=tuning["chosen"]
    for row,label in [(base,"Original pixels, lr=0.01"),(best,f'1 - pixels, lr={best["config"]["lr"]:g}')]:
        history=run_record(row["run"])["history"]
        axes[1].plot([r["round"] for r in history],[100*r["validation"]["accuracy"] for r in history],label=label,lw=1.2)
    axes[1].set(xlabel="Communication round",ylabel="Validation accuracy (%)",title="Matched initialization and clients")
    axes[1].legend(fontsize=8)
    save(fig,"optimization_diagnosis.png")

    seq=run_record(selection["chosen"]["run"])
    trajectory(seq,f'Sequential FedAvg: C={selection["chosen"]["config"]["clients"]}%, E={selection["chosen"]["config"]["epochs"]}',"sequential_curves.png")
    ray_run=run_record(ray["parallel_run"])
    trajectory(ray_run,"Ray Core: four clients per round","ray_curves.png")

    main_net=restore(RUNS/selection["chosen"]["state"])
    points,confusion=client_details(main_net,clients,selection["chosen"]["config"]["inputs"])
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].scatter([p["baseline"] for p in points],[p["accuracy"] for p in points],s=20,alpha=.7)
    axes[0].plot([0,1],[0,1],"--",color="gray")
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel="Client majority baseline accuracy",ylabel="FedAvg validation accuracy",title="One point per client")
    support=confusion.sum(1)
    recalls=np.divide(np.diag(confusion),support,out=np.zeros(62,dtype=float),where=support>0)
    low=np.argsort(recalls)[:12]
    axes[1].bar(range(len(low)),100*recalls[low])
    axes[1].set_xticks(range(len(low)),[str(i) for i in low])
    axes[1].set(xlabel="Class ID",ylabel="Validation recall (%)",title="Twelve hardest classes")
    save(fig,"client_baseline.png")
    masked=confusion.copy()
    np.fill_diagonal(masked,0)
    flat=np.argsort(masked.ravel())[-5:][::-1]
    pairs=[{"true":int(i//62),"predicted":int(i%62),"count":int(masked.ravel()[i])} for i in flat]
    client_summary={"points":points,"baseline_wins":sum(p["accuracy"]>p["baseline"] for p in points),
        "accuracy_mean":float(np.mean([p["accuracy"] for p in points])),
        "accuracy_p10":float(np.quantile([p["accuracy"] for p in points],.1)),
        "accuracy_min":float(min(p["accuracy"] for p in points)),
        "macro_recall":float(recalls[support>0].mean()),"confusion":confusion.tolist(),
        "common_confusions":pairs,"weakest_clients":sorted(points,key=lambda p:p["accuracy"])[:5]}

    fig,axes=plt.subplots(1,2,figsize=(10,3.5),sharey=True)
    for b,rows in noise["records"].items():
        run=run_record(rows[0]["run"])
        for ax,split in zip(axes,["train","validation"]):
            ax.plot([r["round"] for r in run["history"]],[100*r[split]["accuracy"] for r in run["history"]],label=f"b={float(b):g}",lw=1.1)
    for ax,title in zip(axes,["Actual training inputs (with noise)","Clean validation inputs"]):
        ax.set(xlabel="Communication round",ylabel="Accuracy (%)",title=title)
        ax.legend(fontsize=8)
    save(fig,"noise_round_curves.png")

    scales=[float(b) for b in noise["records"]]
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    rows=[v[0] for v in noise["records"].values()]
    axes[0].plot(scales,[100*r["train"]["accuracy"] for r in rows],"o-",label="Noisy training")
    axes[0].plot(scales,[100*r["clean_train"]["accuracy"] for r in rows],"^-",label="Clean training")
    axes[0].plot(scales,[100*test["noise"][str(b)]["accuracy"] for b in scales],"s-",label="Clean test")
    axes[0].set(xlabel="Laplace scale b",ylabel="Final accuracy (%)",title="Seed 42")
    axes[0].legend(fontsize=8)
    values=[[100*r["validation"]["accuracy"] for r in rows] for rows in noise["records"].values()]
    axes[1].errorbar(scales,[np.mean(v) for v in values],yerr=[np.std(v,ddof=1) for v in values],fmt="o-",capsize=4)
    axes[1].set(xlabel="Laplace scale b",ylabel="Validation accuracy (%)",title="Mean +/- sample SD, three seeds")
    save(fig,"noise_final_accuracy.png")

    fresh=validation["fresh_noise_evaluation"]
    fig,ax=plt.subplots(figsize=(8,2.8))
    for key,label,marker in [("stored_noise_accuracy","Stored training noise","o"),
                             ("fresh_noise_accuracy","Fresh independent noise","s"),
                             ("clean_train_accuracy","Clean training images","^")]:
        ax.plot([float(b) for b in fresh],[100*v[key] for v in fresh.values()],marker+"-",label=label)
    ax.set(xlabel="Laplace scale b",ylabel="Accuracy on training images (%)")
    ax.legend(fontsize=9)
    save(fig,"fresh_noise_check.png")

    noisy_net=restore(RUNS/noise["records"][str(noise["choice"])][0]["state"])
    clean_net=restore(RUNS/noise["records"]["0.0"][0]["state"])
    changes=[]
    for c in clients:
        a=evaluate(clean_net,c["vx"],c["vy"],noise["config"]["inputs"])["accuracy"]
        b=evaluate(noisy_net,c["vx"],c["vy"],noise["config"]["inputs"])["accuracy"]
        changes.append(100*(b-a))
    fig,axes=plt.subplots(1,2,figsize=(10,3.3))
    axes[0].scatter([len(c["y"]) for c in clients],changes,s=22,alpha=.75)
    axes[0].axhline(0,color="gray",ls="--")
    axes[0].set(xlabel="Client training images",ylabel="Validation accuracy change (pp)",title=f'Chosen b={noise["choice"]:g} versus b=0')
    axes[1].hist(changes,bins=15,edgecolor="white")
    axes[1].axvline(0,color="gray",ls="--")
    axes[1].set(xlabel="Validation accuracy change (pp)",ylabel="Number of clients",title="Who gains and who loses?")
    save(fig,"noise_client_effect.png")
    noise["client_effect"]={"b":noise["choice"],"median_pp":float(np.median(changes)),
        "clients_worse":sum(c<0 for c in changes),"clients_better":sum(c>0 for c in changes),
        "clients_equal":sum(c==0 for c in changes),"worst_pp":float(min(changes)),"best_pp":float(max(changes)),"changes_pp":changes}

    summary={"data":data,"tuning":tuning,"selection":selection,"sequential":seq,"ray":ray,
             "ray_run":ray_run,"noise":noise,"validation":validation,"test":test,"clients":client_summary}
    (ROOT/"experiment_results.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"test":test,"client_summary":{k:v for k,v in client_summary.items() if k not in ["points","confusion"]},
                      "noise_choice":noise["choice"],
                      "client_noise_effect":{k:v for k,v in noise["client_effect"].items() if k!="changes_pp"}},indent=2))


if __name__=="__main__":
    main()
