"""Figures/tables presenting the frozen panel study, with no model fitting."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd

from . import module02_panel as p

COLORS = {"history":"#4C72B0", "F":"#55A868", "D":"#C44E52", "development":"#8172B3", "main":"#55A868"}
BANDS = ["[0.00, 0.25)","[0.25, 0.50)","[0.50, 0.75)","[0.75, 1.00]"]


def canvas(cfg, factor=1):
    return plt.subplots(figsize=(cfg.img_dim1*cfg.publication_scale,cfg.img_dim2*cfg.publication_scale*factor))


def dates(ax, count=5):
    locator=mdates.AutoDateLocator(minticks=3,maxticks=count)
    ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def architecture(cfg):
    fig,ax=canvas(cfg,.7);ax.axis("off");ax.set(xlim=(0,12),ylim=(0,5))
    boxes=[(.15,3.4,3.3,.8,"Development: CA_1 + TX_1\nChoose one fixed family F"),
           (4.2,3.4,3.3,.8,"Main: 100 products × 3 stores\nRepeat at six forecast origins"),
           (8.35,3.4,3.3,.8,"A, B, R, I, F, D\nSame information and models"),
           (1.5,1.7,3.8,.8,"Shared numerical bank\nReuse exact training-window fits"),
           (6.6,1.7,3.8,.8,"Freeze every main forecast\nThen open outcomes and score"),
           (1.5,.15,3.8,.65,"Paired errors + coverage\nAcross stores, dates and zero bands"),
           (6.6,.15,3.8,.65,"Separate uncached B/D benchmark\n36 predefined cases")]
    for x,y,width,height,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle="round,pad=.06",facecolor="#eaf0f4",edgecolor="#25455d"))
        ax.text(x+width/2,y+height/2,text,ha="center",va="center",fontsize=11)
    for start,end in [((3.55,3.8),(4.1,3.8)),((7.6,3.8),(8.25,3.8)),((10,3.3),(9,2.6)),((5.4,2.1),(6.5,2.1)),((3.4,1.6),(3.4,.9)),((8.5,1.6),(8.5,.9))]:
        ax.annotate("",end,start,arrowprops={"arrowstyle":"->","color":"#25455d"})
    ax.set_title("Routing is evaluated across requests, locations and time",pad=12)
    fig.tight_layout();return fig


def raw_histories(cfg,data,protocol):
    fig,axes=plt.subplots(2,3,figsize=(cfg.img_dim1*cfg.publication_scale,cfg.img_dim2*cfg.publication_scale*1.1),layout="constrained",sharex=True)
    for row,item in enumerate(protocol["raw_plot_item_ids"]):
        for col,store in enumerate(p.STORES):
            case=next(c for c in data.cases if c["item_id"]==item and c["store"]==store and c["day"]==p.MAIN_DAYS[0])
            h=p.history(data,case);ax=axes[row,col]
            ax.plot(h.index,h,color=COLORS["history"],linewidth=1.5)
            ax.set_title(f"{item} · {store}",fontsize=11)
            ax.set_ylabel("Recorded item units / day");dates(ax,4)
    return fig


def timeline(cfg,protocol):
    fig,axes=plt.subplots(2,1,figsize=(cfg.img_dim1*cfg.publication_scale,cfg.img_dim2*cfg.publication_scale),layout="constrained")
    ax=axes[0]
    for y,role in [(1,"development"),(0,"main")]:
        for origin in protocol[role+"_origins"]:
            start=pd.Timestamp(origin)+pd.Timedelta(days=1)
            ax.broken_barh([(mdates.date2num(start),28)],(y-.25,.5),facecolor=COLORS[role],edgecolor="white",linewidth=2)
            ax.plot(pd.Timestamp(origin),y,"|",color="black",markersize=16)
    ax.set_yticks([0,1],["Main: all three stores","Development: CA_1 / TX_1"])
    ax.set_ylim(-.7,1.7);dates(ax,7);ax.set_title("Each colored block is a separate 28-day forecast evaluation",fontsize=12)
    ax.axvline(pd.Timestamp(protocol["main_origins"][0]),color="#25455d",linestyle="--",linewidth=1)
    ax=axes[1]
    origin=pd.Timestamp(protocol["main_origins"][0])
    for i in range(3):
        end=origin-pd.Timedelta(days=(3-i)*28)
        train_start=end-pd.Timedelta(days=364)
        ax.broken_barh([(mdates.date2num(train_start),365)],(3-i-.23,.46),facecolor="#4C72B0")
        ax.broken_barh([(mdates.date2num(end+pd.Timedelta(days=1)),28)],(3-i-.23,.46),facecolor="#DD8452")
    ax.broken_barh([(mdates.date2num(origin-pd.Timedelta(days=364)),365)],(-.23,.46),facecolor="#4C72B0")
    ax.broken_barh([(mdates.date2num(origin+pd.Timedelta(days=1)),28)],(-.23,.46),facecolor=COLORS["main"])
    ax.set_yticks([0,1,2,3],["Final fit + outer test","Inner fold 3","Inner fold 2","Inner fold 1"])
    ax.set_ylim(-.7,4.2);dates(ax,6);ax.set_title("Inside the first main origin: three inner folds select candidates",fontsize=12)
    ax.legend(handles=[Patch(color="#4C72B0",label="365-day fit"),Patch(color="#DD8452",label="28-day validation"),Patch(color=COLORS["main"],label="28-day outer test")],
              loc="upper left",ncol=3,frameon=True,facecolor="white",fontsize=9)
    return fig


def error_by_origin(cfg,summary):
    fig,axes=plt.subplots(1,3,figsize=(cfg.img_dim1*cfg.publication_scale,cfg.img_dim2*cfg.publication_scale*.7),layout="constrained",sharey=True)
    for ax,store in zip(axes,p.STORES):
        rows=[r for r in summary["by_store_origin"] if r["store"]==store]
        xs=pd.to_datetime([r["origin"] for r in rows])
        for other,label in [("F","B − fixed F"),("D","B − all candidates D")]:
            ys=[r[other]["delta_mase"]*1000 if r[other]["delta_mase"] is not None else np.nan for r in rows]
            ax.plot(xs,ys,label=label,color=COLORS[other],marker="o",linewidth=2)
        ax.axhline(0,color="#25455d",linewidth=1)
        ax.set_title(store+" · 100 cases / origin",fontsize=12);dates(ax,4)
        ax.legend(loc="best",fontsize=9,frameon=True,facecolor="white",framealpha=.95)
    axes[0].set_ylabel("Paired mean ΔMASE × 1,000\nNegative favors routing B")
    return fig


def error_by_band(cfg,summary):
    fig,ax=canvas(cfg,.65);xs=np.arange(4);width=.34
    for offset,other,label in [(-.5,"F","B − fixed F"),(.5,"D","B − all candidates D")]:
        values=[r[other]["delta_mase"]*1000 if r[other]["delta_mase"] is not None else np.nan for r in summary["by_zero_band"]]
        ax.bar(xs+offset*width,values,width=width,color=COLORS[other],label=label)
    ax.axhline(0,color="#25455d",linewidth=1)
    ax.set_xticks(xs,[f"{BANDS[r['band']]}\nn = {r['F']['scaled_n']}" for r in summary["by_zero_band"]])
    ax.set_xlabel("Zero fraction in the pre-origin 365-day window")
    ax.set_ylabel("Paired mean ΔMASE × 1,000\nNegative favors routing B")
    ax.set_title("The threshold switches B's family at a zero fraction of 0.50",fontsize=13)
    ax.legend(frameon=True,facecolor="white",framealpha=.95);fig.tight_layout();return fig


def cohort_table(protocol):
    rows=[[d,n,n*3,n*2*3,n*3*6] for d,n in protocol["quotas"].items()]
    rows.append(["Total",100,300,600,1800])
    return [(["Department","Items","Item–store series","Development cases","Main cases"],rows,[.24,.12,.21,.22,.21])]


def protocol_table(fixed=None):
    return [(["Arm","Decision","Numerical candidates"],
             [["A","Seasonal naive","Seasonal naive"],
              ["B","Zero fraction ≥ 0.50 → intermittent; otherwise regular","Selected family's two methods"],
              ["R","Always regular","Seasonal naive; damped additive Holt–Winters"],
              ["I","Always intermittent","Seasonal naive; TSB (0.10, 0.10)"],
              ["F","Choose R/I using development only" if fixed is None else "Development chooses "+fixed["selected_arm"],"Same family for every main case"],
              ["D","All-candidates selection","Seasonal naive; Holt–Winters; TSB"]],[.08,.46,.46]),
            (["Evaluation level","Training / selection","Unseen observations"],
             [["Inner","3 folds; 365 days before each fold","Next 28 days, all before outer origin"],
              ["Outer","449 visible days; final fit uses last 365","Next 28 days after the outer origin"],
              ["Fixed control","600 development cases from CA_1/TX_1","1,800 main cases; WI_1 excluded from choice"]],[.2,.4,.4])]


def accuracy_table(summary):
    rows=[[a,s["delivered_n"],s["common_n"],s["scaled_n"],s["macro_mase"],s["numerical_fallback_n"]] for a,s in summary["arms"].items()]
    contrasts=[]
    for other in ("F","D"):
        c=summary["contrasts"][other];b=summary["bootstrap"][other]
        contrasts.append(["B − "+other,c["delta_mase"]*1000,b["low"]*1000,b["high"]*1000,c["routing_wins"],c["routing_losses"],c["ties"]])
    return [(["Arm","Delivered / 1800","Common n","Scaled n","Macro MASE","Final fallbacks"],rows,[.08,.23,.16,.16,.18,.19]),
            (["Contrast","ΔMASE × 1000","2.5% × 1000","97.5% × 1000","MAE wins","MAE losses","Ties"],contrasts,[.12,.19,.18,.18,.12,.12,.09])]


def work_table(work):
    bank=work["bank"]
    rows=[[a,n] for a,n in work["logical_main"].items()]
    timed=[[a,r["n"],r["fit_attempts"],r["seconds"],r["median_seconds"]] for a,r in work["uncached_timing"].items()]
    return [(["Shared development + main bank","Count / fit seconds"],
             [["Numerical fit lookups",bank["lookups"]],["Unique actual fit attempts",bank["actual_fit_attempts"]],
              ["Exact-fit cache hits",bank["cache_hits"]],["Failed fit attempts retained",bank["failed_fits"]],
              ["Sum of measured fit seconds",bank["seconds"]]],[.68,.32]),
            (["Main arm","Logical fit attempts across 1800 cases"],rows,[.3,.7]),
            (["Uncached arm","Cases","Fit attempts","Total seconds","Median seconds"],timed,[.22,.14,.2,.22,.22])]
