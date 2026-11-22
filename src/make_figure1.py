"""System-architecture figure (Figure 1), English labels; run from the repository root:
    python src/make_figure1.py  ->  figures/fig1_system_architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"]=["DejaVu Sans"]
W,H=5.748,2.72
fig=plt.figure(figsize=(W,H),dpi=400)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis("off")
FS_T,FS_B=7.4,5.9
TITLE_H,PAD=0.10,0.075
LINE=0.108
def box(x,w,cy,fc,title,body):
    n=body.count("\n")+1
    h=PAD+TITLE_H+0.03+n*LINE+0.09
    y0=cy-h/2
    ax.add_patch(FancyBboxPatch((x,y0),w,h,boxstyle="round,pad=0.012,rounding_size=0.05",
                 fc=fc,ec="#333333",lw=0.8,zorder=2))
    ax.text(x+w/2,y0+h-PAD,title,ha="center",va="top",fontsize=FS_T,
            fontweight="bold",color="white",zorder=3)
    ax.text(x+w/2,y0+h-PAD-TITLE_H-0.03,body,ha="center",va="top",fontsize=FS_B,
            color="white",linespacing=1.30,zorder=3)
    return (x,y0,x+w,y0+h)
def arrow(p1,p2):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",lw=1.2,color="#444444",
                 mutation_scale=10,shrinkA=0,shrinkB=0,zorder=1))
C_IN="#2f6fb2"; C_RET="#3f9391"; C_PR="#4f9a49"; C_RV="#d29a3f"; C_CK="#c25c5a"; C_FIN="#4d4d4d"; C_KB="#8a8a8a"
x_in,w_in=0.04,0.86
x_rt,w_rt=1.00,1.36
x_pr,w_pr=2.46,1.14
x_ck,w_ck=3.75,1.08
x_fn,w_fn=4.98,0.72
CY_A,CY_B=2.06,0.82
b_in=box(x_in,w_in,CY_A,C_IN,"User fault input","Hadoop / Spark /\nHive / HBase\nerror text")
b_rt=box(x_rt,w_rt,CY_A,C_RET,"Hybrid retriever","BM25 sparse k=5\nChroma dense k=5\nfusion 0.5 / 0.5\nall-MiniLM-L6-v2, top-5")
b_pr=box(x_pr,w_pr,CY_A,C_PR,"Proposer","drafts from evidence\nwrites NOT COVERED\nwhen unsupported")
b_rv=box(x_pr,w_pr,CY_B,C_RV,"Reviser","checks versions, flags\nINVENTED / UNVERIFIED\npairings")
b_ck=box(x_ck,w_ck,CY_B,C_CK,"Checker","claim-by-claim check\ndrops unsupported text\nignores model knowledge")
b_fn=box(x_fn,w_fn,CY_B,C_FIN,"Final answer","[Cause]\n[Version check]\n[Repair steps]")
b_kb=box(x_rt,w_rt,0.44,C_KB,"Knowledge base","400 verified cases\n8:2 split, seed 42\n100-case held-out test")
arrow((b_in[2],CY_A),(b_rt[0],CY_A))
arrow((b_rt[2],CY_A),(b_pr[0],CY_A))
arrow((x_pr+w_pr/2,b_pr[1]),(x_pr+w_pr/2,b_rv[3]))
arrow((b_rv[2],CY_B),(b_ck[0],CY_B))
arrow((b_ck[2],CY_B),(b_fn[0],CY_B))
arrow((x_rt+w_rt/2,b_kb[3]),(x_rt+w_rt/2,b_rt[1]))
ax.text(x_rt+w_rt/2+0.05,(b_kb[3]+b_rt[1])/2,"index",ha="left",va="center",fontsize=5.4,color="#555555")
fig.savefig("figures/fig1_system_architecture.png",dpi=400,facecolor="white")
print("ok", W, H)
