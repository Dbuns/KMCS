"""Regenerate corrected DOE, reference-trend and example figures; no KMC runs."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {1:(.6,.6,.6), 2:(0,.45,.70), 3:(.9,.6,0)}


def historical_table():
    tree = ast.parse((ROOT/"DOE_plot.py").read_text(encoding="utf-8"))
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="data" for t in n.targets))


def doe_matrix(x1,x2):
    return np.column_stack([np.ones_like(x1),x1,x2,x1*x2,x1*x1,x2*x2])


def main():
    data = json.loads((ROOT/"validation/full_size_doe_2026-10-21.json").read_text())
    rows = data["results"]
    target = ROOT/"figures"
    temperatures = np.array([r["T_C"] for r in rows],dtype=float)
    frequencies = np.array([r["f_Hz"] for r in rows],dtype=float)
    density = np.array([r["density_per_nm2"] for r in rows])
    code_f = lambda f: 2*(f-frequencies.min())/np.ptp(frequencies)-1
    code_t = lambda t: 2*(t-temperatures.min())/np.ptp(temperatures)-1
    matrix = doe_matrix(code_f(frequencies),code_t(temperatures))
    coefficients = np.linalg.lstsq(matrix,density,rcond=None)[0]
    fit = matrix@coefficients
    r2 = 1-np.sum((density-fit)**2)/np.sum((density-density.mean())**2)
    f_grid,t_grid = np.meshgrid(np.linspace(.5,50,201),np.linspace(700,900,201))
    z = (doe_matrix(code_f(f_grid.ravel()),code_t(t_grid.ravel()))@coefficients).reshape(f_grid.shape)
    fig = plt.figure(figsize=(10,7.5))
    grid = fig.add_gridspec(2,3,width_ratios=[5,1.5,.18],height_ratios=[1.4,4.5],wspace=.15,hspace=.1)
    top = fig.add_subplot(grid[0,0])
    ax = fig.add_subplot(grid[1,0],sharex=top)
    side = fig.add_subplot(grid[1,1],sharey=ax)
    cax = fig.add_subplot(grid[1,2])
    levels = np.linspace(0,max(z.max(),density.max()),13)
    contour = ax.contourf(f_grid,t_grid,np.ma.masked_less(z,0),levels=levels,cmap="viridis")
    if z.min()<0:
        ax.contourf(f_grid,t_grid,z,levels=[z.min()-1,0],colors=["#eeeeee"],hatches=["////"])
    ax.scatter(frequencies,temperatures,s=28,facecolors="white",edgecolors="black",linewidths=.7,clip_on=False)
    ax.axhline(800,color="#D55E00",ls="--",lw=1)
    ax.axvline(20,color="#56B4E9",ls="--",lw=1)
    ax.set(xlabel="Frequency (Hz)",ylabel="Temperature (°C)")
    for panel,key,fixed_key,fixed,color in [(top,"f_Hz","T_C",800,"#D55E00"),(side,"T_C","f_Hz",20,"#0072B2")]:
        cut = sorted((r for r in rows if r[fixed_key]==fixed),key=lambda r:r[key])
        xs = [r[key] for r in cut]
        ys = [r["density_per_nm2"] for r in cut]
        if panel is top:
            panel.plot(xs,ys,"o-",color=color,ms=4)
            panel.set(ylabel="Density (nm⁻²)",title="Temperature fixed at 800 °C")
            panel.tick_params(labelbottom=False)
        else:
            panel.plot(ys,xs,"o-",color=color,ms=4)
            panel.set(xlabel="Density (nm⁻²)",title="20 Hz")
            panel.tick_params(labelleft=False)
        panel.grid(alpha=.2)
    fig.colorbar(contour,cax=cax,label="Feature density (nm⁻²)")
    fig.suptitle(f"Corrected KMCS: full-size 25-condition DOE\nQuadratic response-surface fit (R² = {r2:.3f})",fontsize=13)
    fig.text(.5,.025,"Points and side curves: completed simulations | 128×128×35 | 300 pulses | seed 42\n"
             "Hatching: negative quadratic predictions (unphysical fit region, not negative simulation data)",ha="center",fontsize=9)
    fig.subplots_adjust(top=.86,bottom=.15)
    fig.savefig(target/"number_density.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    print(f"DOE fit R2={r2:.6f}; min prediction={z.min():.6g}. Negative fit region explicitly hatched.")

    refs = json.loads((ROOT/"validation/reference_data.json").read_text())
    historical = historical_table()
    fig,axes = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for ax,key,fixed_key,fixed,anchor,refkey in [(axes[0],"T_C","f_Hz",20,800,"temperature_sweep"),
                                               (axes[1],"f_Hz","T_C",800,20,"frequency_sweep")]:
        cut = sorted((r for r in rows if r[fixed_key]==fixed),key=lambda r:r[key])
        xs = np.array([r[key] for r in cut]); ys = np.array([r["density_per_nm2"] for r in cut])
        ax.plot(xs,ys/ys[np.flatnonzero(xs==anchor)[0]],"o-",color="#D55E00",label="Corrected Python (new runs)")
        ref = refs[refkey]
        rx = np.array(ref["temperature_C" if key=="T_C" else "frequency_Hz"])
        index = np.flatnonzero(rx==anchor)[0]
        for name,field,color in [("Experiment","experiment","#222222"),("Original-paper KMC","original_paper_kmcs","#8B5AA6")]:
            values = np.array(ref[field])
            ax.plot(rx,values/values[index],"o:",color=color,label=name,ms=4)
        old = sorted((t if key=="T_C" else f,d) for f,t,d in historical
                     if (f if key=="T_C" else t)==fixed and (key!="T_C" or 700<=t<=900))
        hx,hy = np.array(old).T
        ax.plot(hx,hy/hy[np.flatnonzero(hx==anchor)[0]],"s--",color="#009E73",label="Historical Python DOE",ms=4)
        ax.set(xlabel="Temperature (°C)" if key=="T_C" else "Frequency (Hz)",
               ylabel="Relative feature density (central point = 1)",title="At 20 Hz" if key=="T_C" else "At 800 °C")
        if key=="f_Hz": ax.set_xscale("log")
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle("Full-size trend comparison — each dataset normalized independently",fontsize=12)
    fig.savefig(target/"reference_trends.png",dpi=200)
    plt.close(fig)

    with np.load(ROOT/"examples/snapshot_final.npz",allow_pickle=False) as saved:
        top_species,occ = saved["Occ_top"],saved["Occ"]
    rgb = np.zeros((*top_species.shape,3))
    for species,color in COLORS.items(): rgb[top_species==species] = color
    fig,ax = plt.subplots(figsize=(5.5,5.5),layout="constrained")
    ax.imshow(rgb,origin="lower",interpolation="nearest")
    ax.set_title("Corrected KMCS: 900 °C, 2 Hz\n128×128 cells | 300 pulses | seed 42",fontsize=11)
    ax.axis("off")
    fig.savefig(target/"snapshot_2mat.png",dpi=180)
    plt.close(fig)
    plt.imsave(ROOT/"examples/topview.png",rgb,origin="lower")

    stride = 2
    volume = np.transpose(occ[::stride,::stride,::stride],(1,0,2))
    occupied = volume!=0
    colors = np.zeros((*volume.shape,4))
    for species,color in COLORS.items(): colors[volume==species] = (*color,1)
    x,y,z = np.indices(np.array(volume.shape)+1)*stride
    fig = plt.figure(figsize=(8,6.5),layout="constrained")
    ax = fig.add_subplot(111,projection="3d")
    ax.voxels(x,y,z,occupied,facecolors=colors,edgecolor=(0,0,0,.1),linewidth=.1)
    ax.set(xlabel="x (lattice cells)",ylabel="y (lattice cells)",zlabel="z (lattice cells)",zlim=(0,20))
    ax.set_box_aspect((128,128,45))
    ax.view_init(elev=28,azim=-55)
    ax.set_title("Corrected two-material morphology: 900 °C, 2 Hz\n300 pulses | seed 42 | visualization stride 2",fontsize=11)
    fig.savefig(target/"snapshot_2mat_3d.png",dpi=180)
    plt.close(fig)
    print("Saved corrected DOE, reference trends and two-material example figures.")


if __name__=="__main__":
    main()
