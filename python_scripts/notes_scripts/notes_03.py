import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

r_val, K_val = 1.0, 4.0
f_log = lambda x: r_val * x * (1 - x / K_val)
x_arr = np.linspace(-0.5, 6.5, 400)
f_arr = f_log(x_arr)

fig = plt.figure(figsize=(11, 5))
gs  = fig.add_gridspec(2, 2, width_ratios=[1.8, 1], hspace=0.5, wspace=0.35)
ax_f  = fig.add_subplot(gs[0, 0])
ax_pl = fig.add_subplot(gs[1, 0])
ax_t  = fig.add_subplot(gs[:, 1])

# ── f(x) graph ──────────────────────────────────────────────
ax_f.plot(x_arr, f_arr, color='steelblue', lw=2.5)
ax_f.axhline(0, color='k', lw=0.8)
ax_f.fill_between(x_arr, 0, f_arr, where=(f_arr > 0),
                  alpha=0.15, color='seagreen', label='$f>0$ (increasing)')
ax_f.fill_between(x_arr, 0, f_arr, where=(f_arr < 0),
                  alpha=0.15, color='crimson',  label='$f<0$ (decreasing)')
ax_f.plot([0, K_val], [0, 0], 'ko', markersize=8, zorder=5)
ax_f.set_xlabel('$x$'); ax_f.set_ylabel("$f(x)$")
ax_f.set_title(r"$f(x)=x(1-x/4)$"); ax_f.legend(fontsize=8)
ax_f.set_xlim(-0.5, 6.5)

# ── Phase line ───────────────────────────────────────────────
ax_pl.axhline(0, color='k', lw=1.2)
ax_pl.plot(0,     0, 'o', markersize=12, color='crimson',  zorder=5)
ax_pl.plot(K_val, 0, 'o', markersize=12, color='seagreen', zorder=5)
ax_pl.annotate('', xy=(-0.3, 0), xytext=(0.3, 0),
               arrowprops=dict(arrowstyle='<-', color='crimson', lw=2))
for xm in [1.5, 2.5]:
    ax_pl.annotate('', xy=(xm+0.5, 0), xytext=(xm-0.5, 0),
                   arrowprops=dict(arrowstyle='->', color='seagreen', lw=2))
for xm in [5.0, 5.8]:
    ax_pl.annotate('', xy=(xm-0.5, 0), xytext=(xm+0.5, 0),
                   arrowprops=dict(arrowstyle='<-', color='crimson', lw=2))
ax_pl.text(0,   0.25, '$x^*=0$\n(unstable)', ha='center', fontsize=9, color='crimson')
ax_pl.text(K_val, 0.25, f'$x^*={K_val}$\n(stable)', ha='center', fontsize=9, color='seagreen')
ax_pl.set_xlim(-0.8, 7); ax_pl.set_ylim(-0.6, 0.7)
ax_pl.set_yticks([]); ax_pl.set_xlabel('$x$')
ax_pl.set_title('Phase line')

# ── Solution curves ──────────────────────────────────────────
t_eval = np.linspace(0, 8, 400)
f_ode  = lambda t, x: [r_val * x[0] * (1 - x[0] / K_val)]
colors_t = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
for x0, color in zip([0.3, 1.0, 2.0, 3.0, 6.0, 8.0], colors_t):
    sol = solve_ivp(f_ode, (0, 8), [x0], t_eval=t_eval, max_step=0.05)
    ax_t.plot(sol.t, sol.y[0], color=color, lw=2, label=f'$x_0={x0}$')
ax_t.axhline(K_val, color='seagreen', ls='--', lw=1.8, label=f'$K={K_val}$')
ax_t.axhline(0,     color='crimson',  ls='--', lw=1.2)
ax_t.set_xlabel('$t$'); ax_t.set_ylabel('$x(t)$')
ax_t.set_title('Solutions vs. time')
ax_t.legend(fontsize=7, ncol=2)

plt.suptitle(r"Logistic: $x' = x(1-x/4)$", fontsize=12, y=1.01)
plt.tight_layout()
plt.show()

f_cub  = lambda x: x**3 - x
x_arr2 = np.linspace(-1.6, 1.6, 400)
f_arr2 = f_cub(x_arr2)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

# f(x) graph
ax = axes[0]
ax.plot(x_arr2, f_arr2, color='steelblue', lw=2.5)
ax.axhline(0, color='k', lw=0.8)
ax.fill_between(x_arr2, 0, f_arr2, where=(f_arr2 > 0),
                alpha=0.15, color='crimson',  label='$f>0$')
ax.fill_between(x_arr2, 0, f_arr2, where=(f_arr2 < 0),
                alpha=0.15, color='steelblue', label='$f<0$')
for xe in [-1, 0, 1]:
    ax.plot(xe, 0, 'ko', markersize=8, zorder=5)
ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
ax.set_title(r"$f(x)=x^3-x$"); ax.legend(fontsize=8)

# Phase line
ax2 = axes[1]
ax2.axhline(0, color='k', lw=1.2)
stab_colors = ['crimson', 'seagreen', 'crimson']
for xe, sc in zip([-1, 0, 1], stab_colors):
    ax2.plot(xe, 0, 'o', markersize=12, color=sc, zorder=5)
arrow_cfg = [(-1.35, 'crimson', '->'), (-0.5, 'steelblue', '<-'),
             (0.5, 'steelblue', '->'), (1.35, 'crimson', '<-')]
for xm, color, style in arrow_cfg:
    ax2.annotate('', xy=(xm+0.2*(-1 if '<-' in style else 1), 0),
                 xytext=(xm, 0),
                 arrowprops=dict(arrowstyle='->', color=color, lw=2))
ax2.text(-1, 0.3, 'unstable', ha='center', fontsize=8, color='crimson')
ax2.text(0,  0.3, 'stable',   ha='center', fontsize=8, color='seagreen')
ax2.text(1,  0.3, 'unstable', ha='center', fontsize=8, color='crimson')
ax2.set_xlim(-1.8, 1.8); ax2.set_ylim(-0.5, 0.6)
ax2.set_yticks([]); ax2.set_xlabel('$x$')
ax2.set_title('Phase line')

# Solutions
ax3 = axes[2]
f_ode2  = lambda t, x: [x[0]**3 - x[0]]
t_eval2 = np.linspace(0, 5, 400)
for x0, color in zip([-1.3, -0.8, -0.3, 0.3, 0.8, 1.3],
                     plt.cm.coolwarm(np.linspace(0.05, 0.95, 6))):
    sol = solve_ivp(f_ode2, (0, 3), [x0], t_eval=np.linspace(0,3,400),
                    max_step=0.01, events=lambda t,y: abs(y[0])-8)
    ax3.plot(sol.t, np.clip(sol.y[0], -5, 5), color=color, lw=2, label=f'$x_0={x0}$')
for xe in [-1, 0, 1]:
    ax3.axhline(xe, ls='--', lw=1.2,
                color='crimson' if xe != 0 else 'seagreen')
ax3.set_xlabel('$t$'); ax3.set_ylabel('$x(t)$')
ax3.set_title('Solutions'); ax3.legend(fontsize=7)
ax3.set_ylim(-3, 3)

plt.suptitle(r"$x' = x^3 - x$", fontsize=12)
plt.tight_layout()
plt.show()

t_sym, x_sym, r_sym, K_sym = sym.symbols('t x r K', real=True, positive=True)

for label, f_expr in [
    ("Logistic $f(x)=rx(1-x/K)$",
     r_sym * x_sym * (1 - x_sym / K_sym)),
    ("Cubic $f(x)=x^3-x$",
     x_sym**3 - x_sym),
]:
    fp = sym.diff(f_expr, x_sym)
    print(f"\n{label}")
    print(f"  f'(x)  = {fp}")
    for xe, name in [(0, "x*=0"), (K_sym, "x*=K"), (1, "x*=1"), (-1, "x*=-1")]:
        try:
            val = fp.subs(x_sym, xe)
            val_s = sym.simplify(val)
            print(f"  f'({xe}) = {val_s}")
        except Exception:
            pass

f_nh  = lambda x: x**2
x_arr = np.linspace(-1.5, 1.5, 400)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].plot(x_arr, f_nh(x_arr), color='steelblue', lw=2.5)
axes[0].axhline(0, color='k', lw=0.8)
axes[0].fill_between(x_arr, 0, f_nh(x_arr),
                     where=(f_nh(x_arr) > 0), alpha=0.15, color='crimson', label='$f>0$ (everywhere)')
axes[0].plot(0, 0, 'ks', markersize=10, zorder=5, label='$x^*=0$ (semi-stable)')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$f(x)$')
axes[0].set_title(r"$f(x)=x^2$: non-hyperbolic equilibrium")
axes[0].legend(fontsize=8)

f_ode_nh = lambda t, x: [x[0]**2]
for x0, color in zip([-1.2, -0.7, -0.3, 0.3, 0.7, 1.2],
                     plt.cm.RdBu(np.linspace(0.05, 0.95, 6))):
    sol = solve_ivp(f_ode_nh, (0, 1.8), [x0], t_eval=np.linspace(0,1.8,400),
                    max_step=0.005, events=lambda t,y: abs(y[0])-8)
    axes[1].plot(sol.t, np.clip(sol.y[0], -5, 5), color=color, lw=2, label=f'$x_0={x0}$')
axes[1].axhline(0, color='k', ls='--', lw=1.5, label='$x^*=0$')
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x(t)$')
axes[1].set_title('Semi-stable: solutions below approach 0, solutions above blow up')
axes[1].legend(fontsize=7, ncol=2); axes[1].set_ylim(-4, 4)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

# f(x) for three r values
x_arr = np.linspace(-2.2, 2.2, 400)
colors3 = ['crimson', 'darkorange', 'steelblue']
for r_val, color, lbl in zip([-0.5, 0, 1.0], colors3, ['$r=-0.5$','$r=0$','$r=1$']):
    axes[0].plot(x_arr, r_val - x_arr**2, color=color, lw=2, label=lbl)
axes[0].axhline(0, color='k', lw=0.8)
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$f(x) = r - x^2$')
axes[0].set_title('$f(x)$ for three values of $r$')
axes[0].legend(fontsize=9); axes[0].set_ylim(-2, 2)

# Phase lines
ax_pl = axes[1]
y_offsets = {'$r=-0.5$': -0.7, '$r=0$': 0.0, '$r=1$': 0.7}
for r_val, color, lbl in zip([-0.5, 0, 1.0], colors3, ['$r=-0.5$','$r=0$','$r=1$']):
    y0 = y_offsets[lbl]
    ax_pl.axhline(y0, color=color, lw=1.2, alpha=0.6)
    ax_pl.text(-2.0, y0+0.07, lbl, fontsize=8, color=color)
    if r_val < 0:
        for xm in [-1.5, 0.0, 1.5]:
            sign = -1 if r_val - xm**2 < 0 else 1
            ax_pl.annotate('', xy=(xm + 0.25*sign, y0),
                           xytext=(xm, y0),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    elif r_val == 0:
        ax_pl.plot(0, y0, 's', color=color, markersize=10, zorder=5)
        for xm in [-1.5, 1.5]:
            sign = -1 if -xm**2 < 0 else 1
            ax_pl.annotate('', xy=(xm + 0.25*sign, y0),
                           xytext=(xm, y0),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    else:
        xe_s =  np.sqrt(r_val)
        xe_u = -np.sqrt(r_val)
        ax_pl.plot(xe_s, y0, 'o', color='steelblue', markersize=10, zorder=5)
        ax_pl.plot(xe_u, y0, 'o', color='crimson',   markersize=10, zorder=5,
                   markerfacecolor='white', markeredgewidth=2)
        for xm in [-1.8, 0.0]:
            sign = -1 if r_val - xm**2 < 0 else 1
            ax_pl.annotate('', xy=(xm + 0.25*sign, y0),
                           xytext=(xm, y0),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
        ax_pl.annotate('', xy=(xe_s+0.3, y0), xytext=(xe_s+0.05, y0),
                       arrowprops=dict(arrowstyle='<-', color=color, lw=1.5))
ax_pl.set_xlim(-2.2, 2.2); ax_pl.set_ylim(-1.1, 1.1)
ax_pl.set_xlabel('$x$'); ax_pl.set_yticks([])
ax_pl.set_title('Phase lines')

# Bifurcation diagram
r_bif  = np.linspace(-0.05, 2.5, 400)
r_pos  = r_bif[r_bif >= 0]
axes[2].plot(r_pos,  np.sqrt(r_pos),  color='steelblue', lw=2.5, label='Stable ($x^*=+\\sqrt{r}$)')
axes[2].plot(r_pos, -np.sqrt(r_pos),  color='crimson',   lw=2.5, ls='--', label='Unstable ($x^*=-\\sqrt{r}$)')
axes[2].plot(0, 0, 'ko', markersize=8, zorder=5, label='Bifurcation point')
axes[2].set_xlabel('Parameter $r$'); axes[2].set_ylabel('Equilibrium $x^*$')
axes[2].set_title('Bifurcation diagram')
axes[2].legend(fontsize=8); axes[2].axvline(0, color='k', lw=0.5, ls=':')

plt.suptitle('Saddle-Node Bifurcation: $x\'=r-x^2$', fontsize=12)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# f(x) plots
x_arr = np.linspace(-1.5, 3.0, 400)
for r_val, color, lbl in zip([-0.8, 0.0, 1.2],
                              ['crimson', 'darkorange', 'steelblue'],
                              ['$r=-0.8$', '$r=0$', '$r=1.2$']):
    axes[0].plot(x_arr, r_val*x_arr - x_arr**2, color=color, lw=2, label=lbl)
axes[0].axhline(0, color='k', lw=0.8)
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$f(x)=rx-x^2$')
axes[0].set_title('$f(x)$ for several $r$')
axes[0].legend(fontsize=9); axes[0].set_ylim(-2, 1)

# Bifurcation diagram
r_bif = np.linspace(-1.5, 2.5, 400)
# x*=0: stable for r<0 (solid), unstable for r>0 (dashed)
axes[1].plot(r_bif[r_bif <= 0], np.zeros(np.sum(r_bif<=0)),
             color='steelblue', lw=2.5, label='$x^*=0$ stable')
axes[1].plot(r_bif[r_bif >= 0], np.zeros(np.sum(r_bif>=0)),
             color='steelblue', lw=2.5, ls='--', label='$x^*=0$ unstable')
# x*=r: unstable for r<0 (dashed), stable for r>0 (solid)
axes[1].plot(r_bif[r_bif <= 0], r_bif[r_bif <= 0],
             color='crimson', lw=2.5, ls='--', label='$x^*=r$ unstable')
axes[1].plot(r_bif[r_bif >= 0], r_bif[r_bif >= 0],
             color='crimson', lw=2.5, label='$x^*=r$ stable')
axes[1].plot(0, 0, 'ko', markersize=8, zorder=5, label='Bifurcation point')
axes[1].set_xlabel('Parameter $r$'); axes[1].set_ylabel('Equilibrium $x^*$')
axes[1].set_title('Bifurcation diagram')
axes[1].legend(fontsize=8, ncol=2)
axes[1].axvline(0, color='k', lw=0.5, ls=':')

plt.suptitle(r"Transcritical Bifurcation: $x'=rx-x^2$", fontsize=12)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Bifurcation diagram
r_bif = np.linspace(-1.5, 3.0, 400)
# x*=0: stable for r<0 (solid), unstable for r>0 (dashed)
axes[0].plot(r_bif[r_bif <= 0], np.zeros(np.sum(r_bif <= 0)),
             color='steelblue', lw=2.5, label='$x^*=0$ (stable)')
axes[0].plot(r_bif[r_bif >= 0], np.zeros(np.sum(r_bif >= 0)),
             color='steelblue', lw=2.5, ls='--', label='$x^*=0$ (unstable)')
# x*=±sqrt(r) for r>0: stable
r_pos = r_bif[r_bif >= 0]
axes[0].plot(r_pos,  np.sqrt(r_pos), color='crimson', lw=2.5, label=r'$x^*=\pm\sqrt{r}$ (stable)')
axes[0].plot(r_pos, -np.sqrt(r_pos), color='crimson', lw=2.5)
axes[0].plot(0, 0, 'ko', markersize=8, zorder=5, label='Bifurcation point $r=0$')
axes[0].set_xlabel('Parameter $r$'); axes[0].set_ylabel('Equilibrium $x^*$')
axes[0].set_title('Bifurcation diagram')
axes[0].legend(fontsize=8); axes[0].axvline(0, color='k', lw=0.5, ls=':')

# Solution curves for r=1.5
r_fixed = 1.5
xe_stable = np.sqrt(r_fixed)
f_pf = lambda t, x: [r_fixed*x[0] - x[0]**3]
t_eval = np.linspace(0, 6, 400)
colors_pf = plt.cm.RdYlBu(np.linspace(0.05, 0.95, 8))
for x0, color in zip([-2.5,-1.5,-0.5, 0.3, 0.5, 1.5, 2.5, -0.1],
                      colors_pf):
    sol = solve_ivp(f_pf, (0,6), [x0], t_eval=t_eval, max_step=0.02)
    axes[1].plot(sol.t, sol.y[0], color=color, lw=1.8, label=f'$x_0={x0}$')
axes[1].axhline( xe_stable, color='crimson',  ls='--', lw=1.8, label=f'$x^*=+\\sqrt{{r}}={xe_stable:.2f}$')
axes[1].axhline(-xe_stable, color='crimson',  ls='--', lw=1.8, label=f'$x^*=-\\sqrt{{r}}$')
axes[1].axhline(0,          color='steelblue', ls='--', lw=1.2, label='$x^*=0$ (unstable)')
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x(t)$')
axes[1].set_title(f'Solutions for $r={r_fixed}$')
axes[1].legend(fontsize=7, ncol=2); axes[1].set_ylim(-3, 3)

plt.suptitle(r"Supercritical Pitchfork: $x'=rx-x^3$", fontsize=12)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# Saddle-node: x' = r - x^2
r = np.linspace(-0.2, 2.0, 300)
r_pos = r[r >= 0]
axes[0].plot(r_pos,  np.sqrt(r_pos), color='steelblue', lw=2.5, label='Stable')
axes[0].plot(r_pos, -np.sqrt(r_pos), color='crimson',   lw=2.5, ls='--', label='Unstable')
axes[0].plot(0, 0, 'ko', markersize=8)
axes[0].axvline(0, color='k', lw=0.5, ls=':')
axes[0].set_xlabel('$r$'); axes[0].set_ylabel('$x^*$')
axes[0].set_title("Saddle-Node\n$x'=r-x^2$")
axes[0].legend(fontsize=8)

# Transcritical: x' = rx - x^2
r2 = np.linspace(-1.5, 2.0, 300)
axes[1].plot(r2[r2<=0], np.zeros(np.sum(r2<=0)), color='steelblue', lw=2.5)
axes[1].plot(r2[r2>=0], np.zeros(np.sum(r2>=0)), color='steelblue', lw=2.5, ls='--')
axes[1].plot(r2[r2<=0], r2[r2<=0], color='crimson', lw=2.5, ls='--')
axes[1].plot(r2[r2>=0], r2[r2>=0], color='crimson', lw=2.5)
axes[1].plot(0, 0, 'ko', markersize=8)
axes[1].axvline(0, color='k', lw=0.5, ls=':')
axes[1].set_xlabel('$r$'); axes[1].set_ylabel('$x^*$')
axes[1].set_title("Transcritical\n$x'=rx-x^2$")

# Pitchfork: x' = rx - x^3
r3 = np.linspace(-1.5, 2.5, 300)
r3_pos = r3[r3 >= 0]
axes[2].plot(r3[r3<=0], np.zeros(np.sum(r3<=0)), color='steelblue', lw=2.5)
axes[2].plot(r3[r3>=0], np.zeros(np.sum(r3>=0)), color='steelblue', lw=2.5, ls='--')
axes[2].plot(r3_pos,  np.sqrt(r3_pos), color='crimson', lw=2.5)
axes[2].plot(r3_pos, -np.sqrt(r3_pos), color='crimson', lw=2.5, label='Stable branches')
axes[2].plot(0, 0, 'ko', markersize=8)
axes[2].axvline(0, color='k', lw=0.5, ls=':')
axes[2].set_xlabel('$r$'); axes[2].set_ylabel('$x^*$')
axes[2].set_title("Pitchfork (supercritical)\n$x'=rx-x^3$")
axes[2].legend(fontsize=8)

for ax in axes:
    ax.axhline(0, color='k', lw=0.5)

plt.tight_layout()
plt.show()

K_bw = 10.0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# f(x) for three r values
x_arr = np.linspace(0.01, 12, 400)
pred  = x_arr**2 / (1 + x_arr**2)
for r_val, color, lbl in zip([0.3, 0.55, 0.85],
                               ['steelblue', 'darkorange', 'crimson'],
                               ['$r=0.30$ (one eq.)', '$r=0.55$ (bistable)', '$r=0.85$ (outbreak)']):
    growth = r_val * x_arr * (1 - x_arr/K_bw)
    axes[0].plot(x_arr, growth - pred, color=color, lw=2, label=lbl)
axes[0].axhline(0, color='k', lw=0.8)
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$f(x)$')
axes[0].set_title('Budworm $f(x)$ for three values of $r$')
axes[0].legend(fontsize=8); axes[0].set_ylim(-1.5, 1.5)

# Solutions for bistable case r=0.55
r_bw  = 0.55
f_bw  = lambda t, x: [r_bw*x[0]*(1-x[0]/K_bw) - x[0]**2/(1+x[0]**2)]
t_eval = np.linspace(0, 40, 600)
for x0, color in zip([0.3, 0.8, 1.5, 3.0, 5.0, 7.0, 9.0, 11.0],
                     plt.cm.viridis(np.linspace(0.1, 0.9, 8))):
    sol = solve_ivp(f_bw, (0, 40), [x0], t_eval=t_eval, max_step=0.05)
    axes[1].plot(sol.t, sol.y[0], color=color, lw=1.8, label=f'$x_0={x0}$')
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x(t)$')
axes[1].set_title(f'Bistable solutions ($r={r_bw}$, $K={K_bw}$)')
axes[1].legend(fontsize=7, ncol=2)

plt.suptitle("Spruce Budworm Model", fontsize=12)
plt.tight_layout()
plt.show()