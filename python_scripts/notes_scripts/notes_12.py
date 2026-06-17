#| code-fold: true
#| code-summary: "Show the code"

import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ex51
#| fig-cap: "Phase diagram for Example 5.1: $x'=-x+xy$, $y'=-4y+8xy$. The $x$-nullclines (steelblue dashed: $x=0$ and $y=1$) and $y$-nullclines (crimson dashed: $y=0$ and $x=1/2$) divide the plane into eight regions. The origin is an asymptotically stable node ($\\lambda=-1,-4$); the point $(1/2,1)$ is a saddle ($\\lambda=\\pm 2$). Orbits were computed using `solve_ivp`."

def f51(x, y): return -x + x*y
def g51(x, y): return -4*y + 8*x*y

lim = 2.0
x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 24), np.linspace(-lim, lim, 24))
dx = f51(x_g, y_g); dy = g51(x_g, y_g)
nrm = np.sqrt(dx**2 + dy**2 + 1e-10)

fig, ax = plt.subplots(figsize=(8, 7))
ax.quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.3, color='gray', scale=28)

# Nullclines
ax.axvline(0,  color='steelblue', ls='--', lw=1.8, label=r"$x$-nullcline: $x=0$")
ax.axhline(1,  color='steelblue', ls='--', lw=1.8, label=r"$x$-nullcline: $y=1$")
ax.axhline(0,  color='crimson',   ls='--', lw=1.8, label=r"$y$-nullcline: $y=0$")
ax.axvline(0.5,color='crimson',   ls='--', lw=1.8, label=r"$y$-nullcline: $x=1/2$")

# Orbits
np.random.seed(7)
ics = [(1.5, 1.5), (1.5, 0.5), (1.5, -0.5), (1.5, -1.5),
       (-1.5, 1.5), (-1.5, 0.5), (-1.5, -0.5), (-1.5, -1.5),
       (0.3, 1.5), (0.8, 1.5), (-0.3, 0.5), (-0.8, 0.5),
       (0.3, -0.5), (0.8, -0.5), (0.1, 0.1), (-0.1, 0.1)]
for x0, y0 in ics:
    sol = solve_ivp(lambda t, z: [f51(z[0], z[1]), g51(z[0], z[1])],
                    (0, 6), [x0, y0], dense_output=True, max_step=0.02)
    xy = sol.y; mask = np.all(np.abs(xy) < lim*1.35, axis=0)
    ax.plot(xy[0, mask], xy[1, mask], 'k-', lw=1.0, alpha=0.55)

# Critical points
ax.plot(0, 0, 'o', color='steelblue', markersize=9, zorder=5, label='Stable node $(0,0)$')
ax.plot(0.5, 1, 's', color='crimson', markersize=9, zorder=5, label='Saddle $(1/2,1)$')

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title(r"Example 5.1: $x'=-x+xy$, $y'=-4y+8xy$", fontsize=11)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ex52
#| fig-cap: "Phase diagram for Example 5.2: $x'=x-x^3$, $y'=2y$. Vertical nullclines at $x=0,\\pm 1$ (steelblue dashed) and horizontal nullcline $y=0$ (crimson dashed). The origin is an unstable node; $(\\pm 1, 0)$ are saddle points. In the upper half-plane orbits move upward; in the lower half-plane downward."

def f52(x, y): return x - x**3
def g52(x, y): return 2*y

lim = 2.0
x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 24), np.linspace(-lim, lim, 24))
dx = f52(x_g, y_g); dy = g52(x_g, y_g)
nrm = np.sqrt(dx**2 + dy**2 + 1e-10)

fig, ax = plt.subplots(figsize=(8, 7))
ax.quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.3, color='gray', scale=28)

# Nullclines
for xv in [-1, 0, 1]:
    lbl = r"$x$-nullcline" if xv == -1 else None
    ax.axvline(xv, color='steelblue', ls='--', lw=1.8, label=lbl)
ax.axhline(0, color='crimson', ls='--', lw=1.8, label=r"$y$-nullcline: $y=0$")

# Orbits
ics52 = [(1.5, 0.1), (1.5, -0.1), (-1.5, 0.1), (-1.5, -0.1),
         (0.5, 0.1), (0.5, -0.1), (-0.5, 0.1), (-0.5, -0.1),
         (0.05, 1.5), (0.05, -1.5), (-0.05, 1.5), (-0.05, -1.5),
         (1.5, 1.0), (-1.5, 1.0), (1.5, -1.0), (-1.5, -1.0)]
for x0, y0 in ics52:
    sol = solve_ivp(lambda t, z: [f52(z[0], z[1]), g52(z[0], z[1])],
                    (0, 1.5), [x0, y0], dense_output=True, max_step=0.02)
    xy = sol.y; mask = np.all(np.abs(xy) < lim*1.35, axis=0)
    ax.plot(xy[0, mask], xy[1, mask], 'k-', lw=1.0, alpha=0.55)

# Critical points
ax.plot(0,  0, 'o', color='darkorange', markersize=9, zorder=5, label='Unstable node $(0,0)$')
ax.plot(1,  0, 's', color='crimson',    markersize=9, zorder=5, label='Saddle $(\\pm 1,0)$')
ax.plot(-1, 0, 's', color='crimson',    markersize=9, zorder=5)

ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title(r"Example 5.2: $x'=x-x^3$, $y'=2y$", fontsize=11)
ax.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ex53
#| fig-cap: "Phase diagram for Example 5.3: $x'=y^2$, $y'=-\\frac{2}{3}x$. Orbits are the curves $y=(C-x^2)^{1/3}$ for various values of $C$ (colored curves). The critical point at the origin has $\\det J=0$, so linearization fails. Since $x'=y^2\\geq 0$, all orbits move to the right (arrows shown). Numerical `solve_ivp` orbits (black) confirm the analytical family."

fig, ax = plt.subplots(figsize=(8, 6.5))

# Analytical orbit family
x_plot = np.linspace(-1.0, 1.0, 400)
colors_c = plt.cm.viridis(np.linspace(0.1, 0.9, 9))
for C, color in zip(np.linspace(-0.8, 0.8, 9), colors_c):
    arg = C - x_plot**2
    valid = arg > -1e-8   # cube root is real for all real arg, but avoid domain issues
    y_curve = np.cbrt(C - x_plot**2)
    ax.plot(x_plot[valid], y_curve[valid], color=color, lw=1.8, alpha=0.75)

# Numerical orbits to confirm and show direction
ics53 = [(0.0, 0.7), (0.0, -0.7), (0.0, 0.4), (0.0, -0.4),
         (-0.8, 0.0), (-0.8, 0.3), (-0.8, -0.3), (0.5, 0.6)]
for x0, y0 in ics53:
    sol = solve_ivp(lambda t, z: [z[1]**2, -2/3*z[0]],
                    (0, 3), [x0, y0], dense_output=True, max_step=0.02)
    xy = sol.y; mask = np.all(np.abs(xy) < 1.1, axis=0)
    ax.plot(xy[0, mask], xy[1, mask], 'k-', lw=1.1, alpha=0.55)
    if mask.sum() > 5:
        mid = mask.sum() // 2
        ax.annotate('', xy=(xy[0, mask][mid+2], xy[1, mask][mid+2]),
                    xytext=(xy[0, mask][mid], xy[1, mask][mid]),
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.2))

# Nullclines: x-nullcline is y=0 (x'=y^2=0), y-nullcline is x=0
ax.axhline(0, color='steelblue', ls='--', lw=1.8, label=r"$x$-nullcline: $y=0$")
ax.axvline(0, color='crimson',   ls='--', lw=1.8, label=r"$y$-nullcline: $x=0$")

ax.plot(0, 0, 'ko', markersize=9, zorder=5, label=r'Critical point $(0,0)$, $\det J=0$')
ax.set_xlim(-1.0, 1.0); ax.set_ylim(-1.0, 1.0)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title(r"Example 5.3: $x'=y^2$, $y'=-\frac{2}{3}x$ — degenerate critical point", fontsize=11)
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
