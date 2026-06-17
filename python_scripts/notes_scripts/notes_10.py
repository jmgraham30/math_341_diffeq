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
#| label: fig-eigenpair-verify
#| fig-cap: "SymPy verification of the eigenvalues and eigenvectors for Example 4.27. The characteristic polynomial, eigenvalues, and eigenvectors all match the hand calculations."

s = sym.Symbol('lam')
A427 = sym.Matrix([[1, 1], [4, 1]])
char_poly = A427.charpoly(s)
print("Characteristic polynomial:", char_poly.as_expr())
print("Eigenvalues:", sym.solve(char_poly.as_expr(), s))
print("\nEigenvectors (SymPy):")
for val, mult, vecs in A427.eigenvects():
    for v in vecs:
        display(Math(rf"\lambda = {val},\quad \mathbf{{v}} = {sym.latex(v)}"))


#| code-fold: true
#| code-summary: "Show the code"

A428 = sym.Matrix([[-2, -3], [3, -2]])
print("Eigenvalues of Example 4.28:")
for val, mult, vecs in A428.eigenvects():
    display(Math(rf"\lambda = {sym.latex(val)},\quad \mathbf{{v}} = {sym.latex(vecs[0])}"))


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-saddle
#| fig-cap: "Phase plane portraits for two saddle-point systems. Left: Example 4.29 — the decoupled system with eigenvectors along the coordinate axes; orbits are hyperbolas. Right: Example 4.30 — eigenvectors along $y=-2x$ and $y=2x$; separatrices (linear orbits) are shown in red and blue."

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

def plot_phase(ax, A, title, lim=2.5, n_ic=12):
    x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 22), np.linspace(-lim, lim, 22))
    dx = A[0,0]*x_g + A[0,1]*y_g
    dy = A[1,0]*x_g + A[1,1]*y_g
    nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
    ax.quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.35, color='gray', scale=28)
    evals, evecs = np.linalg.eig(A)
    colors_ev = ['steelblue', 'crimson']
    for i, (lam, color) in enumerate(zip(evals, colors_ev)):
        v = evecs[:, i].real
        v = v / np.linalg.norm(v)
        for s in [1, -1]:
            ax.annotate('', xy=(s*lim*0.85*v[0], s*lim*0.85*v[1]),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.plot([0], [0], color=color, lw=2,
                label=rf'$\lambda={lam:.1f}$, $\mathbf{{v}}=({v[0]:.2f},{v[1]:.2f})$')
    np.random.seed(42)
    for _ in range(n_ic):
        r = lim * (0.4 + 0.6*np.random.rand())
        theta = np.random.uniform(0, 2*np.pi)
        x0 = [r*np.cos(theta), r*np.sin(theta)]
        for sign in [1]:
            sol = solve_ivp(lambda t, y: A @ y, (-3, 3), x0,
                           dense_output=True, max_step=0.05)
            t_f = sol.t; xy_f = sol.y
            mask = np.all(np.abs(xy_f) < lim*1.5, axis=0)
            ax.plot(xy_f[0, mask], xy_f[1, mask], 'k-', lw=0.8, alpha=0.55)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.plot(0, 0, 'ko', markersize=7, zorder=5)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title(title); ax.legend(fontsize=8)

A429 = np.array([[-1., 0.], [0., 1.]])
A430 = np.array([[1., 1.], [4., 1.]])
plot_phase(axes[0], A429, "Example 4.29: Saddle ($\\lambda=-1,1$)")
plot_phase(axes[1], A430, "Example 4.30: Saddle ($\\lambda=-1,3$)")
plt.suptitle("Saddle Points — Opposite-Sign Eigenvalues", fontsize=12)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-nodes
#| fig-cap: "Phase plane portraits for nodal and degenerate cases. Left: asymptotically stable node (Example 4.33, $\\lambda=-1,-6$) — all orbits enter the origin tangent to the slow eigenvector $\\mathbf{v}_1=(2,1)^T$. Right: $\\det A=0$ case (Example 4.36) — line of equilibria along $x+2y=0$ with orbits exiting along parallel lines."

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Stable node: Example 4.33
A433 = np.array([[-2., 2.], [2., -5.]])
lim = 3.0
x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 22), np.linspace(-lim, lim, 22))
dx = A433[0,0]*x_g + A433[0,1]*y_g
dy = A433[1,0]*x_g + A433[1,1]*y_g
nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
axes[0].quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.3, color='gray', scale=28)

np.random.seed(7)
for _ in range(16):
    r = lim * (0.3 + 0.7*np.random.rand())
    theta = np.random.uniform(0, 2*np.pi)
    x0 = [r*np.cos(theta), r*np.sin(theta)]
    sol = solve_ivp(lambda t, y: A433 @ y, (0, 4), x0,
                   dense_output=True, max_step=0.05)
    xy = sol.y; mask = np.all(np.abs(xy) < lim*1.4, axis=0)
    axes[0].plot(xy[0, mask], xy[1, mask], 'steelblue', lw=1.2, alpha=0.7)

# Mark eigenvectors
evals433, evecs433 = np.linalg.eig(A433)
for i, color in enumerate(['crimson', 'darkorange']):
    v = evecs433[:, i].real; v /= np.linalg.norm(v)
    lam = evals433[i].real
    for s in [1, -1]:
        axes[0].annotate('', xy=(s*lim*0.8*v[0], s*lim*0.8*v[1]),
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    axes[0].plot([], [], color=color, lw=2,
                 label=rf'$\lambda={lam:.0f}$, $\mathbf{{v}}=({v[0]:.2f},{v[1]:.2f})$')

axes[0].axhline(0, color='k', lw=0.5); axes[0].axvline(0, color='k', lw=0.5)
axes[0].plot(0, 0, 'ko', markersize=7, zorder=5)
axes[0].set_xlim(-lim, lim); axes[0].set_ylim(-lim, lim)
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
axes[0].set_title("Example 4.33: Stable Node ($\\lambda=-1,-6$)")
axes[0].legend(fontsize=8)

# det A = 0 case: Example 4.36
A436 = np.array([[1., 2.], [2., 4.]])
lim2 = 3.0
x_g2, y_g2 = np.meshgrid(np.linspace(-lim2, lim2, 22), np.linspace(-lim2, lim2, 22))
dx2 = A436[0,0]*x_g2 + A436[0,1]*y_g2
dy2 = A436[1,0]*x_g2 + A436[1,1]*y_g2
nrm2 = np.sqrt(dx2**2 + dy2**2 + 1e-10)
axes[1].quiver(x_g2, y_g2, dx2/nrm2, dy2/nrm2, alpha=0.3, color='gray', scale=28)

# Equilibrium line: x + 2y = 0 => y = -x/2
x_eq = np.linspace(-lim2, lim2, 200)
axes[1].plot(x_eq, -x_eq/2, 'k--', lw=2, label='Equil. line $x+2y=0$')

for x0_val in np.linspace(-lim2*0.9, lim2*0.9, 10):
    for y0_val in [-lim2*0.6, lim2*0.6]:
        sol = solve_ivp(lambda t, y: A436 @ y, (0, 0.4),
                       [x0_val, y0_val], dense_output=True, max_step=0.02)
        xy = sol.y; mask = np.all(np.abs(xy) < lim2*1.4, axis=0)
        axes[1].plot(xy[0, mask], xy[1, mask], 'steelblue', lw=1.2, alpha=0.7)

axes[1].axhline(0, color='k', lw=0.5); axes[1].axvline(0, color='k', lw=0.5)
axes[1].set_xlim(-lim2, lim2); axes[1].set_ylim(-lim2, lim2)
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')
axes[1].set_title(r"Example 4.36: $\det A=0$, line of equilibria")
axes[1].legend(fontsize=9)

plt.suptitle("Real Unequal Eigenvalues — Nodes and Degenerate Cases", fontsize=12)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-complex-eigs
#| fig-cap: "Phase plane portraits for complex eigenvalue cases. Left: Example 4.37 — asymptotically stable spiral with $\\lambda=-2\\pm 3i$; orbits spiral into the origin counterclockwise. Right: purely imaginary case $\\lambda=\\pm 2i$ — closed elliptical orbits (stable center)."

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

def phase_complex(ax, A, title, lim=2.5, t_span=(-0.1, 5)):
    x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 22), np.linspace(-lim, lim, 22))
    dx = A[0,0]*x_g + A[0,1]*y_g
    dy = A[1,0]*x_g + A[1,1]*y_g
    nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
    ax.quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.25, color='gray', scale=30)
    np.random.seed(5)
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for r_frac, color in zip([0.3, 0.7, 1.2], ['steelblue', 'seagreen', 'crimson']):
        for theta in angles[:4]:
            x0 = [r_frac*lim*np.cos(theta), r_frac*lim*np.sin(theta)]
            sol = solve_ivp(lambda t, y: A @ y, t_span, x0,
                           dense_output=True, max_step=0.02)
            xy = sol.y; mask = np.all(np.abs(xy) < lim*1.4, axis=0)
            ax.plot(xy[0, mask], xy[1, mask], color=color, lw=1.3, alpha=0.75)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.plot(0, 0, 'ko', markersize=7, zorder=5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title(title)

# Example 4.37: stable spiral
A437 = np.array([[-2., -3.], [3., -2.]])
phase_complex(axes[0], A437, "Example 4.37: Stable Spiral ($\\lambda=-2\\pm 3i$)",
              t_span=(0, 4))

# Center: purely imaginary eigenvalues lambda = +/- 2i
A_center = np.array([[0., -2.], [2., 0.]])
phase_complex(axes[1], A_center, "Stable Center ($\\lambda=\\pm 2i$)",
              t_span=(0, 2*np.pi + 0.1))

plt.suptitle("Complex Eigenvalues — Spirals and Centers", fontsize=12)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-equal-eigs
#| fig-cap: "Phase plane portraits for equal eigenvalue cases. Left: Example 4.38 — non-deficient stable star node ($\\lambda=-2,-2$); every ray through the origin is a linear orbit. Right: Example 4.39 — deficient unstable node ($\\lambda=3,3$); the single linear orbit is along $y=x$, and all other orbits curve away."

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Example 4.38: star node
A438 = np.array([[-2., 0.], [0., -2.]])
lim = 2.0
x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 22), np.linspace(-lim, lim, 22))
for ax, A, title, t_span in [
    (axes[0], A438, "Example 4.38: Stable Star Node ($\\lambda=-2,-2$)", (0, 1.2)),
    (axes[1], np.array([[2., 1.], [-1., 4.]]),
     "Example 4.39: Unstable Deficient Node ($\\lambda=3,3$)", (-0.8, 0))]:

    dx_q = A[0,0]*x_g + A[0,1]*y_g
    dy_q = A[1,0]*x_g + A[1,1]*y_g
    nrm_q = np.sqrt(dx_q**2 + dy_q**2 + 1e-10)
    ax.quiver(x_g, y_g, dx_q/nrm_q, dy_q/nrm_q, alpha=0.25, color='gray', scale=30)

    np.random.seed(99)
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    for theta in angles:
        x0 = [1.5*np.cos(theta), 1.5*np.sin(theta)]
        sol = solve_ivp(lambda t, y, M=A: M @ y, t_span, x0,
                       dense_output=True, max_step=0.02)
        xy = sol.y; mask = np.all(np.abs(xy) < lim*1.4, axis=0)
        ax.plot(xy[0, mask], xy[1, mask], 'steelblue', lw=1.3, alpha=0.75)

    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.plot(0, 0, 'ko', markersize=7, zorder=5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title(title)

# Highlight the eigenvector direction for deficient case
ax = axes[1]
v = np.array([1., 1.]) / np.sqrt(2)
for s in [1, -1]:
    ax.annotate('', xy=(s*lim*0.8*v[0], s*lim*0.8*v[1]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=2.5))
ax.plot([], [], 'crimson', lw=2, label='Linear orbit $y=x$')
ax.legend(fontsize=9)

plt.suptitle("Real Equal Eigenvalues — Star and Degenerate Nodes", fontsize=12)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ex439-ivp
#| fig-cap: "Solution of the IVP from Example 4.39 with $\\mathbf{x}(0)=(1,0)^T$. Both components grow without bound as $t\\to+\\infty$, confirming the unstable node. The numerical `solve_ivp` solution (dashed) exactly matches the analytical formula $x(t)=(1-t)e^{3t}$, $y(t)=-te^{3t}$."

t_plot = np.linspace(-0.8, 0.4, 400)
x_anal = (1 - t_plot) * np.exp(3*t_plot)
y_anal = -t_plot * np.exp(3*t_plot)

A439 = np.array([[2., 1.], [-1., 4.]])
sol = solve_ivp(lambda t, y: A439 @ y, (-0.8, 0.4), [1., 0.],
               dense_output=True, max_step=0.01)
t_dots = np.linspace(-0.8, 0.4, 20)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_plot, x_anal, color='steelblue', lw=2.5, label=r'$x(t)=(1-t)e^{3t}$')
ax.plot(t_plot, y_anal, color='crimson',   lw=2.5, label=r'$y(t)=-te^{3t}$')
ax.plot(t_dots, sol.sol(t_dots)[0], 'o', color='steelblue', markersize=5, label='Numerical $x(t)$')
ax.plot(t_dots, sol.sol(t_dots)[1], 's', color='crimson',   markersize=5, label='Numerical $y(t)$')
ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5, ls='--', alpha=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('Solution components')
ax.set_title(r"Example 4.39: IVP solution $\mathbf{x}(0)=(1,0)^T$, $\lambda=3$ (deficient)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
