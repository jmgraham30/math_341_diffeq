#| code-fold: true
#| code-summary: "Show the code"

import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-trace-det
#| fig-cap: "The trace–determinant plane (Figure 4.13 in Logan). Each point $(\\text{tr}\\,A, \\det A)$ corresponds to a unique phase portrait type. The parabola $(\\text{tr}\\,A)^2 = 4\\det A$ separates nodes (below or on) from spirals (above). Along $\\det A = 0$ one eigenvalue is zero and there is a line of non-isolated equilibria. The shaded half-plane $\\text{tr}\\,A<0$, $\\det A>0$ is the asymptotically stable region."

tr_vals = np.linspace(-3.5, 3.5, 500)
det_parabola = tr_vals**2 / 4   # 4 det A = (tr A)^2

fig, ax = plt.subplots(figsize=(9, 7))

# Shaded stable region: tr < 0, det > 0
tr_neg = tr_vals[tr_vals <= 0]
ax.fill_betweenx(np.linspace(0, 4, 200),
                 -3.5, 0,
                 alpha=0.08, color='steelblue', label='Asymp. stable region')

# Parabola
ax.plot(tr_vals, det_parabola, 'k-', lw=2.0, label=r'$({\rm tr}\,A)^2 - 4\det A = 0$')

# Axes
ax.axhline(0, color='k', lw=1.2)
ax.axvline(0, color='k', lw=1.2)

# Region labels
label_kwargs = dict(fontsize=11, ha='center', va='center')
ax.text(-2.2, 2.8, 'Stable\nSpirals', color='steelblue', **label_kwargs)
ax.text( 2.2, 2.8, 'Unstable\nSpirals', color='crimson', **label_kwargs)
ax.text(-2.4, 0.5, 'Stable\nNodes', color='steelblue', fontsize=10, ha='center', va='center')
ax.text( 2.4, 0.5, 'Unstable\nNodes', color='crimson', fontsize=10, ha='center', va='center')
ax.text( 0.0, 3.5, 'Centers', color='seagreen', fontsize=11, ha='center', va='center')
ax.text( 0.0, -1.3, 'Saddle Points (Unstable)', color='darkorange', fontsize=11,
         ha='center', va='center')
ax.text( 0.0, -0.4, r'$\det A = 0$: line of equilibria', color='gray',
         fontsize=9, ha='center', va='center')

# Center dot on tr-axis
ax.plot(0, 0, 'ko', markersize=6, zorder=5)

# Arrows indicating the parabola boundary
ax.annotate(r'$({\rm tr}\,A)^2 - 4\det A = 0$', xy=(1.6, 0.64),
            xytext=(2.5, 1.8), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='k', lw=1.2))

ax.set_xlabel(r'${\rm tr}\,A$', fontsize=13)
ax.set_ylabel(r'$\det A$', fontsize=13)
ax.set_title('Trace–Determinant Plane: Classification of Phase Portraits\n'
             r'($\mathbf{x}\'=A\mathbf{x}$, $\det A\neq 0$)', fontsize=12)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-2.0, 4.0)
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-crop-soil
#| fig-cap: "Crop–soil pesticide model with $\\alpha=0.5$, $\\beta=0.3$, $\\gamma=0.2$. Left: phase portrait in the first quadrant showing nullclines (dashed) and several orbits all converging to the origin — the pesticide eventually disappears. Right: time series for a sample orbit confirming that both $x(t)$ and $y(t)$ decay to zero."

alpha, beta, gamma = 0.5, 0.3, 0.2
A_cs = np.array([[-beta, alpha], [beta, -(alpha + gamma)]])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Phase portrait (first quadrant)
lim = 3.0
x_g, y_g = np.meshgrid(np.linspace(0, lim, 20), np.linspace(0, lim, 20))
dx = A_cs[0,0]*x_g + A_cs[0,1]*y_g
dy = A_cs[1,0]*x_g + A_cs[1,1]*y_g
nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
axes[0].quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.3, color='gray', scale=28)

# Nullclines
x_nc = np.linspace(0, lim, 200)
axes[0].plot(x_nc, (beta/alpha)*x_nc, 'steelblue', ls='--', lw=1.8,
             label=r"$x'=0$: $y=\frac{\beta}{\alpha}x$")
axes[0].plot(x_nc, (beta/(alpha+gamma))*x_nc, 'crimson', ls='--', lw=1.8,
             label=r"$y'=0$: $y=\frac{\beta}{\alpha+\gamma}x$")

# Orbits
np.random.seed(3)
for x0 in [(2.5, 0.3), (0.3, 2.5), (2.0, 2.0), (1.5, 0.1), (0.2, 1.8)]:
    sol = solve_ivp(lambda t, y: A_cs @ y, (0, 15), list(x0),
                    dense_output=True, max_step=0.05)
    xy = sol.y
    mask = (xy[0] >= -0.05) & (xy[1] >= -0.05)
    axes[0].plot(xy[0, mask], xy[1, mask], 'k-', lw=1.2, alpha=0.6)

axes[0].plot(0, 0, 'ko', markersize=8, zorder=5, label='Origin (stable)')
axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
axes[0].set_xlabel('$x$ (crop)'); axes[0].set_ylabel('$y$ (soil)')
axes[0].set_title('Phase portrait — first quadrant\n'
                  r'($\alpha=0.5,\,\beta=0.3,\,\gamma=0.2$)')
axes[0].legend(fontsize=8)

# Time series
t_plot = np.linspace(0, 20, 400)
sol_ts = solve_ivp(lambda t, y: A_cs @ y, (0, 20), [2.0, 0.5],
                   dense_output=True, max_step=0.05)
axes[1].plot(t_plot, sol_ts.sol(t_plot)[0], color='steelblue', lw=2.5,
             label='$x(t)$ (crop)')
axes[1].plot(t_plot, sol_ts.sol(t_plot)[1], color='crimson', lw=2.5,
             label='$y(t)$ (soil)')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('Concentration')
axes[1].set_title('Time series: $x(0)=2,\\ y(0)=0.5$')
axes[1].legend(fontsize=9)

plt.suptitle("Example 4.42: Crop–Soil Pesticide Model", fontsize=12)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ex443
#| fig-cap: "Example 4.43: saddle point with $\\det A=-50<0$. Nullclines are dashed (steelblue: $x'=0$, crimson: $y'=0$). Separatrices along eigenvectors are shown in bold. The direction field in each of the four nullcline regions is also visible."

A443 = np.array([[-7., 6.], [6., 2.]])
A443_sym = sym.Matrix([[-7, 6], [6, 2]])
print("Eigenvalues of Example 4.43:")
for val, mult, vecs in A443_sym.eigenvects():
    display(Math(rf"\lambda={sym.latex(val)},\quad \mathbf{{v}}={sym.latex(vecs[0])}"))

lim = 3.0
x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 22), np.linspace(-lim, lim, 22))
dx = A443[0,0]*x_g + A443[0,1]*y_g
dy = A443[1,0]*x_g + A443[1,1]*y_g
nrm = np.sqrt(dx**2 + dy**2 + 1e-10)

fig, ax = plt.subplots(figsize=(8, 6.5))
ax.quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.3, color='gray', scale=28)

# Nullclines
x_nc = np.linspace(-lim, lim, 300)
ax.plot(x_nc, (7/6)*x_nc, 'steelblue', ls='--', lw=1.8, label=r"$x'=0$: $y=\frac{7}{6}x$")
ax.plot(x_nc, -3*x_nc, 'crimson', ls='--', lw=1.8, label=r"$y'=0$: $y=-3x$")

# Orbits
np.random.seed(17)
for _ in range(12):
    r = lim*(0.4 + 0.5*np.random.rand())
    theta = np.random.uniform(0, 2*np.pi)
    x0 = [r*np.cos(theta), r*np.sin(theta)]
    sol = solve_ivp(lambda t, y: A443 @ y, (-0.6, 0.6), x0,
                    dense_output=True, max_step=0.02)
    xy = sol.y; mask = np.all(np.abs(xy) < lim*1.4, axis=0)
    ax.plot(xy[0, mask], xy[1, mask], 'k-', lw=0.9, alpha=0.5)

# Separatrices
evals443, evecs443 = np.linalg.eig(A443)
for i, (color, lbl) in enumerate(zip(['seagreen', 'darkorange'],
                                      [r'$\lambda=-3$ (entering)', r'$\lambda=8$ (exiting)'])):
    v = evecs443[:, i].real; v /= np.linalg.norm(v)
    for s in [1, -1]:
        ax.annotate('', xy=(s*lim*0.82*v[0], s*lim*0.82*v[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax.plot([], [], color=color, lw=2.5, label=lbl)

ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
ax.plot(0, 0, 'ko', markersize=7, zorder=5)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title("Example 4.43: Saddle Point ($\\det A < 0$)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-glucose-insulin
#| fig-cap: "Glucose–insulin dynamics (Example 4.44). Left: phase portrait with nullclines (dashed). Right: time series starting at $(x,y)=(3,0)$ — the representative parameter values give a stable spiral (complex eigenvalues with negative real part), indicating a slight hypoglycemic overshoot before returning to equilibrium."

g_val, r_val, s_val, d_val = 2.9, 4.3, 0.21, 0.78
A444 = np.array([[-g_val, -r_val], [s_val, -d_val]])

disc = (g_val - d_val)**2 - 4*r_val*s_val
print(f"(g-d)^2 - 4rs = {disc:.4f}")
evals = np.linalg.eigvals(A444)
print(f"Eigenvalues: {evals[0]:.4f}, {evals[1]:.4f}")
print("Case:", "stable node" if disc > 0 else "stable spiral")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

lim = 4.0
x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 22), np.linspace(-lim, lim, 22))
dx = A444[0,0]*x_g + A444[0,1]*y_g
dy = A444[1,0]*x_g + A444[1,1]*y_g
nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
axes[0].quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.25, color='gray', scale=28)

# Nullclines
x_nc = np.linspace(-lim, lim, 300)
axes[0].plot(x_nc, -(g_val/r_val)*x_nc, 'steelblue', ls='--', lw=1.8,
             label=r"$x'=0$: $y=-(g/r)x$")
axes[0].plot(x_nc, (s_val/d_val)*x_nc, 'crimson', ls='--', lw=1.8,
             label=r"$y'=0$: $y=(s/d)x$")

# Orbits
for r_frac, color in zip([0.5, 1.0, 1.8, 2.8],
                          ['steelblue', 'seagreen', 'crimson', 'darkorange']):
    for theta in np.linspace(0, 2*np.pi, 5, endpoint=False):
        x0 = [r_frac*np.cos(theta), r_frac*np.sin(theta)]
        sol = solve_ivp(lambda t, y: A444 @ y, (0, 8), x0,
                        dense_output=True, max_step=0.05)
        xy = sol.y; mask = np.all(np.abs(xy) < lim*1.4, axis=0)
        axes[0].plot(xy[0, mask], xy[1, mask], color=color, lw=1.1, alpha=0.65)

axes[0].axhline(0, color='k', lw=0.5); axes[0].axvline(0, color='k', lw=0.5)
axes[0].plot(0, 0, 'ko', markersize=7, zorder=5)
axes[0].set_xlim(-lim, lim); axes[0].set_ylim(-lim, lim)
axes[0].set_xlabel('$x$ (glucose)'); axes[0].set_ylabel('$y$ (insulin)')
axes[0].set_title('Phase portrait (stable spiral)')
axes[0].legend(fontsize=8)

# Time series starting at (3, 0)
t_plot = np.linspace(0, 12, 600)
sol_ts = solve_ivp(lambda t, y: A444 @ y, (0, 12), [3.0, 0.0],
                   dense_output=True, max_step=0.05)
axes[1].plot(t_plot, sol_ts.sol(t_plot)[0], color='steelblue', lw=2.5,
             label='$x(t)$ (glucose)')
axes[1].plot(t_plot, sol_ts.sol(t_plot)[1], color='crimson', lw=2.5,
             label='$y(t)$ (insulin)')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('Excess concentration')
axes[1].set_title('Time series: $x(0)=3,\\ y(0)=0$')
axes[1].legend(fontsize=9)
axes[1].set_xlim(0, 12)

plt.suptitle("Example 4.44: Glucose–Insulin Interaction", fontsize=12)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-circuit
#| fig-cap: "Example 4.45: Two-loop circuit phase portraits. Left: stable node (overdamped, $L/(R^2C)>4$, parameters $R=1,C=1,L=5$). Right: stable spiral (underdamped, $L/(R^2C)<4$, parameters $R=2,C=1,L=1$). In both cases $\\text{tr}\\,A<0$ and $\\det A>0$."

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

params = [
    (1.0, 1.0, 5.0, "Stable Node ($L/R^2C=5>4$, overdamped)"),
    (2.0, 1.0, 1.0, "Stable Spiral ($L/R^2C=0.25<4$, underdamped)"),
]
t_spans = [(0, 5), (0, 8)]

for ax, (R, C, L, title), t_span in zip(axes, params, t_spans):
    A_circ = np.array([[-1/(R*C), 1/(R*C)], [-R/L, 0]])
    ratio = L/(R**2 * C)
    print(f"R={R}, C={C}, L={L}: L/(R^2 C)={ratio:.2f}, "
          f"tr={np.trace(A_circ):.3f}, det={np.linalg.det(A_circ):.3f}")

    lim = 2.5
    x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 22), np.linspace(-lim, lim, 22))
    dx = A_circ[0,0]*x_g + A_circ[0,1]*y_g
    dy = A_circ[1,0]*x_g + A_circ[1,1]*y_g
    nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
    ax.quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.25, color='gray', scale=28)

    for r_frac, color in zip([0.4, 0.9, 1.5, 2.1],
                              ['steelblue', 'seagreen', 'crimson', 'darkorange']):
        for theta in np.linspace(0, 2*np.pi, 5, endpoint=False):
            x0 = [r_frac*np.cos(theta), r_frac*np.sin(theta)]
            sol = solve_ivp(lambda t, y, M=A_circ: M @ y, t_span, x0,
                            dense_output=True, max_step=0.05)
            xy = sol.y; mask = np.all(np.abs(xy) < lim*1.4, axis=0)
            ax.plot(xy[0, mask], xy[1, mask], color=color, lw=1.2, alpha=0.7)

    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.plot(0, 0, 'ko', markersize=7, zorder=5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('$I_2$'); ax.set_ylabel('$I_3$')
    ax.set_title(title, fontsize=10)

plt.suptitle("Example 4.45: Two-Loop Electrical Circuit", fontsize=12)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ex447
#| fig-cap: "Example 4.47: variation of parameters. The general solution is confirmed numerically (red dots) against the analytical formula (blue curves) for the particular solution components with $k_1=k_2=0$: $x_p(t)=t+4/3$ and $y_p(t)=-4t/3-13/9$."

t_sym, s_sym = sym.symbols('t s')
# Verify by substitution
x_p = sym.Matrix([t_sym + sym.Rational(4,3),
                  -sym.Rational(4,3)*t_sym - sym.Rational(13,9)])
A447 = sym.Matrix([[4,3],[-1,0]])
f447 = sym.Matrix([0, t_sym])
lhs = x_p.diff(t_sym)
rhs = A447 * x_p + f447
print("Verification: x_p' - A x_p - f =", sym.simplify(lhs - rhs).T)

t_plot = np.linspace(0, 2, 300)
xp = t_plot + 4/3
yp = -4*t_plot/3 - 13/9

# Numerical check (k1=k2=0 means starting at x_p(0))
A447_np = np.array([[4., 3.], [-1., 0.]])
x0_num = [4/3, -13/9]
f_func = lambda t: np.array([0., t])
sol_num = solve_ivp(lambda t, y: A447_np @ y + f_func(t),
                    (0, 2), x0_num, dense_output=True, max_step=0.01)
t_dots = np.linspace(0, 2, 15)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for i, (ax, comp, lbl, color) in enumerate(zip(axes,
        [xp, yp],
        [r'$x_p(t)=t+\frac{4}{3}$', r'$y_p(t)=-\frac{4}{3}t-\frac{13}{9}$'],
        ['steelblue', 'crimson'])):
    ax.plot(t_plot, comp, color=color, lw=2.5, label=lbl)
    ax.plot(t_dots, sol_num.sol(t_dots)[i], 'ro', markersize=6, label='Numerical')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('$t$'); ax.set_ylabel('Value')
    ax.set_title(f'Particular solution component {i+1}')
    ax.legend(fontsize=9)

plt.suptitle("Example 4.47: Variation of Parameters — Particular Solution Check",
             fontsize=11)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-undetermined
#| fig-cap: "Example 4.49: undetermined coefficients. Both components of the particular solution found by undetermined coefficients (solid lines) agree exactly with the variation of parameters result (red dots). The calculation is considerably shorter."

t_plot = np.linspace(0, 2, 300)
xp = t_plot + 4/3
yp = -4*t_plot/3 - 13/9

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, y_anal, vop, lbl, color in zip(
        axes,
        [xp, yp],
        [sol_num.sol(t_dots)[0], sol_num.sol(t_dots)[1]],
        [r'$x_p(t)=t+\frac{4}{3}$ (undetermined coeff.)',
         r'$y_p(t)=-\frac{4}{3}t-\frac{13}{9}$ (undetermined coeff.)'],
        ['steelblue', 'crimson']):
    ax.plot(t_plot, y_anal, color=color, lw=2.5, label=lbl)
    ax.plot(t_dots, vop, 'ro', markersize=6, label='Variation of parameters')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('$t$'); ax.legend(fontsize=8)

axes[0].set_title('$x_p(t)$ component'); axes[1].set_title('$y_p(t)$ component')
plt.suptitle("Example 4.49: Undetermined Coefficients vs. Variation of Parameters",
             fontsize=11)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
