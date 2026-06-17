#| label: setup
#| code-fold: true
#| code-summary: "Show the code"

# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


#| label: ex1-sympy

t = sym.Symbol('t')
c1, c2 = sym.symbols('c1 c2')

lam1, lam2 = -1, -4
v1 = sym.Matrix([1, 3])
v2 = sym.Matrix([2, 1])

x_gen = c1 * v1 * sym.exp(lam1*t) + c2 * v2 * sym.exp(lam2*t)
display(Math(r'\mathbf{x}(t) = c_1 '
            + sym.latex(v1) + r'e^{-t} + c_2 '
            + sym.latex(v2) + r'e^{-4t}'))
display(Math(r'x(t) = ' + sym.latex(sym.simplify(x_gen[0]))))
display(Math(r'y(t) = ' + sym.latex(sym.simplify(x_gen[1]))))


#| label: fig-ex1
#| fig-cap: "Phase portrait for Example 1: asymptotically stable node with eigenvalues $\\lambda_1=-1$ and $\\lambda_2=-4$. The linear orbits along $\\mathbf{v}_1=(1,3)^T$ (tomato) and $\\mathbf{v}_2=(2,1)^T$ (steelblue) are highlighted; all other orbits approach the origin tangent to $\\mathbf{v}_1$."
#| code-fold: true
#| code-summary: "Show the code"

v1_np = np.array([1, 3], dtype=float)
v2_np = np.array([2, 1], dtype=float)

t_fwd = np.linspace(0,  5, 400)
t_bwd = np.linspace(0, -3, 400)

fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(0, color='gray', lw=0.7)
ax.axvline(0, color='gray', lw=0.7)

# Several orbits: vary (c1, c2)
for c1v, c2v in [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (1,  1), (1, -1), (-1,  1), (-1, -1),
                 (2,  1), (-2, 1)]:
    for t_range in [t_fwd, t_bwd]:
        xc = c1v * v1_np[0] * np.exp(-1*t_range) + c2v * v2_np[0] * np.exp(-4*t_range)
        yc = c1v * v1_np[1] * np.exp(-1*t_range) + c2v * v2_np[1] * np.exp(-4*t_range)
        # Highlight pure eigenvector orbits
        if c2v == 0 and c1v != 0:
            ax.plot(xc, yc, color='tomato', lw=2.0, alpha=0.9)
        elif c1v == 0 and c2v != 0:
            ax.plot(xc, yc, color='steelblue', lw=2.0, alpha=0.9)
        else:
            ax.plot(xc, yc, color='gray', lw=1.0, alpha=0.5)

# Eigenvector direction arrows
ax.annotate('', xy=1.5*v1_np, xytext=np.zeros(2),
            arrowprops=dict(arrowstyle='->', color='tomato', lw=1.8))
ax.annotate('', xy=-1.5*v1_np, xytext=np.zeros(2),
            arrowprops=dict(arrowstyle='->', color='tomato', lw=1.8))
ax.annotate('', xy=1.5*v2_np, xytext=np.zeros(2),
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.8))
ax.annotate('', xy=-1.5*v2_np, xytext=np.zeros(2),
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.8))

ax.plot(0, 0, 'ko', ms=6, zorder=5)
ax.set_xlim(-4, 4)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title('Asymptotically stable node: '
             r'$\lambda_1=-1,\;\lambda_2=-4$', fontsize=11)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], color='tomato',    lw=2, label=r'Linear orbit along $\mathbf{v}_1$'),
    Line2D([0],[0], color='steelblue', lw=2, label=r'Linear orbit along $\mathbf{v}_2$'),
    Line2D([0],[0], color='gray',      lw=1, label='Other orbits'),
]
ax.legend(handles=legend_handles, fontsize=9, loc='upper right')
plt.tight_layout()
plt.show()


#| label: ex2-sympy

A2 = sym.Matrix([[2, 3], [1, 4]])
lam = sym.Symbol('lambda')

char2 = A2.charpoly(lam)
display(Math(r'\text{Characteristic polynomial: }\quad'
            + sym.latex(sym.Eq(char2.as_expr(), 0))))

for ev, mult, evecs in A2.eigenvects():
    display(Math(r'\lambda = ' + sym.latex(ev)
                 + r',\quad \mathbf{v} = ' + sym.latex(evecs[0])))

c1, c2 = sym.symbols('c1 c2')
ev_list = [(ev, evecs[0]) for ev, mult, evecs in A2.eigenvects()]
ev_list.sort(key=lambda p: float(p[0]))
x_gen2 = c1*ev_list[0][1]*sym.exp(ev_list[0][0]*t) \
        + c2*ev_list[1][1]*sym.exp(ev_list[1][0]*t)
display(Math(r'x(t) = ' + sym.latex(sym.simplify(x_gen2[0]))))
display(Math(r'y(t) = ' + sym.latex(sym.simplify(x_gen2[1]))))


#| label: fig-ex2
#| fig-cap: "Phase portrait for Example 2: unstable node with $\\lambda_1=1$ and $\\lambda_2=5$. Orbits along the eigenvector directions are highlighted; all trajectories move away from the origin."
#| code-fold: true
#| code-summary: "Show the code"

v1_np = np.array([-3, 1], dtype=float)
v2_np = np.array([1,  1], dtype=float)

t_fwd = np.linspace(0,  1.0, 400)
t_bwd = np.linspace(0, -3.0, 400)

fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(0, color='gray', lw=0.7)
ax.axvline(0, color='gray', lw=0.7)

for c1v, c2v in [(1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 0.3), (-1, 0.3), (1, -0.3), (-1, -0.3),
                 (0.5, 0.5), (-0.5, -0.5)]:
    for t_range in [t_fwd, t_bwd]:
        xc = c1v*v1_np[0]*np.exp(1*t_range) + c2v*v2_np[0]*np.exp(5*t_range)
        yc = c1v*v1_np[1]*np.exp(1*t_range) + c2v*v2_np[1]*np.exp(5*t_range)
        mask = (np.abs(xc) < 5) & (np.abs(yc) < 5)
        if c2v == 0 and c1v != 0:
            ax.plot(xc[mask], yc[mask], color='tomato', lw=2.0, alpha=0.9)
        elif c1v == 0 and c2v != 0:
            ax.plot(xc[mask], yc[mask], color='steelblue', lw=2.0, alpha=0.9)
        else:
            ax.plot(xc[mask], yc[mask], color='gray', lw=1.0, alpha=0.5)

ax.plot(0, 0, 'ko', ms=6, zorder=5)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r'Unstable node: $\lambda_1=1,\;\lambda_2=5$', fontsize=11)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], color='tomato',    lw=2, label=r'Linear orbit along $\mathbf{v}_1$'),
    Line2D([0],[0], color='steelblue', lw=2, label=r'Linear orbit along $\mathbf{v}_2$'),
    Line2D([0],[0], color='gray',      lw=1, label='Other orbits'),
]
ax.legend(handles=legend_handles, fontsize=9, loc='upper right')
plt.tight_layout()
plt.show()


#| label: ex3-sympy

A3 = sym.Matrix([[-1, 4], [1, -4]])
x0 = sym.Matrix([2, 1])

# General solution via dsolve
x_vec = sym.Matrix([sym.Function('x')(t), sym.Function('y')(t)])
ode_sys = x_vec.diff(t) - A3*x_vec

sol = sym.dsolve([sym.Eq(x_vec[0].diff(t), (A3*x_vec)[0]),
                  sym.Eq(x_vec[1].diff(t), (A3*x_vec)[1])])
for eq in sol:
    display(Math(sym.latex(eq)))

# Solve IVP
ics = {x_vec[0].subs(t,0): x0[0], x_vec[1].subs(t,0): x0[1]}
sol_ivp = sym.dsolve([sym.Eq(x_vec[0].diff(t), (A3*x_vec)[0]),
                      sym.Eq(x_vec[1].diff(t), (A3*x_vec)[1])], ics=ics)
display(Math(r'\text{IVP solution:}'))
for eq in sol_ivp:
    display(Math(sym.latex(eq)))


#| label: fig-ex3
#| fig-cap: "Solution of the IVP in Example 3. The particular solution (tomato) starts at $(2,1)$ and converges to $\\frac{3}{5}(4,1)^T \\approx (2.4, 0.6)$ on the line of equilibria $y = \\frac{1}{4}x$ (dashed)."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(0, 2, 500)

x_sol = 12/5 - (2/5)*np.exp(-5*t_vals)
y_sol =  3/5 + (2/5)*np.exp(-5*t_vals)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: time series
ax = axes[0]
ax.plot(t_vals, x_sol, color='steelblue', lw=2, label=r'$x(t)$')
ax.plot(t_vals, y_sol, color='tomato',    lw=2, label=r'$y(t)$')
ax.axhline(12/5, color='steelblue', lw=0.9, ls='--', alpha=0.6)
ax.axhline( 3/5, color='tomato',    lw=0.9, ls='--', alpha=0.6)
ax.plot(0, 2, 'ko', ms=5)
ax.plot(0, 1, 'ko', ms=5)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel('Component', fontsize=12)
ax.set_title('Time series', fontsize=11)
ax.legend(fontsize=10)

# Right: phase plane
ax = axes[1]
# Line of equilibria: y = x/4
x_line = np.linspace(-1, 4, 200)
ax.plot(x_line, x_line/4, 'k--', lw=1.2, label=r'Equil. line $y=\frac{1}{4}x$')

# Several orbits
v1_np = np.array([4, 1], dtype=float)
v2_np = np.array([-1, 1], dtype=float)
for c1v, c2v in [(3/5, 2/5), (1, 0), (-1, 0), (0, 1), (0, -1),
                 (0.5, 1), (0.5, -1), (-0.5, 1)]:
    xc = c1v*v1_np[0] + c2v*v2_np[0]*np.exp(-5*t_vals)
    yc = c1v*v1_np[1] + c2v*v2_np[1]*np.exp(-5*t_vals)
    if c1v == 3/5 and c2v == 2/5:
        ax.plot(xc, yc, color='tomato', lw=2.2, zorder=4,
                label='IVP solution')
    else:
        ax.plot(xc, yc, color='gray', lw=1.0, alpha=0.5)

ax.plot(2, 1, 'ko', ms=6, zorder=5, label=r'$\mathbf{x}(0)=(2,1)^T$')
ax.set_xlim(-2, 5)
ax.set_ylim(-1, 3)
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title('Phase plane', fontsize=11)
ax.legend(fontsize=8)

plt.suptitle(r"Example 3: zero eigenvalue ($\lambda_1=0$, $\lambda_2=-5$)",
             fontsize=11)
plt.tight_layout()
plt.show()


#| label: ex4-sympy

A4 = sym.Matrix([[-1, -3], [3, -1]])

char4 = A4.charpoly(lam)
display(Math(r'\text{Characteristic polynomial: }\quad'
            + sym.latex(sym.Eq(char4.as_expr(), 0))))

roots4 = sym.solve(char4.as_expr(), lam)
display(Math(r'\lambda = ' + sym.latex(roots4)))

x4 = sym.Function('x')(t)
y4 = sym.Function('y')(t)
sol4 = sym.dsolve([sym.Eq(x4.diff(t), -x4 - 3*y4),
                   sym.Eq(y4.diff(t),  3*x4 - y4)])
for eq in sol4:
    display(Math(sym.latex(eq)))


#| label: fig-ex4
#| fig-cap: "Phase portrait for Example 4: asymptotically stable spiral sink with $\\lambda = -1 \\pm 3i$. Trajectories spiral inward toward the origin. The tomato orbit starts at $(2,0)$."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(0, 5, 2000)
fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(0, color='gray', lw=0.7)
ax.axvline(0, color='gray', lw=0.7)

for c1v, c2v, col in [
        ( 2,  0, 'tomato'),
        (-2,  0, 'steelblue'),
        ( 0,  2, 'steelblue'),
        ( 0, -2, 'steelblue'),
        ( 1,  1, 'steelblue'),
        (-1, -1, 'steelblue'),
        ( 1, -1, 'steelblue'),
        (-1,  1, 'steelblue'),
]:
    xc = np.exp(-t_vals)*(c1v*np.cos(3*t_vals) + c2v*np.sin(3*t_vals))
    yc = np.exp(-t_vals)*(c1v*np.sin(3*t_vals) - c2v*np.cos(3*t_vals))
    ax.plot(xc, yc, color=col,
            lw=2.0 if col == 'tomato' else 1.2,
            alpha=0.9 if col == 'tomato' else 0.55)

ax.plot(0, 0, 'ko', ms=6, zorder=5)
ax.plot(2, 0, 'o', color='tomato', ms=7, zorder=5)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r'Spiral sink: $\lambda = -1 \pm 3i$', fontsize=11)
plt.tight_layout()
plt.show()


#| label: ex5-sympy

A5 = sym.Matrix([[0, -4], [1, 0]])

roots5 = sym.solve(A5.charpoly(lam).as_expr(), lam)
display(Math(r'\lambda = ' + sym.latex(roots5)))

x5 = sym.Function('x')(t)
y5 = sym.Function('y')(t)
sol5 = sym.dsolve([sym.Eq(x5.diff(t), -4*y5),
                   sym.Eq(y5.diff(t),   x5)])
for eq in sol5:
    display(Math(sym.latex(eq)))


#| label: fig-ex5
#| fig-cap: "Phase portrait for Example 5: stable center with $\\lambda = \\pm 2i$. Orbits are concentric ellipses satisfying $x^2/4 + y^2 = $ const. The tomato ellipse corresponds to $c_1=0$, $c_2=1$."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(0, 2*np.pi, 1000)
fig, ax = plt.subplots(figsize=(6, 6))
ax.axhline(0, color='gray', lw=0.7)
ax.axvline(0, color='gray', lw=0.7)

for c1v, c2v, col, lw in [
        (0, 1, 'tomato', 2.2),
        (1, 0, 'steelblue', 1.4),
        (0, 2, 'steelblue', 1.4),
        (1, 1, 'steelblue', 1.4),
        (0, 3, 'steelblue', 1.4),
        (2, 1, 'steelblue', 1.4),
]:
    xc = 2*(-c1v*np.sin(2*t_vals) + c2v*np.cos(2*t_vals))
    yc =    c1v*np.cos(2*t_vals)  + c2v*np.sin(2*t_vals)
    ax.plot(xc, yc, color=col, lw=lw, alpha=0.9 if col=='tomato' else 0.6)

ax.plot(0, 0, 'ko', ms=6, zorder=5)
ax.set_xlim(-7, 7)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r'Stable center: $\lambda = \pm 2i$', fontsize=11)
plt.tight_layout()
plt.show()


#| label: ex6-sympy

A6 = sym.Matrix([[2, -5], [1, -2]])

roots6 = sym.solve(A6.charpoly(lam).as_expr(), lam)
display(Math(r'\lambda = ' + sym.latex(roots6)))

x6 = sym.Function('x')(t)
y6 = sym.Function('y')(t)
sol6_ivp = sym.dsolve(
    [sym.Eq(x6.diff(t),  2*x6 - 5*y6),
     sym.Eq(y6.diff(t),    x6 - 2*y6)],
    ics={x6.subs(t, 0): 1, y6.subs(t, 0): 0}
)
display(Math(r'\text{IVP solution:}'))
for eq in sol6_ivp:
    display(Math(sym.latex(eq)))


#| label: fig-ex6
#| fig-cap: "Solution of the IVP in Example 6. Left: time series showing $x(t) = 2\\sin t + \\cos t$ (steelblue) and $y(t)=\\sin t$ (tomato), both periodic with period $2\\pi$. Right: phase portrait showing the closed elliptical orbit through $(1,0)$."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(0, 4*np.pi, 1000)
x_sol6 = 2*np.sin(t_vals) + np.cos(t_vals)
y_sol6 = np.sin(t_vals)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: time series
ax = axes[0]
ax.plot(t_vals, x_sol6, color='steelblue', lw=2,
        label=r'$x(t) = 2\sin t + \cos t$')
ax.plot(t_vals, y_sol6, color='tomato', lw=2,
        label=r'$y(t) = \sin t$')
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel('Component', fontsize=12)
ax.set_title('Time series', fontsize=11)
ax.legend(fontsize=9)

# Right: phase portrait
ax = axes[1]
# Background orbits (other initial conditions)
for c2v in [0.5, 1.5, 2.0]:
    xb = c2v*(2*np.sin(t_vals) + np.cos(t_vals))
    yb = c2v*np.sin(t_vals)
    ax.plot(xb, yb, color='steelblue', lw=1.2, alpha=0.4)

# IVP orbit (c1=0, c2=1)
ax.plot(x_sol6, y_sol6, color='tomato', lw=2.2, label='IVP orbit')
ax.plot(1, 0, 'ko', ms=7, zorder=5, label=r'$\mathbf{x}(0)=(1,0)^T$')
ax.plot(0, 0, 'ko', ms=5, zorder=4)

# Arrow to show direction
idx = 50
ax.annotate('', xy=(x_sol6[idx+5], y_sol6[idx+5]),
            xytext=(x_sol6[idx], y_sol6[idx]),
            arrowprops=dict(arrowstyle='->', color='tomato', lw=1.5))

ax.set_xlim(-5, 5)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r'Stable center: $\lambda = \pm i$', fontsize=11)
ax.legend(fontsize=9)

plt.suptitle(r'Example 6: $\lambda = \pm i$, IVP with $\mathbf{x}(0)=(1,0)^T$',
             fontsize=11)
plt.tight_layout()
plt.show()


#| label: session-info

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}'
                for m in globals().values()
                if getattr(m, '__version__', None)))
