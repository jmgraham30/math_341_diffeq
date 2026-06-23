# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from scipy.integrate import solve_ivp     # SciPy ODE solver
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

x, y = sym.symbols('x y')

F1 = x*(1 - y)
G1 = -y + x**2

# Jacobian
J1 = sym.Matrix([[sym.diff(F1,x), sym.diff(F1,y)],
                 [sym.diff(G1,x), sym.diff(G1,y)]])
display(Math(r'J = ' + sym.latex(J1)))

# Equilibria
equil1 = sym.solve([F1, G1], [x, y])
display(Math(r'\text{Equilibria: }' + sym.latex(equil1)))

# Classify each
for pt in equil1:
    Jpt = J1.subs([(x, pt[0]), (y, pt[1])])
    eigs = Jpt.eigenvals()
    tau  = Jpt.trace()
    delta = Jpt.det()
    display(Math(r'(' + sym.latex(pt[0]) + r',' + sym.latex(pt[1]) + r'):\quad'
                 + r'\tau=' + sym.latex(tau)
                 + r',\;\delta=' + sym.latex(delta)
                 + r',\;\lambda=' + sym.latex(list(eigs.keys()))))

def sys1(t, state):
    xv, yv = state
    return [xv*(1 - yv), -yv + xv**2]

fig, ax = plt.subplots(figsize=(7, 6))

# Nullclines
xr = np.linspace(-2.5, 2.5, 400)
ax.axvline(0,   color='steelblue',  lw=1.6, ls='--', label=r"$x'=0$: $x=0$")
ax.axhline(1,   color='steelblue',  lw=1.6, ls='--', label=r"$x'=0$: $y=1$")
ax.plot(xr, xr**2, color='darkorange', lw=1.6, ls='--', label=r"$y'=0$: $y=x^2$")

# Direction field
X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 20), np.linspace(-0.5, 3.0, 20))
U = X*(1 - Y)
V = -Y + X**2
spd = np.sqrt(U**2 + V**2)
spd[spd == 0] = 1
ax.quiver(X, Y, U/spd, V/spd, alpha=0.3, color='gray',
          scale=25, width=0.003)

# Trajectories from several initial conditions
ic_list = [
    (0.1, 0.1), (-0.1, 0.1), (0.5, 2.5), (-0.5, 2.5),
    (1.8, 0.2), (-1.8, 0.2), (0.3, 2.8), (-0.3, 2.8),
    (1.5, 2.5), (-1.5, 2.5), (2.0, 1.5), (-2.0, 1.5),
    (0.8, 0.1), (-0.8, 0.1),
]
for x0, y0 in ic_list:
    sol = solve_ivp(sys1, [0, 12], [x0, y0], max_step=0.05,
                    dense_output=True)
    ts = np.linspace(0, 12, 1200)
    xs, ys = sol.sol(ts)
    mask = (np.abs(xs) < 2.6) & (np.abs(ys) < 3.1) & (ys > -0.6)
    ax.plot(xs[mask], ys[mask], color='steelblue', lw=1.0, alpha=0.55)

# Equilibria
ax.plot(0,  0, 's', color='tomato',    ms=9, zorder=6, label='Saddle $(0,0)$')
ax.plot(1,  1, 'o', color='seagreen',  ms=9, zorder=6, label='Sink $(1,1)$')
ax.plot(-1, 1, 'o', color='seagreen',  ms=9, zorder=6, label='Sink $(-1,1)$')

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.5, 3.0)
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"Phase portrait: $x'=x(1-y)$, $y'=-y+x^2$", fontsize=11)
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.show()

k = sym.Symbol('k')

F2 = x*(2 - x - k*y)
G2 = y*(3 - 2*y - x)

J2 = sym.Matrix([[sym.diff(F2,x), sym.diff(F2,y)],
                 [sym.diff(G2,x), sym.diff(G2,y)]])
display(Math(r'J = ' + sym.latex(J2)))

# Interior equilibrium (general k)
interior = sym.solve([2 - x - k*y, 3 - 2*y - x], [x, y])
display(Math(r'\text{Interior equilibrium (general }k\text{): }'
            + sym.latex(interior)))

# Classify for each k value
for kval in [1, 2, 3]:
    display(Math(r'--- k = ' + str(kval) + r' ---'))
    eqs_k = sym.solve([F2.subs(k, kval), G2.subs(k, kval)], [x, y])
    for pt in eqs_k:
        if pt[0] >= 0 and pt[1] >= 0:
            Jpt = J2.subs([(k, kval), (x, pt[0]), (y, pt[1])])
            tau_  = Jpt.trace()
            delta_ = Jpt.det()
            display(Math(r'\quad(' + sym.latex(pt[0]) + r','
                        + sym.latex(pt[1]) + r'):\quad'
                        + r'\tau=' + sym.latex(tau_)
                        + r',\;\delta=' + sym.latex(delta_)))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, kval, title in [
    (axes[0], 1, r'$k=1$: coexistence node at $(1,1)$'),
    (axes[1], 3, r'$k=3$: no interior equilibrium'),
]:
    def sys2(t, s, kv=kval):
        xv, yv = s
        return [xv*(2 - xv - kv*yv), yv*(3 - 2*yv - xv)]

    # Nullclines
    xr = np.linspace(0, 3, 300)
    # x-nullclines: x=0 and y=(2-x)/k
    ax.axvline(0, color='steelblue', lw=1.5, ls='--')
    ync = (2 - xr)/kval
    mask_nc = ync >= 0
    ax.plot(xr[mask_nc], ync[mask_nc], color='steelblue', lw=1.5, ls='--',
            label=r"$x'=0$")
    # y-nullclines: y=0 and y=(3-x)/2
    ax.axhline(0, color='darkorange', lw=1.5, ls='--')
    ync2 = (3 - xr)/2
    mask_nc2 = ync2 >= 0
    ax.plot(xr[mask_nc2], ync2[mask_nc2], color='darkorange', lw=1.5, ls='--',
            label=r"$y'=0$")

    # Trajectories
    ic_list2 = [(0.2,0.2),(0.5,1.5),(1.5,0.5),(0.2,1.2),(1.2,0.2),
                (0.1,0.8),(0.8,0.1),(1.8,0.8),(0.8,1.8),(1.5,1.5)]
    for x0, y0 in ic_list2:
        try:
            sol = solve_ivp(sys2, [0, 20], [x0, y0], max_step=0.05,
                            dense_output=True)
            ts = np.linspace(0, 20, 1000)
            xs, ys = sol.sol(ts)
            msk = (xs >= 0) & (ys >= 0) & (xs < 3.5) & (ys < 2.5)
            ax.plot(xs[msk], ys[msk], color='steelblue', lw=1.0, alpha=0.55)
        except Exception:
            pass

    # Equilibria
    ax.plot(0, 0,   's', color='gray',    ms=8, zorder=5)
    ax.plot(2, 0,   'o', color='tomato',  ms=8, zorder=5)
    ax.plot(0, 1.5, 'o', color='tomato',  ms=8, zorder=5)
    if kval == 1:
        ax.plot(1, 1, 'o', color='seagreen', ms=10, zorder=6,
                label='Stable node $(1,1)$')

    ax.set_xlim(-0.1, 3.0)
    ax.set_ylim(-0.1, 2.5)
    ax.set_xlabel(r'$x$', fontsize=13)
    ax.set_ylabel(r'$y$', fontsize=13)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc='upper right')

plt.suptitle('Example 2: competing species for two values of $k$', fontsize=11)
plt.tight_layout()
plt.show()

F3 = y
G3 = x - x**3

J3 = sym.Matrix([[sym.diff(F3,x), sym.diff(F3,y)],
                 [sym.diff(G3,x), sym.diff(G3,y)]])
display(Math(r'J = ' + sym.latex(J3)))

equil3 = sym.solve([F3, G3], [x, y])
display(Math(r'\text{Equilibria: }' + sym.latex(equil3)))

for pt in equil3:
    Jpt = J3.subs([(x, pt[0]), (y, pt[1])])
    display(Math(r'(' + sym.latex(pt[0]) + r',' + sym.latex(pt[1])
                + r'):\;\tau=' + sym.latex(Jpt.trace())
                + r',\;\delta=' + sym.latex(Jpt.det())))

# Orbit equation via separation
C_sym = sym.Symbol('C')
orbit_eq = sym.Eq(y**2, x**2 - sym.Rational(1,2)*x**4 + C_sym)
display(Math(r'\text{Orbit equation: }' + sym.latex(orbit_eq)))

fig, ax = plt.subplots(figsize=(7, 6))
ax.axhline(0, color='gray', lw=0.6)
ax.axvline(0, color='gray', lw=0.6)

xr = np.linspace(-1.8, 1.8, 1200)
# H(x,y) = y^2/2 - x^2/2 + x^4/4; orbit: y^2 = x^2 - x^4/2 + C
# At saddle (0,0): H = 0 => C_saddle = 0 (separatrix)
for C_val, col, lw, alpha in [
    ( 0.00, 'tomato',    2.2, 0.9),   # separatrix
    (-0.10, 'steelblue', 1.4, 0.8),   # inner closed orbits around (±1,0)
    (-0.20, 'steelblue', 1.4, 0.8),
    (-0.35, 'steelblue', 1.4, 0.8),
    (-0.45, 'steelblue', 1.4, 0.8),
    ( 0.20, 'seagreen',  1.4, 0.7),   # outer orbits
    ( 0.50, 'seagreen',  1.4, 0.7),
    ( 0.80, 'seagreen',  1.4, 0.7),
]:
    rhs = xr**2 - 0.5*xr**4 + C_val
    valid = rhs >= 0
    y_pos =  np.where(valid, np.sqrt(np.maximum(rhs, 0)), np.nan)
    y_neg = -y_pos
    ax.plot(xr, y_pos, color=col, lw=lw, alpha=alpha)
    ax.plot(xr, y_neg, color=col, lw=lw, alpha=alpha)

# Equilibria
ax.plot( 0, 0, 's', color='tomato',   ms=10, zorder=6, label='Saddle $(0,0)$')
ax.plot( 1, 0, 'o', color='seagreen', ms=10, zorder=6, label='Centers $(\\pm1,0)$')
ax.plot(-1, 0, 'o', color='seagreen', ms=10, zorder=6)

ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r"Orbits: $y^2 = x^2 - x^4/2 + C$", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

x_s = sym.Symbol('x')

F4 = 3*x_s**2 - 3
V4 = -sym.integrate(F4, x_s)
V4_const = V4 - V4.subs(x_s, 0)
display(Math(r'V(x) = ' + sym.latex(V4_const)))

# Equilibria
equil4 = sym.solve(F4, x_s)
display(Math(r'\text{Critical points of }V: x = ' + sym.latex(equil4)))
display(Math(r'V(-1) = ' + sym.latex(V4_const.subs(x_s, -1))
            + r',\quad V(1) = ' + sym.latex(V4_const.subs(x_s, 1))))

E_val = sym.Rational(1,2)*4 + V4_const.subs(x_s, 0)
display(Math(r'E = \tfrac{1}{2}(2)^2 + V(0) = ' + sym.latex(E_val)))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

xr = np.linspace(-2.2, 2.2, 600)
V_np = -xr**3 + 3*xr

# Left: potential energy
ax = axes[0]
ax.plot(xr, V_np, color='steelblue', lw=2.2)
ax.axhline(0, color='gray', lw=0.7)
ax.axhline(2, color='tomato', lw=1.2, ls='--', label=r'$E=2$ (saddle energy)')
ax.plot( 1,  2, 's', color='tomato',   ms=9, zorder=5, label='Saddle $(1,0)$')
ax.plot(-1, -2, 'o', color='seagreen', ms=9, zorder=5, label='Center $(-1,0)$')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-5, 6)
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$V(x)$', fontsize=13)
ax.set_title(r'Potential energy $V(x) = -x^3+3x$', fontsize=11)
ax.legend(fontsize=9)

# Right: phase portrait
ax = axes[1]
ax.axhline(0, color='gray', lw=0.6)
ax.axvline(0, color='gray', lw=0.6)

for E_val_np, col, lw, alpha in [
    ( 2.0, 'tomato',    2.2, 0.95),   # separatrix E = V(saddle) = 2
    (-1.5, 'steelblue', 1.4, 0.8),   # closed orbits near center
    (-1.0, 'steelblue', 1.4, 0.8),
    ( 0.0, 'steelblue', 1.4, 0.7),
    ( 1.0, 'steelblue', 1.4, 0.7),
    ( 3.5, 'seagreen',  1.4, 0.7),   # unbounded orbit above separatrix
]:
    rhs = 2*(E_val_np - V_np)   # y^2 = 2(E - V(x))
    valid = rhs >= 0
    y_pos =  np.where(valid, np.sqrt(np.maximum(rhs, 0)), np.nan)
    y_neg = -y_pos
    ax.plot(xr, y_pos, color=col, lw=lw, alpha=alpha)
    ax.plot(xr, y_neg, color=col, lw=lw, alpha=alpha)

ax.plot( 1, 0, 's', color='tomato',   ms=9, zorder=6, label='Saddle $(1,0)$')
ax.plot(-1, 0, 'o', color='seagreen', ms=9, zorder=6, label='Center $(-1,0)$')
ax.plot( 0, 2, 'D', color='black',    ms=7, zorder=6, label='IC $(0,2)$')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-4, 4)
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel("$y = x'$", fontsize=13)
ax.set_title(r'Energy orbits: $y^2 = 2(E + x^3 - 3x)$', fontsize=11)
ax.legend(fontsize=9)
plt.suptitle(r"Example 4: $x'' = 3x^2-3$, $m=1$", fontsize=11)
plt.tight_layout()
plt.show()

F5 = -4*x_s - 2*x_s**3
V5 = -sym.integrate(F5, x_s)
display(Math(r'V(x) = ' + sym.latex(V5)))
display(Math(r'V(0) = ' + sym.latex(V5.subs(x_s, 0))))

# Equilibria
equil5 = sym.solve(F5, x_s)
display(Math(r'\text{Equilibria (}y=0\text{)}: x = ' + sym.latex(equil5)))

# Jacobian at origin
J5 = sym.Matrix([[0, 1],[sym.diff(F5, x_s), 0]])
J5_0 = J5.subs(x_s, 0)
display(Math(r'J(0,0) = ' + sym.latex(J5_0)
            + r',\quad \lambda = ' + sym.latex(J5_0.eigenvals())))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

xr = np.linspace(-2.5, 2.5, 800)
V5_np = 2*xr**2 + 0.5*xr**4

# Left: potential energy
ax = axes[0]
ax.plot(xr, V5_np, color='steelblue', lw=2.2)
ax.plot(0, 0, 'o', color='seagreen', ms=9, zorder=5, label='Center $(0,0)$')
for E_lev in [0.5, 1.0, 2.0, 4.0]:
    ax.axhline(E_lev, color='gray', lw=0.8, ls='--', alpha=0.6)
    ax.text(2.1, E_lev, f'E={E_lev}', fontsize=7, va='center')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-0.3, 6)
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$V(x)$', fontsize=13)
ax.set_title(r'Potential: $V(x)=2x^2+x^4/2$', fontsize=11)
ax.legend(fontsize=9)

# Right: phase orbits
ax = axes[1]
ax.axhline(0, color='gray', lw=0.6)
ax.axvline(0, color='gray', lw=0.6)
colors_e = ['steelblue', 'steelblue', 'tomato', 'seagreen']
for E_val_np, col in zip([0.5, 1.0, 2.0, 4.0], colors_e):
    rhs = 2*(E_val_np - V5_np)
    valid = rhs >= 0
    y_pos =  np.where(valid, np.sqrt(np.maximum(rhs, 0)), np.nan)
    y_neg = -y_pos
    lw = 2.2 if col == 'tomato' else 1.4
    ax.plot(xr, y_pos, color=col, lw=lw, alpha=0.85, label=f'E={E_val_np}')
    ax.plot(xr, y_neg, color=col, lw=lw, alpha=0.85)
ax.plot(0, 0, 'o', color='seagreen', ms=9, zorder=5)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel("$y = x'$", fontsize=13)
ax.set_title('Energy orbits (hardening spring)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
plt.suptitle(r"Example 5: $x'' = -4x-2x^3$", fontsize=11)
plt.tight_layout()
plt.show()

F6 = -sym.sin(x_s)
V6 = -sym.integrate(F6, x_s) + 1   # V(0) = 0 => constant = 1
display(Math(r'V(x) = ' + sym.latex(V6)))

# Energy of IVP
E_ivp = sym.Rational(1,2)*0 + V6.subs(x_s, sym.pi/3)
display(Math(r'E\big|_{x_0=\pi/3,\,y_0=0} = ' + sym.latex(sym.simplify(E_ivp))))

# Saddle energy
E_saddle = V6.subs(x_s, sym.pi)
display(Math(r'E_{\rm saddle} = V(\pi) = ' + sym.latex(E_saddle)))

fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(0, color='gray', lw=0.6)

xr = np.linspace(-np.pi - 0.05, 2*np.pi + 0.05, 1200)
V6_np = 1 - np.cos(xr)

for E_val_np, col, lw, alpha in [
    (2.0,  'tomato',    2.2, 0.95),   # separatrix
    (0.3,  'steelblue', 1.3, 0.8),
    (0.5,  'seagreen',  2.0, 0.95),   # IVP orbit
    (1.0,  'steelblue', 1.3, 0.7),
    (1.6,  'steelblue', 1.3, 0.7),
    (2.8,  'steelblue', 1.3, 0.65),   # rotational orbits
    (4.0,  'steelblue', 1.3, 0.55),
]:
    rhs = 2*(E_val_np - V6_np)
    valid = rhs >= 0
    y_pos =  np.where(valid, np.sqrt(np.maximum(rhs, 0)), np.nan)
    y_neg = -y_pos
    ax.plot(xr, y_pos, color=col, lw=lw, alpha=alpha)
    ax.plot(xr, y_neg, color=col, lw=lw, alpha=alpha)

# Equilibria
for xeq, marker, col, label in [
    (-np.pi, 's', 'tomato',   'Saddle'),
    ( 0,     'o', 'seagreen', 'Center'),
    ( np.pi, 's', 'tomato',   None),
    (2*np.pi,'o', 'seagreen', None),
]:
    ax.plot(xeq, 0, marker, color=col, ms=9, zorder=6,
            label=label if label else '')

# IVP initial condition
ax.plot(np.pi/3, 0, 'D', color='black', ms=8, zorder=7,
        label=r'IC $(\pi/3, 0)$, $E=1/2$')

ax.set_xlim(-np.pi - 0.1, 2*np.pi + 0.1)
ax.set_ylim(-3.2, 3.2)
ax.set_xlabel(r'$x$ (angle)', fontsize=13)
ax.set_ylabel(r"$y = x'$ (angular velocity)", fontsize=13)
ax.set_title('Nonlinear pendulum: separatrix (tomato) at $E=2$', fontsize=11)

# x-axis tick labels at multiples of pi
ax.set_xticks([-np.pi, 0, np.pi, 2*np.pi])
ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$'])

from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color='tomato',    lw=2.2, label='Separatrix ($E=2$)'),
    Line2D([0],[0], color='seagreen',  lw=2.0, label=r'IVP orbit ($E=1/2$)'),
    Line2D([0],[0], color='steelblue', lw=1.3, label='Other orbits'),
    Line2D([0],[0], marker='s', color='tomato',  lw=0, ms=8, label='Saddle'),
    Line2D([0],[0], marker='o', color='seagreen',lw=0, ms=8, label='Center'),
    Line2D([0],[0], marker='D', color='black',   lw=0, ms=7,
           label=r'IC $(\pi/3,0)$'),
]
ax.legend(handles=handles, fontsize=8, loc='upper right')
plt.tight_layout()
plt.show()