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

v1 = sym.Matrix([1, 2])
v2 = sym.Matrix([3, 1])
lam1, lam2 = -5, -1

# Recover A
P   = sym.Matrix([[1, 3], [2, 1]])
Lam = sym.diag(-5, -1)
A1  = P * Lam * P.inv()
display(Math(r'A = ' + sym.latex(A1)))

# Nullclines
x, y = sym.symbols('x y')
nc_x = sym.solve((A1 * sym.Matrix([x, y]))[0], y)
nc_y = sym.solve((A1 * sym.Matrix([x, y]))[1], y)
display(Math(r'x\text{-nullcline: } y = ' + sym.latex(nc_x[0]) + r'x'))
display(Math(r'y\text{-nullcline: } y = ' + sym.latex(nc_y[0]) + r'x'))

# IVP 1
x0a = sym.Matrix([4, 3])
coeffs_a = P.solve(x0a)
display(Math(r'\mathbf{x}(0)=(4,3)^T:\quad c_1=' + sym.latex(coeffs_a[0])
            + r',\quad c_2=' + sym.latex(coeffs_a[1])))

# IVP 2
x0b = sym.Matrix([-2, -5])
coeffs_b = P.solve(x0b)
display(Math(r'\mathbf{x}(0)=(-2,-5)^T:\quad c_1=' + sym.latex(coeffs_b[0])
            + r',\quad c_2=' + sym.latex(coeffs_b[1])))


#| label: fig-ex1
#| fig-cap: "Phase portrait for Example 1: stable node with $\\lambda_1=-5$, $\\lambda_2=-1$. The $x$-nullcline (blue dashed) and $y$-nullcline (orange dashed) pass through the origin. Linear orbits along $\\mathbf{v}_1$ (gray) and $\\mathbf{v}_2$ (gray) are shown, with the two IVP solution curves highlighted."
#| code-fold: true
#| code-summary: "Show the code"

v1n = np.array([1, 2], dtype=float)
v2n = np.array([3, 1], dtype=float)
t_f = np.linspace(0,  4, 600)
t_b = np.linspace(0, -1.5, 300)

fig, ax = plt.subplots(figsize=(7, 7))
ax.axhline(0, color='gray', lw=0.6)
ax.axvline(0, color='gray', lw=0.6)

# Background orbits
for c1v, c2v in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(2,1),(-2,-1)]:
    for tr in [t_f, t_b]:
        xc = c1v*v1n[0]*np.exp(-5*tr) + c2v*v2n[0]*np.exp(-1*tr)
        yc = c1v*v1n[1]*np.exp(-5*tr) + c2v*v2n[1]*np.exp(-1*tr)
        ax.plot(xc, yc, color='gray', lw=0.9, alpha=0.4)

# Nullclines: x-null: x = -12y => y = -x/12; y-null: y = 8x/29
xr = np.linspace(-8, 8, 200)
ax.plot(xr, -xr/12,    color='steelblue', lw=1.6, ls='--', label=r"$x'=0$: $y = -x/12$")
ax.plot(xr,  8*xr/29,  color='darkorange', lw=1.6, ls='--', label=r"$y'=0$: $y = 8x/29$")

# IVP orbit 1: c1=1, c2=1, x(0)=(4,3)
x_a = 1*v1n[0]*np.exp(-5*t_f) + 1*v2n[0]*np.exp(-1*t_f)
y_a = 1*v1n[1]*np.exp(-5*t_f) + 1*v2n[1]*np.exp(-1*t_f)
ax.plot(x_a, y_a, color='tomato', lw=2.2, label=r'IVP: $\mathbf{x}(0)=(4,3)^T$')
ax.plot(4, 3, 'o', color='tomato', ms=7, zorder=5)

# IVP orbit 2: c1=-13/5, c2=1/5, x(0)=(-2,-5)
x_b = (-13/5)*v1n[0]*np.exp(-5*t_f) + (1/5)*v2n[0]*np.exp(-1*t_f)
y_b = (-13/5)*v1n[1]*np.exp(-5*t_f) + (1/5)*v2n[1]*np.exp(-1*t_f)
ax.plot(x_b, y_b, color='seagreen', lw=2.2, label=r'IVP: $\mathbf{x}(0)=(-2,-5)^T$')
ax.plot(-2, -5, 'o', color='seagreen', ms=7, zorder=5)

# Direction arrows on IVP 1
for idx in [60, 140, 240]:
    ax.annotate('', xy=(x_a[idx+6], y_a[idx+6]), xytext=(x_a[idx], y_a[idx]),
                arrowprops=dict(arrowstyle='->', color='tomato', lw=1.4))

ax.plot(0, 0, 'ko', ms=7, zorder=6)
ax.set_xlim(-8, 8)
ax.set_ylim(-7, 7)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title('Stable node with nullclines and two IVP orbits', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.show()


#| label: ex2-sympy

A2 = sym.Matrix([[-3, 4], [0, -3]])
lam = sym.Symbol('lambda')

display(Math(r'\text{Eigenvalues: }' + sym.latex(A2.eigenvals())))

for ev, mult, evecs in A2.eigenvects():
    display(Math(r'\lambda=' + sym.latex(ev)
                 + r'\;(\text{mult }' + str(mult) + r'),\quad'
                 + r'\mathbf{v}=' + sym.latex(evecs[0])))

# Generalized eigenvector: (A + 3I)w = v  (singular system, use gauss_jordan_solve)
v_eig = sym.Matrix([1, 0])
# gauss_jordan_solve returns (solution, free_variable_matrix)
w_particular, w_free = (A2 + 3*sym.eye(2)).gauss_jordan_solve(v_eig)
# Set free parameter(s) to zero to get the simplest particular solution
w_gen = w_particular.subs(list(zip(w_free, [0]*len(w_free))))
display(Math(r'\text{Generalized eigenvector: }\mathbf{w}=' + sym.latex(w_gen)))


#| label: fig-ex2
#| fig-cap: "Phase portrait for Example 2: asymptotically stable improper node ($\\lambda=-3$ repeated, defective). The single linear orbit along $(1,0)^T$ is shown in tomato; the $x'$-nullcline $y=3x/4$ (steelblue dashed) and $y'$-nullcline $y=0$ (orange dashed) are included."
#| code-fold: true
#| code-summary: "Show the code"

t_f = np.linspace(0,  3.0, 600)
t_b = np.linspace(0, -0.8, 200)

fig, ax = plt.subplots(figsize=(7, 6))
ax.axhline(0, color='gray', lw=0.6)
ax.axvline(0, color='gray', lw=0.6)

# Background orbits: x=(c1+c2*t)*e^{-3t}, y=c2/4*e^{-3t}
for c1v, c2v in [(1,0),(-1,0),(0,4),(0,-4),(1,4),(1,-4),(-1,4),(-1,-4),(2,2),(-2,2)]:
    for tr in [t_f, t_b]:
        xc = (c1v + c2v*tr)*np.exp(-3*tr)
        yc = (c2v/4)*np.exp(-3*tr)
        if c2v == 0:
            ax.plot(xc, yc, color='tomato', lw=2.0, alpha=0.9)
        else:
            mask = (np.abs(xc) < 6) & (np.abs(yc) < 4)
            ax.plot(xc[mask], yc[mask], color='gray', lw=1.0, alpha=0.45)

# Nullclines
xr = np.linspace(-6, 6, 200)
ax.plot(xr, 0.75*xr, color='steelblue', lw=1.7, ls='--', label=r"$x'=0$: $y=\frac{3}{4}x$")
ax.plot(xr, np.zeros_like(xr), color='darkorange', lw=1.7, ls='--', label=r"$y'=0$: $y=0$")

ax.plot(0, 0, 'ko', ms=7, zorder=6)
ax.set_xlim(-6, 6)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_xlabel(r'$x$', fontsize=13)
ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title(r'Stable improper node: $\lambda=-3$ (repeated, defective)', fontsize=11)
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0], color='tomato',    lw=2, label='Linear orbit (eigenvector dir.)'),
    Line2D([0],[0], color='gray',      lw=1, label='Other orbits'),
    Line2D([0],[0], color='steelblue', lw=1.7, ls='--', label=r"$x'=0$ nullcline"),
    Line2D([0],[0], color='darkorange',lw=1.7, ls='--', label=r"$y'=0$ nullcline"),
]
ax.legend(handles=handles, fontsize=9, loc='upper right')
plt.tight_layout()
plt.show()


#| label: ex3-sympy

alpha = sym.Symbol('alpha', real=True)
A3 = sym.Matrix([[2, -5], [alpha, -2]])

tau3   = A3.trace()
delta3 = A3.det()
display(Math(r'\tau = \operatorname{tr}(A) = ' + sym.latex(tau3)))
display(Math(r'\delta = \det(A) = '           + sym.latex(delta3)))

lam_sym = sym.Symbol('lambda')
char3 = A3.charpoly(lam_sym)
roots3 = sym.solve(char3.as_expr(), lam_sym)
display(Math(r'\lambda(\alpha) = ' + sym.latex(roots3)))

bifurc = sym.solve(delta3, alpha)
display(Math(r'\text{Bifurcation at }\alpha = ' + sym.latex(bifurc)))


#| label: fig-ex3
#| fig-cap: "Left: real and imaginary parts of the eigenvalues $\\lambda(\\alpha)=\\pm\\sqrt{4-5\\alpha}$ vs.\\ $\\alpha$. Right: path traced in the trace-determinant plane as $\\alpha$ increases from $-1$ to $3$; the system crosses from saddle (below the $\\tau$-axis) to center (on the vertical axis, above the parabola) at $\\alpha=4/5$."
#| code-fold: true
#| code-summary: "Show the code"

alpha_vals = np.linspace(-1, 3, 800)
disc = 4 - 5*alpha_vals     # = 4 - 5*alpha

# Eigenvalues
lam_real = np.where(disc >= 0,  np.sqrt(np.maximum(disc, 0)), 0.0)
lam_imag = np.where(disc <  0,  np.sqrt(np.maximum(-disc, 0)), 0.0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: eigenvalue vs alpha
ax = axes[0]
ax.plot(alpha_vals,  lam_real, color='steelblue', lw=2, label=r'$\operatorname{Re}(\lambda)=+\sqrt{|4-5\alpha|}$')
ax.plot(alpha_vals, -lam_real, color='steelblue', lw=2, ls='--')
ax.plot(alpha_vals,  lam_imag, color='tomato',    lw=2, label=r'$\operatorname{Im}(\lambda)=+\sqrt{5\alpha-4}$')
ax.plot(alpha_vals, -lam_imag, color='tomato',    lw=2, ls='--')
ax.axvline(4/5, color='gray', lw=1.0, ls=':')
ax.axhline(0,   color='gray', lw=0.6)
ax.annotate(r'$\alpha=4/5$', xy=(4/5, 0.15), xytext=(1.2, 1.2),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
ax.set_xlabel(r'$\alpha$', fontsize=13)
ax.set_ylabel(r'$\lambda$', fontsize=13)
ax.set_title('Eigenvalues vs. parameter', fontsize=11)
ax.legend(fontsize=8)

# Right: trace-determinant plane
tau_path   = np.zeros_like(alpha_vals)          # always 0
delta_path = -4 + 5*alpha_vals

tau_range  = np.linspace(-4, 4, 400)
parab = tau_range**2 / 4

ax = axes[1]
ax.fill_between(tau_range, parab, 8, alpha=0.07, color='steelblue', label='Spiral/Center region')
ax.fill_between(tau_range, -3, 0, alpha=0.07, color='tomato', label='Saddle region')
ax.plot(tau_range, parab, color='gray', lw=1.2, label='Parabola $\\delta=\\tau^2/4$')
ax.axhline(0, color='gray', lw=0.8)
ax.axvline(0, color='gray', lw=0.8)

# Path of the system
ax.plot(tau_path, delta_path, color='black', lw=2.5, label='System path ($\\tau=0$)')
# Mark key points
ax.plot(0, -4 + 5*0,   'o', color='steelblue', ms=8, label=r'$\alpha=0$ (saddle)')
ax.plot(0, -4 + 5*0.8, 's', color='gray',      ms=8, label=r'$\alpha=4/5$ (bifurcation)')
ax.plot(0, -4 + 5*1.5, '^', color='tomato',    ms=8, label=r'$\alpha=3/2$ (center)')

ax.annotate(r'saddle', xy=(0.1, -4), fontsize=8, color='steelblue')
ax.annotate(r'center', xy=(0.1,  3), fontsize=8, color='tomato')

ax.set_xlim(-4, 4)
ax.set_ylim(-5, 8)
ax.set_xlabel(r'$\tau = \operatorname{tr}(A)$', fontsize=12)
ax.set_ylabel(r'$\delta = \det(A)$', fontsize=12)
ax.set_title('Trace-determinant plane', fontsize=11)
ax.legend(fontsize=7, loc='upper right')

plt.suptitle(r'Example 3: $\lambda(\alpha)=\pm\sqrt{4-5\alpha}$, bifurcation at $\alpha=4/5$',
             fontsize=11)
plt.tight_layout()
plt.show()


#| label: ex4-sympy

# Part (a)
Aa = sym.Matrix([[-2, 1], [0, -3]])
ba = sym.Matrix([4, 3])
xstar_a = Aa.solve(-ba)
display(Math(r'\text{(a) Equilibrium: }\mathbf{x}^* = ' + sym.latex(xstar_a)))
display(Math(r'\text{(a) Eigenvalues: }' + sym.latex(list(Aa.eigenvals().keys()))))

# Part (b)
Ab = sym.Matrix([[2, 1], [-3, -2]])
bb = sym.Matrix([-1, 1])
xstar_b = Ab.solve(-bb)
display(Math(r'\text{(b) Equilibrium: }\mathbf{x}^* = ' + sym.latex(xstar_b)))
display(Math(r'\text{(b) Eigenvalues: }' + sym.latex(list(Ab.eigenvals().keys()))))


#| label: fig-ex4
#| fig-cap: "Phase portraits for Example 4. Left (part a): stable node at $\\mathbf{x}^*=(5/2,1)^T$; all orbits converge to the equilibrium. Right (part b): saddle at $\\mathbf{x}^*=(1,-1)^T$; the stable manifold (tomato) and unstable manifold (steelblue) intersect at the equilibrium."
#| code-fold: true
#| code-summary: "Show the code"

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# ---- Part (a) ----
ax = axes[0]
xs_a = np.array([5/2, 1])
v1a = np.array([1, 0]); v2a = np.array([-1, 1])
t_f = np.linspace(0, 3, 500); t_b = np.linspace(0, -1, 200)

ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
for c1v, c2v in [(2,0),(-2,0),(0,2),(0,-2),(1,1),(1,-1),(-1,1),(-1,-1),(3,1),(-3,1)]:
    for tr in [t_f, t_b]:
        xc = xs_a[0] + c1v*v1a[0]*np.exp(-2*tr) + c2v*v2a[0]*np.exp(-3*tr)
        yc = xs_a[1] + c1v*v1a[1]*np.exp(-2*tr) + c2v*v2a[1]*np.exp(-3*tr)
        mask = (np.abs(xc) < 6) & (np.abs(yc) < 5)
        ax.plot(xc[mask], yc[mask], color='steelblue', lw=1.0, alpha=0.45)
ax.plot(*xs_a, 'ko', ms=7, zorder=5)
ax.annotate(r'$\mathbf{x}^*=(5/2,1)$', xy=xs_a, xytext=(3.2, 2.0), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.set_xlim(-4, 6); ax.set_ylim(-4, 5)
ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title('Part (a): stable node at non-origin equilibrium', fontsize=10)

# ---- Part (b) ----
ax = axes[1]
xs_b = np.array([1, -1])
v1b = np.array([1, -3]); v2b = np.array([1, -1])
t_f2 = np.linspace(0,  1.2, 500)
t_b2 = np.linspace(0, -1.2, 500)

ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
# Stable manifold (c2=0)
for c1v in [1.5, -1.5]:
    for tr in [t_f2, t_b2]:
        xc = xs_b[0] + c1v*v1b[0]*np.exp(-tr)
        yc = xs_b[1] + c1v*v1b[1]*np.exp(-tr)
        mask = (np.abs(xc) < 7) & (np.abs(yc) < 7)
        ax.plot(xc[mask], yc[mask], color='tomato', lw=2.0, alpha=0.85)
# Unstable manifold (c1=0)
for c2v in [0.8, -0.8]:
    for tr in [t_f2, t_b2]:
        xc = xs_b[0] + c2v*v2b[0]*np.exp(tr)
        yc = xs_b[1] + c2v*v2b[1]*np.exp(tr)
        mask = (np.abs(xc) < 7) & (np.abs(yc) < 7)
        ax.plot(xc[mask], yc[mask], color='steelblue', lw=2.0, alpha=0.85)
# Generic orbits
for c1v, c2v in [(1,0.4),(1,-0.4),(-1,0.4),(-1,-0.4)]:
    for tr in [t_f2, t_b2]:
        xc = xs_b[0] + c1v*v1b[0]*np.exp(-tr) + c2v*v2b[0]*np.exp(tr)
        yc = xs_b[1] + c1v*v1b[1]*np.exp(-tr) + c2v*v2b[1]*np.exp(tr)
        mask = (np.abs(xc) < 7) & (np.abs(yc) < 7)
        ax.plot(xc[mask], yc[mask], color='gray', lw=1.0, alpha=0.4)

ax.plot(*xs_b, 'ko', ms=7, zorder=5)
ax.annotate(r'$\mathbf{x}^*=(1,-1)$', xy=xs_b, xytext=(2.3, 0.5), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8))
ax.set_xlim(-5, 7); ax.set_ylim(-7, 5)
ax.set_xlabel(r'$x$', fontsize=13); ax.set_ylabel(r'$y$', fontsize=13)
ax.set_title('Part (b): saddle at non-origin equilibrium', fontsize=10)
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color='tomato',    lw=2, label='Stable manifold ($c_2=0$)'),
           Line2D([0],[0], color='steelblue', lw=2, label='Unstable manifold ($c_1=0$)'),
           Line2D([0],[0], color='gray',      lw=1, label='Other orbits')]
ax.legend(handles=handles, fontsize=8)
plt.tight_layout()
plt.show()


#| label: ex5-sympy

t = sym.Symbol('t')

A5 = sym.Matrix([[2, -1], [1, 0]])
f5 = sym.Matrix([sym.exp(t), 0])

Phi5 = sym.Matrix([[(1+t)*sym.exp(t), -t*sym.exp(t)],
                   [t*sym.exp(t),      (1-t)*sym.exp(t)]])

display(Math(r'\det\Phi = ' + sym.latex(sym.simplify(Phi5.det()))))

Phi5_inv = sym.simplify(Phi5.inv())
display(Math(r'\Phi^{-1}(t) = ' + sym.latex(Phi5_inv)))

integrand = sym.simplify(Phi5_inv * f5)
display(Math(r'\Phi^{-1}\mathbf{f} = ' + sym.latex(integrand)))

integral = sym.integrate(integrand, t)
display(Math(r'\int\Phi^{-1}\mathbf{f}\,dt = ' + sym.latex(sym.simplify(integral))))

xp = sym.simplify(Phi5 * integral)
display(Math(r'\mathbf{x}_p = ' + sym.latex(xp)))


#| label: ex6-sympy

A6 = sym.Matrix([[-5, 3], [2, -10]])
f6 = sym.Matrix([sym.exp(-t), 2*sym.exp(-t)])

display(Math(r'\text{Eigenvalues of }A: '
            + sym.latex(list(A6.eigenvals().keys()))))

# Trial: x_p = a*e^{-t}; solve (A+I)a = -(1,2)^T
rhs6 = sym.Matrix([-1, -2])
a_vec = (A6 + sym.eye(2)).solve(rhs6)
display(Math(r'\mathbf{a} = ' + sym.latex(a_vec)))

xp6 = a_vec * sym.exp(-t)
display(Math(r'\mathbf{x}_p = ' + sym.latex(xp6)))

# Verify
lhs_check = sym.diff(xp6, t) - A6*xp6 - f6
display(Math(r'\mathbf{x}_p\prime - A\mathbf{x}_p - \mathbf{f} = '
            + sym.latex(sym.simplify(lhs_check))
            + r'\quad\checkmark'))


#| label: fig-ex6
#| fig-cap: "Time-series plot for Example 6 showing $x(t)$ (steelblue) and $y(t)$ (tomato) for the particular solution $\\mathbf{x}_p = (\\tfrac{1}{2}, \\tfrac{1}{3})^T e^{-t}$ (dashed) and the complete solution with $c_1=1$, $c_2=0$ (solid). Both decay to zero as $t\\to+\\infty$."
#| code-fold: true
#| code-summary: "Show the code"

t_vals = np.linspace(0, 3, 500)

# Particular solution
xp_x = 0.5 * np.exp(-t_vals)
xp_y = (1/3) * np.exp(-t_vals)

# Complete solution: c1=1, c2=0
x_full = 3*np.exp(-4*t_vals) + 0.5*np.exp(-t_vals)
y_full = 1*np.exp(-4*t_vals) + (1/3)*np.exp(-t_vals)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_vals, x_full, color='steelblue', lw=2.2,
        label=r'$x(t)$, full solution ($c_1=1,c_2=0$)')
ax.plot(t_vals, y_full, color='tomato', lw=2.2,
        label=r'$y(t)$, full solution ($c_1=1,c_2=0$)')
ax.plot(t_vals, xp_x, color='steelblue', lw=1.4, ls='--',
        label=r'$x_p = (1/2)e^{-t}$ (particular)')
ax.plot(t_vals, xp_y, color='tomato', lw=1.4, ls='--',
        label=r'$y_p = (1/3)e^{-t}$ (particular)')
ax.axhline(0, color='gray', lw=0.8)
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel('Component', fontsize=12)
ax.set_title(r'Example 6: exponential forcing, all solutions decay to $\mathbf{0}$',
             fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| label: session-info

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}'
                for m in globals().values()
                if getattr(m, '__version__', None)))
