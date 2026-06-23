import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

t_plot = np.linspace(0, 5, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: damped integrand for two values of s
x_func = np.sin(2*t_plot)
for s_val, color, lbl in [(0.5, 'steelblue', '$s=0.5$'), (2.0, 'crimson', '$s=2.0$')]:
    damp = np.exp(-s_val*t_plot)
    integrand = x_func * damp
    axes[0].plot(t_plot, integrand, color=color, lw=2, label=f'$\\sin(2t)e^{{-{s_val}t}}$ ({lbl})')
    axes[0].fill_between(t_plot, 0, integrand, where=(integrand>0), alpha=0.12, color=color)
    axes[0].fill_between(t_plot, 0, integrand, where=(integrand<0), alpha=0.12, color=color)

axes[0].plot(t_plot, x_func, 'k--', lw=1.2, alpha=0.5, label='$\\sin(2t)$ (undamped)')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('integrand')
axes[0].set_title(r'Integrand $\sin(2t)\,e^{-st}$ for two values of $s$')
axes[0].legend(fontsize=8.5)

# Right: X(s) = 2/(s^2+4)
s_arr = np.linspace(0.01, 6, 400)
X_s   = 2/(s_arr**2 + 4)
axes[1].plot(s_arr, X_s, color='seagreen', lw=2.5, label=r'$X(s)=\frac{2}{s^2+4}$')
for s_val, color in [(0.5, 'steelblue'), (2.0, 'crimson')]:
    val = 2/(s_val**2+4)
    axes[1].plot(s_val, val, 'o', color=color, markersize=9,
                 label=f'$X({s_val})={val:.3f}$ (area under left plot)')
axes[1].set_xlabel('$s$'); axes[1].set_ylabel('$X(s)$')
axes[1].set_title(r'$\mathcal{L}[\sin 2t](s) = 2/(s^2+4)$')
axes[1].legend(fontsize=8.5); axes[1].set_ylim(0, 0.55)

plt.suptitle('How the Laplace transform works', fontsize=12)
plt.tight_layout()
plt.show()

s_v = sym.Symbol('s')
t_v = sym.Symbol('t', positive=True)

# Demonstrate partial fractions with SymPy
examples_inv = [
    (r"\frac{2s+9}{(s+1)(s+3)}",          (2*s_v+9)/((s_v+1)*(s_v+3))),
    (r"\frac{3}{s^2-4}",                    3/(s_v**2-4)),
    (r"\frac{2s+3}{s^2+2s+5}",             (2*s_v+3)/(s_v**2+2*s_v+5)),
    (r"\frac{1}{s(s+2)}",                   1/(s_v*(s_v+2))),
]

for label, expr in examples_inv:
    pf = sym.apart(expr, s_v)
    inv = sym.inverse_laplace_transform(expr, s_v, t_v)
    print(f"$X(s) = {label}$")
    display(Math(r"\text{PF: } " + sym.latex(pf)))
    display(Math(r"x(t) = " + sym.latex(sym.simplify(inv))))
    print()

t_plot = np.linspace(0, 4, 300)
x_exact = 3.5*np.exp(-t_plot) - 1.5*np.exp(-3*t_plot)

def ode_ex1(t, y): return [y[1], -4*y[1] - 3*y[0]]
sol = solve_ivp(ode_ex1, (0, 4), [2.0, 1.0], dense_output=True, max_step=0.02)
t_dots = np.linspace(0, 4, 20)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_plot, x_exact, color='steelblue', lw=2.5, label=r'$\frac{7}{2}e^{-t}-\frac{3}{2}e^{-3t}$')
ax.plot(t_dots, sol.sol(t_dots)[0], 'ro', markersize=6, label='Numerical (solve_ivp)')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"$x''+4x'+3x=0$, $x(0)=2$, $x'(0)=1$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

t_plot = np.linspace(-0.3, 8, 600)
fig, axes = plt.subplots(2, 1, figsize=(10, 7))

# Piecewise function
def f_piece(tv):
    return np.where(tv<0, 0,
           np.where(tv<2, 3,
           np.where(tv<3, 4,
           np.where(tv<6, 2, 0))))

axes[0].step(t_plot, f_piece(t_plot), where='post', color='k', lw=2.5,
             label='$f(t)$', zorder=5)

# Individual Heaviside contributions
H = lambda tv, a: np.where(tv >= a, 1.0, 0.0)
contribs = [
    (3*H(t_plot,0),      'steelblue',  '+3$H(t)$'),
    (1*H(t_plot,2),      'darkorange', '+1$H(t-2)$'),
    (-2*H(t_plot,3),     'seagreen',   '-2$H(t-3)$'),
    (-2*H(t_plot,6),     'crimson',    '-2$H(t-6)$'),
]
running = np.zeros_like(t_plot)
for contrib, color, lbl in contribs:
    axes[0].fill_between(t_plot, running, running+contrib, alpha=0.25, color=color)
    running += contrib

for _, color, lbl in contribs:
    axes[0].plot([], [], color=color, linewidth=8, alpha=0.4, label=lbl)

axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$f(t)$')
axes[0].set_title('Piecewise function as Heaviside steps')
axes[0].legend(fontsize=9, ncol=4)
axes[0].set_xlim(-0.3, 8); axes[0].set_ylim(-0.5, 5)

# Laplace transform F(s)
s_arr = np.linspace(0.01, 4, 400)
F_s = (3 + np.exp(-2*s_arr) - 2*np.exp(-3*s_arr) - 2*np.exp(-6*s_arr))/s_arr
axes[1].plot(s_arr, F_s, color='seagreen', lw=2.5)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$s$'); axes[1].set_ylabel('$F(s)$')
axes[1].set_title(r'Laplace transform $F(s) = \frac{1}{s}(3+e^{-2s}-2e^{-3s}-2e^{-6s})$')

plt.tight_layout()
plt.show()

s_v = sym.Symbol('s')
t_v = sym.Symbol('t', positive=True)

print("=== Shift property examples (SymPy verification) ===")
for f_t, lbl in [(t_v*sym.exp(-2*t_v), 'te^{-2t}'),
                  (sym.exp(-t_v)*sym.sin(3*t_v), 'e^{-t}sin(3t)'),
                  (sym.exp(2*t_v)*sym.cos(3*t_v), 'e^{2t}cos(3t)')]:
    F = sym.laplace_transform(f_t, t_v, s_v, noconds=True)
    display(Math(r"\mathcal{L}[" + lbl + r"] = " + sym.latex(sym.simplify(F))))

t_plot = np.linspace(0, 8, 600)
H = lambda tv, a: np.where(tv >= a, 1.0, 0.0)

x_exact = 0.5*(1 - np.exp(-2*t_plot)) - 0.5*H(t_plot, 3)*(1 - np.exp(-2*(t_plot-3)))
f_plot  = H(t_plot, 0) - H(t_plot, 3)

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

axes[0].step(t_plot, f_plot, where='post', color='crimson', lw=2.5)
axes[0].set_ylabel('$f(t)$'); axes[0].set_ylim(-0.2, 1.4)
axes[0].set_title('Forcing $f(t) = H(t)-H(t-3)$')
axes[0].fill_between(t_plot, 0, f_plot, alpha=0.2, color='crimson')

axes[1].plot(t_plot, x_exact, color='steelblue', lw=2.5, label='Analytical (Laplace)')
def rhs(tv, y): return [(-H(tv,0)-H(tv,3))*0 + (1 if tv<3 else 0) - 2*y[0]]
sol_num = solve_ivp(lambda tv, y: [(1 if tv<3 else 0) - 2*y[0]],
                   (0, 8), [0.0], dense_output=True, max_step=0.02)
t_dots = np.linspace(0, 8, 25)
axes[1].plot(t_dots, sol_num.sol(t_dots)[0], 'ro', markersize=6, label='Numerical')
axes[1].axhline(0.5, color='gray', ls=':', lw=1.2, label='Steady state $x=0.5$')
axes[1].axvline(3, color='gray', ls='--', lw=1.2, alpha=0.7)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x(t)$')
axes[1].set_title("Solution via Laplace transform")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

# Verify a selection of table entries with SymPy
s_v = sym.Symbol('s', positive=True)
t_v = sym.Symbol('t', positive=True)
k_v = sym.Symbol('k', positive=True)
a_v = sym.Symbol('a', positive=True)

entries = [
    (sym.sin(k_v*t_v), f'sin(kt)',     k_v/(s_v**2+k_v**2)),
    (sym.cos(k_v*t_v), f'cos(kt)',     s_v/(s_v**2+k_v**2)),
    (sym.exp(a_v*t_v)*sym.sin(k_v*t_v), 'e^{at}sin(kt)',
     k_v/((s_v-a_v)**2+k_v**2)),
    (t_v*sym.sin(k_v*t_v), 't*sin(kt)',
     2*k_v*s_v/(s_v**2+k_v**2)**2),
]

print("Spot-checking table entries via SymPy:")
for f_t, name, F_expected in entries:
    F_computed = sym.laplace_transform(f_t, t_v, s_v, noconds=True)
    F_diff = sym.simplify(F_computed - F_expected)
    status = "✓" if F_diff == 0 else f"Diff={F_diff}"
    print(f"  {name}: {status}")