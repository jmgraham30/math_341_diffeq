#| code-fold: true
#| code-summary: "Show the code"

import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import airy
from math import factorial
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-partial-sums
#| fig-cap: "Partial sums $S_N(t) = \\sum_{n=0}^N t^n$ converging to $f(t)=1/(1-t)$ (black dashed) for $|t|<1$. Each successive partial sum adds one more term and fits the true function better. Outside $|t|\\geq 1$ the partial sums diverge — the radius of convergence $r=1$ is sharp."

t_plot = np.linspace(-0.95, 0.95, 400)
f_exact = 1/(1 - t_plot)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: partial sums
colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
for N_ps, color in zip([0, 1, 2, 3, 5, 10], colors):
    S_N = sum(t_plot**n for n in range(N_ps+1))
    axes[0].plot(t_plot, S_N, color=color, lw=1.8, label=f'$S_{{{N_ps}}}$')
axes[0].plot(t_plot, f_exact, 'k--', lw=2.5, label=r'$1/(1-t)$ (exact)')
axes[0].set_ylim(-4, 6); axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$S_N(t)$')
axes[0].set_title(r'Partial sums of $\sum t^n$ converging to $1/(1-t)$')
axes[0].legend(fontsize=8.5, ncol=2)

# Right: error vs N at t=0.5
N_arr = np.arange(1, 25)
errors = [abs(sum(0.5**n for n in range(N+1)) - 2.0) for N in N_arr]
axes[1].semilogy(N_arr, errors, 'o-', color='steelblue', lw=2, markersize=6)
axes[1].set_xlabel('Number of terms $N$')
axes[1].set_ylabel('$|S_N(0.5) - f(0.5)|$')
axes[1].set_title('Exponential convergence at $t=0.5$')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-first-order
#| fig-cap: "Series solution of $(t-3)y'+2y=0$, $y(0)=1/9$. Left: partial sums converging to the exact solution $y=1/(3-t)^2$ (black dashed) for $|t|<3$. The series diverges outside the radius of convergence $r=3$ (dashed vertical lines). Right: partial-sum errors at $t=1.5$ decreasing toward zero, demonstrating convergence."

N_terms_list = [3, 5, 8, 15, 30]
t_plot = np.linspace(-2.8, 2.8, 500)
y_exact = 1/(3 - t_plot)**2 / 9  # a0=1/9 so y=1/(3-t)^2

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(N_terms_list)))
for N_ps, color in zip(N_terms_list, colors):
    coeffs = [(n+1)/3**n / 9 for n in range(N_ps)]  # a_n = (n+1)/3^n * a0, a0=1/9
    y_ps = sum(coeffs[n]*t_plot**n for n in range(N_ps))
    axes[0].plot(t_plot, np.clip(y_ps, -1, 5), color=color, lw=1.8, label=f'$N={N_ps}$')

axes[0].plot(t_plot, y_exact, 'k--', lw=2.5, label=r'$y=1/(3-t)^2$ (exact)')
axes[0].axvline(3, color='gray', ls=':', lw=1.5, label='$r=3$')
axes[0].axvline(-3, color='gray', ls=':', lw=1.5)
axes[0].set_ylim(-0.5, 5); axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$y(t)$')
axes[0].set_title(r'$(t-3)y\prime+2y=0$: partial sums')
axes[0].legend(fontsize=8)

# Error at t=1.5
t_val = 1.5
y_true = 1/(3-t_val)**2 / 9
N_arr = np.arange(1, 40)
errors = []
for N_ps in N_arr:
    y_ps_val = sum((n+1)/3**n/9 * t_val**n for n in range(N_ps))
    errors.append(abs(y_ps_val - y_true))
axes[1].semilogy(N_arr, errors, 'o-', color='steelblue', lw=2, markersize=5)
axes[1].set_xlabel('$N$ (number of terms)')
axes[1].set_ylabel(r'$|S_N(1.5) - y(1.5)|$')
axes[1].set_title('Convergence at $t=1.5$')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-cos-sin-series
#| fig-cap: "Power series partial sums recovering $\\cos t$ (left) and $\\sin t$ (right). Even-indexed terms build $\\cos t$; odd-indexed terms build $\\sin t$. More terms are needed to capture the oscillations accurately over a larger interval."

t_plot = np.linspace(-2*np.pi, 2*np.pi, 500)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (a0, a1), y_exact, name in [
    (axes[0], (1, 0), np.cos(t_plot), r'$\cos t$'),
    (axes[1], (0, 1), np.sin(t_plot), r'$\sin t$'),
]:
    ax.plot(t_plot, y_exact, 'k--', lw=2.5, label=f'Exact {name}', zorder=5)
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, 5))
    for N_terms, color in zip([1, 2, 3, 5, 8], colors):
        a = [0.0]*(2*N_terms+2)
        a[0] = a0; a[1] = a1
        for n_v in range(len(a)-2):
            a[n_v+2] = -a[n_v]/((n_v+2)*(n_v+1))
        y_ps = sum(a[n]*t_plot**n for n in range(len(a)))
        ax.plot(t_plot, np.clip(y_ps, -5, 5), color=color, lw=1.8, label=f'$N={2*N_terms}$ terms')
    ax.set_ylim(-3, 3); ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('$t$'); ax.set_ylabel('$y$')
    ax.set_title(f'Series for {name}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-airy
#| fig-cap: "Airy equation $y''-ty=0$: the two linearly independent series solutions $y_1(t)$ (blue) and $y_2(t)$ (orange) satisfying $y_1(0)=1$, $y_1'(0)=0$ and $y_2(0)=0$, $y_2'(0)=1$. For $t<0$ both solutions oscillate (the ODE has no dissipation and the coefficient $-t>0$); for $t>0$ they grow or decay. Red dots confirm the series against a direct numerical ODE solve."

N_a = 50
a_y1 = [0.0]*N_a; a_y1[0] = 1.0
a_y2 = [0.0]*N_a; a_y2[1] = 1.0
for n_v in range(1, N_a-2):
    a_y1[n_v+2] = a_y1[n_v-1]/((n_v+2)*(n_v+1))
    a_y2[n_v+2] = a_y2[n_v-1]/((n_v+2)*(n_v+1))

t_plot = np.linspace(-4, 2.5, 500)

# Series evaluation (safe range — N=50 terms converge well for |t|<=3)
def eval_series(coeffs, t_arr):
    result = np.zeros_like(t_arr)
    for k, ck in enumerate(coeffs):
        result = result + ck * t_arr**k
    return result

y1_ser = eval_series(a_y1, t_plot)
y2_ser = eval_series(a_y2, t_plot)

# Numerical ODE solve for comparison
def airy_ode(t, y): return [y[1], t*y[0]]
sol1 = solve_ivp(airy_ode, (-4, 2.5), [1.0, 0.0], dense_output=True, max_step=0.01)
sol2 = solve_ivp(airy_ode, (-4, 2.5), [0.0, 1.0], dense_output=True, max_step=0.01)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, y_ser, sol, lbl, color in [
    (axes[0], y1_ser, sol1, r'$y_1$ ($a_0=1$, $a_1=0$)', 'steelblue'),
    (axes[1], y2_ser, sol2, r'$y_2$ ($a_0=0$, $a_1=1$)', 'darkorange'),
]:
    ax.plot(t_plot, np.clip(y_ser, -6, 8), color=color, lw=2.5, label='Series (N=50)')
    t_dots = np.linspace(-3, 2, 20)
    ax.plot(t_dots, sol.sol(t_dots)[0], 'ro', markersize=5, label='Numerical')
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5, ls='--')
    ax.set_xlabel('$t$'); ax.set_ylabel('$y(t)$')
    ax.set_title(f'Airy: {lbl}')
    ax.legend(fontsize=8.5); ax.set_ylim(-6, 8)

plt.suptitle(r"Airy equation $y'' - ty = 0$", fontsize=12)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-legendre
#| fig-cap: "Left: power series partial sums for $y_1$ (even) and $y_2$ (odd) solutions of Legendre's equation with $p=2.5$ (non-integer — infinite series, $r=1$). Right: Legendre polynomials $P_0,\\ldots,P_4$ obtained when $p$ is an integer — the series terminates and gives a polynomial. The recurrence coefficient $(p-n)=0$ at $n=p$ causes early termination."

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Non-integer p: series solution
p_val = 2.5
N_leg = 40
t_plot = np.linspace(-0.95, 0.95, 400)

a_even = [0.0]*N_leg; a_even[0] = 1.0
a_odd  = [0.0]*N_leg; a_odd[1]  = 1.0
for n_v in range(N_leg-2):
    coeff = -(p_val-n_v)*(p_val+n_v+1)/((n_v+2)*(n_v+1))
    a_even[n_v+2] = coeff*a_even[n_v]
    a_odd [n_v+2] = coeff*a_odd[n_v]

y_even = sum(a_even[k]*t_plot**k for k in range(N_leg))
y_odd  = sum(a_odd[k]*t_plot**k  for k in range(N_leg))

# Numerical verification
def legendre_ode(t, y):
    if abs(1-t**2) < 1e-10: return [0.0, 0.0]
    return [y[1], (2*t*y[1] - p_val*(p_val+1)*y[0])/(1-t**2)]
sol_e = solve_ivp(legendre_ode, (0, 0.9), [1.0, 0.0], dense_output=True, max_step=0.005)
sol_o = solve_ivp(legendre_ode, (0, 0.9), [0.0, 1.0], dense_output=True, max_step=0.005)

t_pos = np.linspace(0, 0.9, 200)
axes[0].plot(t_plot, y_even, color='steelblue', lw=2, label=f'$y_{{\\rm even}}$, $p={p_val}$')
axes[0].plot(t_plot, y_odd,  color='darkorange', lw=2, label=f'$y_{{\\rm odd}}$, $p={p_val}$')
axes[0].plot(t_pos, sol_e.sol(t_pos)[0], 'r:', lw=1.5, label='Numerical $y_{\\rm even}$')
axes[0].plot(t_pos, sol_o.sol(t_pos)[0], 'g:', lw=1.5, label='Numerical $y_{\\rm odd}$')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$y$')
axes[0].set_title(f'Legendre series ($p={p_val}$, $r=1$)')
axes[0].legend(fontsize=8)

# Integer p: Legendre polynomials (terminating series)
t_leg = np.linspace(-1, 1, 300)
colors_leg = plt.cm.tab10(np.linspace(0, 0.5, 5))
for p_int, color in zip([0, 1, 2, 3, 4], colors_leg):
    N_lp = 20
    if p_int % 2 == 0:
        a = [0.0]*N_lp; a[0] = 1.0
    else:
        a = [0.0]*N_lp; a[1] = 1.0
    for n_v in range(N_lp-2):
        a[n_v+2] = -(p_int-n_v)*(p_int+n_v+1)/((n_v+2)*(n_v+1)) * a[n_v]
    y_ps = sum(a[k]*t_leg**k for k in range(N_lp))
    # Normalize: P_p(1) = 1
    val_at_1 = sum(a[k] for k in range(N_lp))
    if abs(val_at_1) > 1e-12:
        y_ps = y_ps / val_at_1
    axes[1].plot(t_leg, y_ps, color=color, lw=2, label=f'$P_{{{p_int}}}(t)$')

axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$P_p(t)$')
axes[1].set_title('Legendre polynomials $P_0,\\ldots,P_4$ (terminating series)')
axes[1].legend(fontsize=8.5)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

t_sym = sym.Symbol('t')
y_sym = sym.Function('y')

print("SymPy series solutions to key equations:\n")

# y'' + y = 0
ode1 = sym.Eq(y_sym(t_sym).diff(t_sym, 2) + y_sym(t_sym), 0)
sol1 = sym.dsolve(ode1, y_sym(t_sym))
print("y''+y=0:")
display(Math(sym.latex(sol1)))
print()

# y' - y = 0
ode2 = sym.Eq(y_sym(t_sym).diff(t_sym) - y_sym(t_sym), 0)
sol2 = sym.dsolve(ode2, y_sym(t_sym))
print("y'-y=0:")
display(Math(sym.latex(sol2)))
print()

# First few terms of Airy series via Taylor expansion
t_s = sym.Symbol('t')
y_s = sym.Function('y')
# For Airy, use explicit series representation via sympy.functions
airy_sym = sym.airyai(t_s)
airy_series = sym.series(airy_sym, t_s, 0, 10)
print("Ai(t) Taylor series about t=0:")
display(Math(sym.latex(airy_series)))


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
