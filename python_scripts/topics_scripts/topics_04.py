import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

t_plot = np.linspace(0, 1.5, 400)
x_exact = np.exp(t_plot**2 / 2)

iterates = [
    np.ones_like(t_plot),
    1 + t_plot**2/2,
    1 + t_plot**2/2 + t_plot**4/8,
    1 + t_plot**2/2 + t_plot**4/8 + t_plot**6/48,
    1 + t_plot**2/2 + t_plot**4/8 + t_plot**6/48 + t_plot**8/384,
]
labels = ['$x_0 = 1$',
          r'$x_1 = 1+\frac{t^2}{2}$',
          r'$x_2 = 1+\frac{t^2}{2}+\frac{t^4}{8}$',
          r'$x_3$ (4 terms)',
          r'$x_4$ (5 terms)']
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(iterates)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for xn, label, color in zip(iterates, labels, colors):
    axes[0].plot(t_plot, xn, color=color, lw=1.8, label=label)
axes[0].plot(t_plot, x_exact, 'k--', lw=2.5, label=r'Exact $e^{t^2/2}$')
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x(t)$')
axes[0].set_title('Picard iterates converging to $e^{t^2/2}$')
axes[0].set_ylim(0.8, 4); axes[0].legend(fontsize=8)

errors = [np.max(np.abs(xn - x_exact)) for xn in iterates]
axes[1].semilogy(range(len(errors)), errors, 'o-', color='steelblue',
                 lw=2, markersize=8)
axes[1].set_xlabel('Iteration $n$')
axes[1].set_ylabel('Max error on $[0, 1.5]$')
axes[1].set_title('Convergence of Picard iteration (log scale)')
axes[1].set_xticks(range(len(errors)))

plt.tight_layout()
plt.show()

t_plot = np.linspace(-0.3, 2.5, 500)

fig, ax = plt.subplots(figsize=(8, 5))

# Trivial solution
ax.plot(t_plot, np.zeros_like(t_plot), color='black', lw=2.5, ls='--',
        label='$x(t) \equiv 0$')

# Family of solutions
a_vals = [0.0, 0.4, 0.8, 1.2, 1.8]
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(a_vals)))
for a, color in zip(a_vals, colors):
    x_a = np.where(t_plot >= a, ((t_plot - a)/3)**3, 0.0)
    lbl = f'$x_{{a}}(t)$, $a={a}$' if a == 0.0 else f'$a={a}$'
    ax.plot(t_plot, x_a, color=color, lw=2, label=lbl)

ax.plot(0, 0, 'ko', markersize=10, zorder=5, label='IC $(0,0)$')
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"Non-uniqueness: $x' = x^{2/3}$, $x(0)=0$ — infinitely many solutions")
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(-0.2, 1.5)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
p_vals = [2, 3, 4, 5]
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(p_vals)))
x0 = 1.0

for p, color in zip(p_vals, colors):
    t_star = 1.0 / ((p-1) * x0**(p-1))
    t_plot = np.linspace(0, t_star * 0.97, 400)
    x_sol = x0 / (1 - (p-1)*x0**(p-1)*t_plot)**(1/(p-1))
    ax.plot(t_plot, x_sol, color=color, lw=2.2, label=f'$p={p}$, $t^*={t_star:.3f}$')
    ax.axvline(t_star, color=color, ls=':', lw=1.2, alpha=0.8)

ax.set_ylim(0, 15)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"Finite-time blow-up: $x' = x^p$, $x(0)=1$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

L_val = 1.0   # Lipschitz constant for f(t,x) = x
t_plot = np.linspace(0, 3, 400)

x0_base = 1.0
delta_vals = [0.1, 0.05, 0.01]
colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(delta_vals)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

x_base = x0_base * np.exp(t_plot)
axes[0].plot(t_plot, x_base, color='black', lw=2.5, label=f'$x_0={x0_base}$ (base)')
axes[0].fill_between(t_plot,
                     (x0_base - 0.1)*np.exp(t_plot),
                     (x0_base + 0.1)*np.exp(t_plot),
                     alpha=0.15, color='steelblue', label='$\\pm\\delta e^t$ tube ($\\delta=0.1$)')

for delta, color in zip(delta_vals, colors):
    x_pert = (x0_base + delta) * np.exp(t_plot)
    axes[0].plot(t_plot, x_pert, color=color, lw=1.8, ls='--',
                 label=f'$x_0 + \\delta$, $\\delta={delta}$')

axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x(t)$')
axes[0].set_title(r"Continuous dependence: $x'=x$")
axes[0].legend(fontsize=8); axes[0].set_ylim(0, 25)

for delta, color in zip(delta_vals, colors):
    diff = np.abs((x0_base + delta)*np.exp(t_plot) - x_base)
    gronwall_bound = delta * np.exp(L_val * t_plot)
    axes[1].semilogy(t_plot, diff, color=color, lw=2, label=f'$\\delta={delta}$')
    axes[1].semilogy(t_plot, gronwall_bound, color=color, lw=1, ls=':')

axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$|x(t) - \\tilde{x}(t)|$')
axes[1].set_title('Solution difference vs. Gronwall bound (dotted)')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

r_val = 3.0
f_log = lambda t, x: [r_val * x[0] * (1 - x[0])]

x0_base = 0.1
epsilon = 1e-4

t_span = (0, 8)
t_eval = np.linspace(0, 8, 800)

sol_base = solve_ivp(f_log, t_span, [x0_base],       t_eval=t_eval, max_step=0.01)
sol_pert = solve_ivp(f_log, t_span, [x0_base+epsilon], t_eval=t_eval, max_step=0.01)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(t_eval, sol_base.y[0], color='steelblue', lw=2,   label=f'$x_0={x0_base}$')
axes[0].plot(t_eval, sol_pert.y[0], color='crimson',   lw=1.5, ls='--',
             label=f'$x_0+\\varepsilon$, $\\varepsilon=10^{{-4}}$')
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x(t)$')
axes[0].set_title(r"Nonlinear ODE $x'=3x(1-x)$: two nearby solutions")
axes[0].legend(fontsize=9)

diff = np.abs(sol_pert.y[0] - sol_base.y[0])
axes[1].semilogy(t_eval, diff + 1e-16, color='seagreen', lw=2)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$|x(t)-\\tilde{x}(t)|$')
axes[1].set_title('Separation between solutions (log scale)')

plt.tight_layout()
plt.show()