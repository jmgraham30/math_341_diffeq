import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

H = lambda tv, a: np.where(np.asarray(tv, dtype=float) >= a, 1.0, 0.0)

s_v = sym.Symbol('s', positive=True)
t_v = sym.Symbol('t', positive=True)

X_ex24 = sym.Integer(3) / (s_v * (s_v**2 + 9))
x_sym  = sym.inverse_laplace_transform(X_ex24, s_v, t_v)
x_sym_simplified = sym.simplify(x_sym)

print("SymPy inverse transform of 3/[s(s²+9)]:")
display(Math(r"\mathcal{L}^{-1}\!\left[\frac{3}{s(s^2+9)}\right] = " + sym.latex(x_sym_simplified)))

t_plot = np.linspace(0, 4, 400)
x_conv = (1/3) * (1 - np.cos(3*t_plot))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_plot, x_conv, color='steelblue', lw=2.5,
        label=r'$\frac{1}{3}(1-\cos 3t)$ from convolution')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r'$\mathcal{L}^{-1}[3/(s(s^2+9))] = \frac{1}{3}(1-\cos 3t)$')
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()

t_plot = np.linspace(0, 4, 2000)
a = 2.0

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ['steelblue', 'darkorange', 'seagreen']
eps_vals = [1.0, 0.5, 0.2]

for eps, color in zip(eps_vals, colors):
    f_eps = np.where((t_plot > a - eps/2) & (t_plot < a + eps/2), 1/eps, 0.0)
    ax.plot(t_plot, f_eps, color=color, lw=2.5,
            label=rf'$\varepsilon = {eps}$, height $= {1/eps:.1f}$')
    ax.fill_between(t_plot, 0, f_eps, alpha=0.15, color=color)

ax.axvline(a, color='gray', ls='--', lw=1.2, label='$t=a=2$')
ax.set_xlabel('$t$'); ax.set_ylabel(r'$f_\varepsilon(t)$')
ax.set_title(r'Rectangular impulse $f_\varepsilon(t)$ approaching $\delta_2(t)$ as $\varepsilon \to 0$')
ax.legend(fontsize=9); ax.set_ylim(-0.2, 6)
ax.set_xlim(0, 4)
plt.tight_layout()
plt.show()

t_plot = np.linspace(0, 10, 600)
x_anal = H(t_plot, 2) * (1 - np.exp(-(t_plot - 2)))

def ode_326(tv, y):
    # x'' + x' = delta_2(t) approximated by a narrow pulse
    eps = 0.05
    f = (1/eps) if abs(tv - 2.0) < eps/2 else 0.0
    return [y[1], f - y[1]]

sol_326 = solve_ivp(ode_326, (0, 10), [0.0, 0.0],
                   dense_output=True, max_step=0.01)
t_dots = np.linspace(0.1, 10, 30)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_plot, x_anal, color='steelblue', lw=2.5,
        label=r'$x(t) = H(t-2)(1-e^{-(t-2)})$')
ax.plot(t_dots, sol_326.sol(t_dots)[0], 'ro', markersize=5,
        label='Numerical (narrow-pulse approximation)')
ax.axvline(2, color='gray', ls='--', lw=1.2, label='$t=2$ (impulse)')
ax.axhline(1, color='gray', ls=':', lw=1.2, label='Limiting value $x=1$')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"Example 3.26: $x''+x'=\delta_2(t)$, $x(0)=x'(0)=0$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

t0 = 2.0
t_plot = np.linspace(0, 10, 600)
x_anal = H(t_plot, t0) * 0.5 * np.exp(-(t_plot - t0)) * np.sin(2*(t_plot - t0))

def ode_328(tv, y):
    eps = 0.05
    f = (1/eps) if abs(tv - t0) < eps/2 else 0.0
    return [y[1], f - 2*y[1] - 5*y[0]]

sol_328 = solve_ivp(ode_328, (0, 10), [0.0, 0.0],
                   dense_output=True, max_step=0.01)
t_dots = np.linspace(0.1, 10, 35)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_plot, x_anal, color='steelblue', lw=2.5,
        label=r'$\frac{1}{2}H(t-2)e^{-(t-2)}\sin 2(t-2)$')
ax.plot(t_dots, sol_328.sol(t_dots)[0], 'ro', markersize=5,
        label='Numerical check')
ax.axvline(t0, color='gray', ls='--', lw=1.2, label=f'$t_0={t0}$ (impulse)')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"Example 3.28: $x''+2x'+5x=\delta_2(t)$, $x(0)=x'(0)=0$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()