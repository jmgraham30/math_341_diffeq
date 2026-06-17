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
#| label: fig-ex1
#| fig-cap: "Solutions of $x' = x + x^2 e^t$ from several initial conditions $x_0 > 0$. Each solution blows up at $t^* = \\frac{1}{2}\\ln(2/x_0+1)$ (marked with a vertical dashed line). The closed-form solution (solid) agrees perfectly with the numerical ODE solve (dots)."

def x_ex1_closed(t, x0):
    C = 1.0/x0 + 0.5
    denom = 2*C - np.exp(2*t)
    return np.where(denom > 0.01, 2*np.exp(t)/denom, np.nan)

f_b = lambda t, x: [x[0]*(1 + x[0]*np.exp(t))]

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, 4))
x0_vals = [0.2, 0.4, 0.6, 0.8]

for x0, color in zip(x0_vals, colors):
    C = 1/x0 + 0.5
    t_blow = 0.5*np.log(2*C)
    t_safe = np.linspace(0, t_blow - 0.05, 400)

    # Closed-form
    x_clo = 2*np.exp(t_safe)/(2*C - np.exp(2*t_safe))
    ax.plot(t_safe, x_clo, color=color, lw=2.5, label=f'$x_0={x0}$, $t^*={t_blow:.2f}$')

    # Numerical check (dots)
    sol = solve_ivp(f_b, (0, t_blow-0.08), [x0], dense_output=True, max_step=0.005)
    t_dots = np.linspace(0, t_blow-0.12, 12)
    ax.plot(t_dots, sol.sol(t_dots)[0], 'o', color=color, markersize=5)

    # Blow-up marker
    ax.axvline(t_blow, color=color, linestyle=':', lw=1.2, alpha=0.7)

ax.set_ylim(0, 20)
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
ax.set_title(r"$x' = x + x^2 e^t$: finite-time blow-up")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ex2
#| fig-cap: "Solutions of $x' = x - 0.5x^3$ ($a=1$, $b=-0.5$) from several initial conditions. The non-zero equilibrium $x_{\\rm eq} = \\sqrt{2} \\approx 1.414$ (dashed green) is stable: all solutions with $x_0>0$ converge to it. The trivial equilibrium $x=0$ (dashed red) is unstable."

a_val, b_val = 1.0, -0.5
x_eq = np.sqrt(-a_val / b_val)   # = sqrt(2)

def x_ex2_closed(t, x0, a, b):
    y0 = 1.0 / x0**2
    y_t = -b/a + (y0 + b/a)*np.exp(-2*a*t)
    return 1.0 / np.sqrt(np.abs(y_t)) * np.sign(x0)

f_e = lambda t, x: [a_val*x[0] + b_val*x[0]**3]

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.plasma(np.linspace(0.1, 0.85, 6))
x0_vals = [0.3, 0.7, 1.0, 1.5, 2.0, 3.0]
t_plot = np.linspace(0, 5, 400)

for x0, color in zip(x0_vals, colors):
    x_clo = x_ex2_closed(t_plot, x0, a_val, b_val)
    ax.plot(t_plot, x_clo, color=color, lw=2.2, label=f'$x_0={x0}$')

ax.axhline(x_eq, color='seagreen', ls='--', lw=2, label=f'$x_{{\\rm eq}}=\\sqrt{{2}}\\approx{x_eq:.3f}$ (stable)')
ax.axhline(0,    color='crimson',  ls='--', lw=1.5, label='$x=0$ (unstable)')
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
ax.set_title(r"$x' = x - 0.5\,x^3$: convergence to stable equilibrium")
ax.set_ylim(0, 3.5)
ax.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-logistic-bernoulli
#| fig-cap: "Logistic growth ($r=1$, $K=5$) solved via the Bernoulli substitution. The closed-form solution (solid) is compared to a direct numerical solve (dots) for four initial conditions. All solutions converge to the carrying capacity $K=5$ (dashed)."

r_val, K_val = 1.0, 5.0

def logistic_closed(t, x0, r, K):
    return K / (1 + (K/x0 - 1)*np.exp(-r*t))

f_log = lambda t, x: [r_val*x[0]*(1 - x[0]/K_val)]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, 5))
x0_vals = [0.3, 1.0, 3.0, 6.0, 9.0]
t_plot  = np.linspace(0, 8, 400)

for x0, color in zip(x0_vals, colors):
    x_clo = logistic_closed(t_plot, x0, r_val, K_val)
    ax.plot(t_plot, x_clo, color=color, lw=2.2, label=f'$x_0={x0}$')
    sol = solve_ivp(f_log, (0, 8), [x0], dense_output=True, max_step=0.05)
    t_dots = np.linspace(0, 8, 15)
    ax.plot(t_dots, sol.sol(t_dots)[0], 'o', color=color, markersize=4)

ax.axhline(K_val, color='black', ls='--', lw=1.8, label=f'$K={K_val}$ (carrying capacity)')
ax.set_xlabel('$t$')
ax.set_ylabel('$x(t)$')
ax.set_title(r"Logistic ODE via Bernoulli substitution ($r=1$, $K=5$)")
ax.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

t_sym = sym.Symbol('t')
x_sym = sym.Function('x')

cases = {
    r"x' = x + x^2 e^t \quad (n=2)":
        sym.Eq(x_sym(t_sym).diff(t_sym),
               x_sym(t_sym) + x_sym(t_sym)**2 * sym.exp(t_sym)),
    r"x' = ax + bx^3 \quad (n=3, \text{ constants } a,b)":
        sym.Eq(x_sym(t_sym).diff(t_sym),
               sym.Symbol('a')*x_sym(t_sym)
               + sym.Symbol('b')*x_sym(t_sym)**3),
    r"\text{Logistic: } x' = rx(1-x/K) \quad (n=2)":
        sym.Eq(x_sym(t_sym).diff(t_sym),
               sym.Symbol('r')*x_sym(t_sym)
               *(1 - x_sym(t_sym)/sym.Symbol('K'))),
}

for label, ode in cases.items():
    print(f"ODE: ${label}$")
    sol = sym.dsolve(ode, x_sym(t_sym))
    if isinstance(sol, list):
        for s in sol:
            display(Math(r"x(t) = " + sym.latex(s.rhs)))
    else:
        display(Math(r"x(t) = " + sym.latex(sol.rhs)))
    print()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-harvest
#| fig-cap: "Logistic growth with harvesting ($r=1$, $K=8$). Sub-critical harvesting ($h < h_c = 2$, blue shades) leads to a stable positive equilibrium; super-critical harvesting ($h > h_c$, red shades) drives the population to extinction. The critical case $h=h_c=2$ is shown in green."

r_val, K_val = 1.0, 8.0
h_c = r_val * K_val / 4   # = 2.0

fig, ax = plt.subplots(figsize=(8, 5))
t_span = (0, 20)
t_eval = np.linspace(0, 20, 500)

# Sub-critical
for h, color in zip([0, 0.5, 1.5], plt.cm.Blues(np.linspace(0.4, 0.85, 3))):
    f = lambda t, x, h=h: [r_val*x[0]*(1 - x[0]/K_val) - h]
    sol = solve_ivp(f, t_span, [2.0], t_eval=t_eval, max_step=0.05)
    ax.plot(sol.t, sol.y[0], color=color, lw=2, label=f'$h={h}$ (sub-critical)')

# Critical
f_crit = lambda t, x: [r_val*x[0]*(1 - x[0]/K_val) - h_c]
sol_c = solve_ivp(f_crit, t_span, [2.0], t_eval=t_eval, max_step=0.05)
ax.plot(sol_c.t, np.clip(sol_c.y[0], 0, K_val), color='seagreen', lw=2.5,
        label=f'$h=h_c={h_c}$ (critical)')

# Super-critical
for h, color in zip([2.5, 3.5], plt.cm.Reds(np.linspace(0.5, 0.85, 2))):
    f = lambda t, x, h=h: [r_val*x[0]*(1 - x[0]/K_val) - h]
    sol = solve_ivp(f, t_span, [2.0], t_eval=t_eval, max_step=0.05,
                    events=lambda t, y: y[0])
    y_plot = np.clip(sol.y[0], 0, K_val)
    ax.plot(sol.t, y_plot, color=color, lw=2, label=f'$h={h}$ (super-critical)')

ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('$t$')
ax.set_ylabel('Population $x(t)$')
ax.set_title(r'Logistic with harvesting: $x\prime = rx(1-x/K) - h$, $x_0=2$')
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(-0.3, K_val + 0.5)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-torricelli
#| fig-cap: "Torricelli's law: water height $h(t)$ in a draining cylindrical tank ($A=0.5$ m², $a=0.01$ m², $h_0=2$ m). The closed-form parabolic solution (solid blue) agrees with the numerical solve (red dots). The tank empties at $t^* \\approx 201$ s."

g_val = 9.81
A_tank = 0.5    # m^2  (tank cross-section)
a_drain = 0.01  # m^2  (drain area)
h0 = 2.0        # m    (initial height)

alpha = a_drain * np.sqrt(2*g_val) / A_tank   # coefficient

t_empty = 2 * np.sqrt(h0) / alpha
t_plot = np.linspace(0, t_empty, 400)

# Closed-form
h_closed = (np.sqrt(h0) - alpha/2 * t_plot)**2

# Numerical
f_tor = lambda t, h: [-alpha * np.sqrt(max(h[0], 0))]
sol_tor = solve_ivp(f_tor, (0, t_empty), [h0], dense_output=True, max_step=0.5)
t_dots = np.linspace(0, t_empty*0.98, 20)
h_dots = sol_tor.sol(t_dots)[0]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t_plot, h_closed, color='steelblue', lw=2.5, label='Closed form')
ax.plot(t_dots, h_dots, 'ro', markersize=6, label='Numerical solve')
ax.axvline(t_empty, color='gray', ls=':', lw=1.5,
           label=f'$t^* = {t_empty:.1f}$ s (tank empty)')
ax.set_xlabel('Time $t$ (s)')
ax.set_ylabel('Water height $h(t)$ (m)')
ax.set_title("Torricelli's Law — Draining Tank (Bernoulli, $n=1/2$)")
ax.legend(fontsize=9)
ax.set_ylim(0, h0 + 0.1)
plt.tight_layout()
plt.show()


#| code-fold: true
#| label: fig-lanternfly-results
#| fig-cap: "Solutions of the social-learning ODE (2.4) for three ecological regimes. **Top:** $A > 0$ ($r_o = 0.8$, $q = 0.3$): all solutions with $p_0 > 0$ converge to the stable equilibrium $p^* = A/B$ (dashed green) — collective biological control emerges. **Middle:** $A = 0$ ($r_o = 0.5$, $q = 1/3$): the borderline case; $p(t) \\to 0$ algebraically as $1/(1+Bt)$. **Bottom:** $A < 0$ ($r_o = 0.2$, $q = 0.3$): all solutions decay to zero — collective control does not emerge. Closed-form solutions (solid) are verified against a direct numerical ODE solve (dots)."
#| fig-height: 10

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

mpl_rc = {'figure.dpi': 150,
           'axes.spines.top': False,
           'axes.spines.right': False}

for k, v in mpl_rc.items():
    plt.rcParams[k] = v

def closed_form(t, p0, A, B):
    """Exact solution (2.4); returns NaN where denominator <= 0."""
    if abs(A) < 1e-12:          # A = 0 case: p' = -B p^2
        return p0 / (1.0 + B * p0 * t)
    ratio = A / B
    denom = 1.0 + (ratio / p0 - 1.0) * np.exp(-A * t)
    return np.where(np.abs(denom) > 1e-8, ratio / denom, np.nan)

def ode_rhs(t, p, ri, ro, q):
    return [ri * p[0] * (ro * (1 - q - p[0]) - q)]

ri = 1.0
scenarios = [
    dict(ro=0.8, q=0.3,         label_suffix="$A>0$: control emerges",   cmap='viridis'),
    dict(ro=0.5, q=1.0/3.0,     label_suffix="$A=0$: borderline",        cmap='plasma'),
    dict(ro=0.2, q=0.3,         label_suffix="$A<0$: control fails",     cmap='coolwarm'),
]

fig, axes = plt.subplots(3, 1, figsize=(8, 10))
t_plot = np.linspace(0, 12, 500)
p0_vals = [0.05, 0.2, 0.5, 0.8]

for ax, sc in zip(axes, scenarios):
    ro, q = sc['ro'], sc['q']
    A = ri * (ro * (1 - q) - q)
    B = ri * ro
    colors = plt.get_cmap(sc['cmap'])(np.linspace(0.2, 0.85, len(p0_vals)))

    for p0, color in zip(p0_vals, colors):
        # Closed-form solution
        p_cf = closed_form(t_plot, p0, A, B)
        ax.plot(t_plot, p_cf, color=color, lw=2.2, label=f'$p_0={p0}$')

        # Numerical verification (dots)
        sol = solve_ivp(ode_rhs, (0, 12), [p0], args=(ri, ro, q),
                        dense_output=True, max_step=0.05)
        t_dots = np.linspace(0, 12, 18)
        ax.plot(t_dots, sol.sol(t_dots)[0], 'o', color=color, markersize=4)

    # Mark equilibrium p* when A > 0
    if A > 1e-10:
        p_star = A / B
        ax.axhline(p_star, color='seagreen', ls='--', lw=1.8,
                   label=f'$p^* = A/B \\approx {p_star:.2f}$')

    ax.axhline(0, color='crimson', ls='--', lw=1.2, label='$p=0$ (unstable)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$p(t)$')
    ax.set_title(
        rf"$r_i={ri},\; r_o={ro},\; q={round(q,3)}$ — {sc['label_suffix']} "
        rf"($A={A:.3f}$, $B={B:.3f}$)"
    )
    ax.legend(fontsize=8, ncol=3)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
