#| code-fold: true
#| code-summary: "Show the code"

# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import scipy as sp                        # SciPy for scientific computing
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from scipy.integrate import solve_ivp     # ODE solver
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


#| code-fold: true
#| code-summary: "Show the code"

def slope_field(f, t_range, x_range, n=20, ax=None, color='steelblue', alpha=0.7):
    """
    Draw the slope field for x' = f(t, x).

    Parameters
    ----------
    f        : callable  f(t, x) — the right-hand side of the ODE
    t_range  : (t_min, t_max)
    x_range  : (x_min, x_max)
    n        : grid resolution (n x n arrows)
    ax       : matplotlib Axes (creates one if None)
    color    : arrow color
    alpha    : arrow transparency
    """
    if ax is None:
        fig, ax = plt.subplots()
    t_vals = np.linspace(*t_range, n)
    x_vals = np.linspace(*x_range, n)
    T, X = np.meshgrid(t_vals, x_vals)
    dT = np.ones_like(T)          # dt component always 1
    dX = f(T, X)                  # dx component = f(t, x)
    # Normalize arrow lengths for visual clarity
    norm = np.sqrt(dT**2 + dX**2)
    norm[norm == 0] = 1
    ax.quiver(T, X, dT / norm, dX / norm,
              angles='xy', scale=n * 0.8, width=0.003,
              color=color, alpha=alpha, headlength=3, headwidth=3)
    ax.set_xlabel('$t$'); ax.set_ylabel('$x$')
    return ax


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-sf-linear
#| fig-cap: "Slope field for $x' = -x + 2t$ with several solution curves (blue) and the nullcline $x = 2t$ (red dashed). Solutions appear to converge toward the line $x = 2t - 2$ — the particular solution satisfying the ODE for all $t$ (verification: if $x = 2t-2$ then $x' = 2$ and $-x+2t = -(2t-2)+2t = 2$ ✓)."

f1 = lambda t, x: -x + 2*t

fig, ax = plt.subplots(figsize=(7, 5))
slope_field(f1, (-1, 4), (-4, 8), n=22, ax=ax)

# Several solution curves via solve_ivp
t_eval = np.linspace(-1, 4, 400)
for x0 in [-3, -1, 0, 2, 4, 6, 7.5]:
    sol = solve_ivp(f1, (-1, 4), [x0], t_eval=t_eval, max_step=0.05)
    ax.plot(sol.t, sol.y[0], color='steelblue', lw=1.8)

# Nullcline x = 2t
t_nc = np.linspace(-1, 4, 200)
ax.plot(t_nc, 2*t_nc, 'r--', lw=1.8, label='Nullcline $x = 2t$')

ax.set_xlim(-1, 4); ax.set_ylim(-4, 8)
ax.set_title(r"Slope field: $x' = -x + 2t$")
ax.legend()
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-sf-auto
#| fig-cap: "Slope field for the autonomous ODE $x' = x(1-x/4)$ (logistic-type). Horizontal dashed lines mark the nullclines/equilibria $x=0$ and $x=4$. All solutions starting with $x_0 > 0$ converge to $x=4$."

f2 = lambda t, x: x * (1 - x / 4)

fig, ax = plt.subplots(figsize=(7, 5))
slope_field(f2, (0, 8), (-1, 6.5), n=22, ax=ax)

t_eval = np.linspace(0, 8, 400)
for x0 in [0.2, 0.5, 1, 2, 3, 4.5, 6]:
    sol = solve_ivp(f2, (0, 8), [x0], t_eval=t_eval, max_step=0.05)
    ax.plot(sol.t, sol.y[0], color='steelblue', lw=1.8)

# Equilibria
ax.axhline(0, color='crimson', linestyle='--', lw=1.5, label='$x=0$ (unstable)')
ax.axhline(4, color='seagreen', linestyle='--', lw=1.5, label='$x=4$ (stable)')

ax.set_title(r"Slope field: $x' = x(1 - x/4)$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-sf-nonlinear
#| fig-cap: "Slope field for $x' = x(x-t)$ with nullclines $x=0$ (red) and $x=t$ (orange). Signs of $x'$ in the four regions are annotated. Four solution curves are attempted (initial conditions at $t=-1$: $x_0 \\in \\{-1.5, 0.5, 2.0, 3.5\\}$); some may exit the plot window due to finite-time blowup."

f3 = lambda t, x: x * (x - t)

fig, ax = plt.subplots(figsize=(7, 5))
slope_field(f3, (-1, 4), (-3, 4.5), n=22, ax=ax)

t_eval = np.linspace(-1, 4, 600)
for x0, t0 in [(2.0, -1), (0.5, -1), (-1.5, -1), (3.5, -1)]:
    try:
        sol = solve_ivp(f3, (-1, 4), [x0], t_eval=t_eval, max_step=0.02,
                        events=lambda t, y: y[0] - 15)
        ax.plot(sol.t, np.clip(sol.y[0], -4, 5), color='steelblue', lw=1.8)
    except Exception:
        pass

t_nc = np.linspace(-1, 4, 200)
ax.axhline(0, color='crimson', linestyle='--', lw=1.8, label='Nullcline $x=0$')
ax.plot(t_nc, t_nc, color='darkorange', linestyle='--', lw=1.8, label='Nullcline $x=t$')

# Sign annotations
for txt, pos in [(r"$x'>0$", (2, -1.8)), (r"$x'<0$", (2, 0.6)),
                 (r"$x'>0$", (0.5, 2.5)), (r"$x'<0$", (-0.5, -1.5))]:
    ax.text(*pos, txt, fontsize=10, ha='center', color='black')

ax.set_xlim(-1, 4); ax.set_ylim(-3, 4.5)
ax.set_title(r"Slope field: $x' = x(x-t)$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

t, r, C, x0 = sym.symbols('t r C x_0', real=True, positive=True)
x = sym.Function('x')

ode = sym.Eq(x(t).diff(t), r * x(t))
sol = sym.dsolve(ode, x(t))
print("General solution:", sol)

sol_ivp = sym.dsolve(ode, x(t), ics={x(0): x0})
print("IVP solution (x(0)=x0):", sol_ivp)


#| code-fold: true
#| code-summary: "Show the code"

t_s, r_s, K_s, x0_s = sym.symbols('t r K x_0', positive=True)
x_s = sym.Function('x')

logistic_ode = sym.Eq(x_s(t_s).diff(t_s), r_s * x_s(t_s) * (1 - x_s(t_s) / K_s))
sol_logistic = sym.dsolve(logistic_ode, x_s(t_s), ics={x_s(0): x0_s})
print("Logistic IVP solution:")
display(Math(sym.latex(sol_logistic)))


#| code-fold: true
#| code-summary: "Show the code"

# Simplify the SymPy result and compare with the standard form
sol_simplified = sym.simplify(sol_logistic.rhs)
standard_form = K_s / (1 + (K_s/x0_s - 1)*sym.exp(-r_s*t_s))
print("SymPy simplified rhs equals standard form:",
      sym.simplify(sol_simplified - standard_form) == 0)
display(Math(r"x(t) = " + sym.latex(sym.simplify(sol_simplified))))


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-nonclosed
#| fig-cap: "Solution to the IVP $x' = 2\\sqrt{x}\\,e^{-t}/t$, $x(1)=4$, expressed via a non-elementary integral. The solid blue curve shows the formula; the red dots confirm it by comparing with a direct numerical ODE solve."

from scipy.integrate import quad

def x_formula(t_val):
    integral, _ = quad(lambda s: np.exp(-s) / s, 1, t_val)
    return (integral + 2)**2

t_plot = np.linspace(1, 5, 300)
x_plot = np.array([x_formula(tv) for tv in t_plot])

# Cross-check with direct ODE solve
f_ode = lambda t, x: 2 * np.sqrt(max(x[0], 0)) * np.exp(-t) / t
sol_check = solve_ivp(f_ode, (1, 5), [4.0], t_eval=t_plot, max_step=0.02)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_plot, x_plot, color='steelblue', lw=2.5, label='Integral formula')
ax.plot(sol_check.t[::15], sol_check.y[0][::15], 'ro', markersize=5, label='Numerical ODE solve')
ax.plot(1, 4, 'k*', markersize=10, label='IC $(1, 4)$')
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"$x' = 2\sqrt{x}\,e^{-t}/t$, $x(1)=4$")
ax.legend()
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-linear1
#| fig-cap: "General solution $x(t) = t/2 + C/t$ of $x' + x/t = 1$ for several values of $C$ (gray), with the particular solution satisfying $x(1)=3$ highlighted (blue). The initial condition is marked in red."

t_plot = np.linspace(0.2, 5, 400)
fig, ax = plt.subplots(figsize=(7, 4))
for C_val in np.linspace(-6, 6, 13):
    ax.plot(t_plot, t_plot/2 + C_val/t_plot, color='lightsteelblue', lw=0.9, alpha=0.8)
# Particular solution C = 5/2
ax.plot(t_plot, t_plot/2 + 2.5/t_plot, color='steelblue', lw=2.5, label=r'$C=5/2$, $x(1)=3$')
ax.plot(1, 3, 'ro', markersize=8, zorder=5, label='IC $(1, 3)$')
ax.set_ylim(-4, 8); ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"$x' + x/t = 1$: general solution family")
ax.legend()
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-linear2
#| fig-cap: "Solutions of $x' + 2x = \\sin t$ for various initial conditions. The transient $Ce^{-2t}$ decays quickly and all solutions converge to the same steady-state oscillation $x_p = \\frac{2}{5}\\sin t - \\frac{1}{5}\\cos t$ (dashed black)."

t_s = sym.Symbol('t')
x_s = sym.Function('x')
ode2 = sym.Eq(x_s(t_s).diff(t_s) + 2*x_s(t_s), sym.sin(t_s))
gen_sol = sym.dsolve(ode2, x_s(t_s))
print("General solution:")
display(Math(sym.latex(gen_sol)))

# Numerical plot
t_plot = np.linspace(0, 8, 500)
xp = 0.4*np.sin(t_plot) - 0.2*np.cos(t_plot)   # particular solution

fig, ax = plt.subplots(figsize=(7, 4))
for C_val in [-3, -2, -1, 0, 1, 2, 3]:
    x_sol = xp + C_val * np.exp(-2 * t_plot)
    ax.plot(t_plot, x_sol, color='steelblue', lw=1.5, alpha=0.8)
ax.plot(t_plot, xp, 'k--', lw=2.2, label=r'Steady state $x_p = \frac{2}{5}\sin t - \frac{1}{5}\cos t$')
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$'); ax.set_ylim(-4, 4)
ax.set_title(r"$x' + 2x = \sin t$: transient decay to steady state")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-cooling
#| fig-cap: "Newton's law of cooling: temperature $T(t) = T_e + (T_0 - T_e)e^{-ht}$ for $T_e = 20^\\circ$C and $h=0.5$ hr$^{-1}$, starting from several initial temperatures. All curves converge to the equilibrium $T_e = 20^\\circ$C (dashed)."

h_val = 0.5
Te = 20.0
t_plot = np.linspace(0, 10, 300)

fig, ax = plt.subplots(figsize=(7, 4))
for T0 in [0, 5, 10, 40, 60, 80, 100]:
    T_sol = Te + (T0 - Te) * np.exp(-h_val * t_plot)
    ax.plot(t_plot, T_sol, lw=2,
            label=f'$T_0 = {T0}°C$' if T0 in [0, 40, 100] else None)
ax.axhline(Te, color='black', linestyle='--', lw=1.5,
           label=f'Equilibrium $T_e = {int(Te)}°C$')
ax.set_xlabel('Time (hours)'); ax.set_ylabel('Temperature (°C)')
ax.set_title(r"Newton's Law of Cooling, $h=0.5$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-rc
#| fig-cap: "RC circuit ($R=1$, $C=1/2$) with the voltage source switched on for $0 \\le t < 2$ then switched off. Top: charge $Q(t)$. Bottom: current $I(t) = Q'(t)$. Both confirm the matching condition at $t=2$ and the characteristic exponential transient with time constant $\\tau = RC = 1/2$."

R_val, C_val = 1.0, 0.5
tau = R_val * C_val     # = 0.5
Q0 = 0.0

t1 = np.linspace(0, 2, 300)
t2 = np.linspace(2, 6, 300)

Q1 = 0.5 * (1 - np.exp(-2 * t1))
Q_switch = 0.5 * (1 - np.exp(-4))           # Q at t=2, matching condition
Q2 = Q_switch * np.exp(-2 * (t2 - 2))

# Compute I = Q' analytically to avoid numerical-derivative artifacts at the switch point.
# For 0 <= t < 2:  Q1 = (1/2)(1 - e^{-2t})  =>  I1 = e^{-2t}
# For t >= 2:      Q2 = Q_switch * e^{-2(t-2)}  =>  I2 = -2*Q_switch*e^{-2(t-2)}
I1 = np.exp(-2 * t1)
I2 = -2 * Q_switch * np.exp(-2 * (t2 - 2))

fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

axes[0].plot(t1, Q1, color='steelblue', lw=2)
axes[0].plot(t2, Q2, color='steelblue', lw=2)
axes[0].axvline(2, color='gray', linestyle=':', lw=1.2)
axes[0].set_ylabel('$Q(t)$ (coulombs)')
axes[0].set_title(r'RC Circuit: $R=1$, $C=1/2$, $E(t)=1$ for $0\leq t<2$, else $0$')

axes[1].plot(t1, I1, color='crimson', lw=2)
axes[1].plot(t2, I2, color='crimson', lw=2)
axes[1].axvline(2, color='gray', linestyle=':', lw=1.2, label='Switch off at $t=2$')
axes[1].set_ylabel('$I(t)$ (amperes)')
axes[1].set_xlabel('$t$ (seconds)')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-chemostat
#| fig-cap: "Chemical concentration in a well-mixed tank ($V=100$ m$^3$, $q=0.5$ m$^3$/min, $C_{\\text{in}} = 0.2$ kg/m$^3$) for several initial concentrations $C_0$. All solutions converge to the equilibrium $C_{\\text{in}} = 0.2$ (dashed)."

V_val = 100.0     # m^3
q_v   = 0.5       # m^3/min  (volumetric flow rate; written q_v to distinguish from the forcing term q(t) in the linear ODE)
C_in  = 0.2       # kg/m^3
t_plot = np.linspace(0, 600, 500)

fig, ax = plt.subplots(figsize=(7, 4))
for C0 in [0, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]:
    C_sol = C_in + (C0 - C_in) * np.exp(-q_v / V_val * t_plot)
    ax.plot(t_plot, C_sol, lw=2,
            label=f'$C_0={C0}$' if C0 in [0, 0.5, 1.0] else None)
ax.axhline(C_in, color='black', linestyle='--', lw=1.5,
           label=f'Equilibrium $C_{{\\rm in}}={C_in}$ kg/m³')
ax.set_xlabel('Time (min)'); ax.set_ylabel('Concentration (kg/m³)')
ax.set_title('Chemical Reactor (Chemostat)')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys # sys for system-specific parameters and functions
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
