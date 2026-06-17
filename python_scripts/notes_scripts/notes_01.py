#| code-fold: true
#| code-summary: "Show the code"

# This is a code cell that imports the necessary libraries for our session.
import numpy as np                        # NumPy for numerical computations
import scipy as sp                        # SciPy for scientific computing
import sympy as sym                       # SymPy for symbolic mathematics
import matplotlib as mpl                  # Matplotlib for plotting
import matplotlib.pyplot as plt           # Matplotlib pyplot interface
from scipy.integrate import solve_ivp     # ODE solver
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-family
#| fig-cap: "Family of solutions $y(t) = Ce^{2t}$ to the ODE $y' - 2y = 0$. Each curve corresponds to a different value of $C$. The highlighted trajectory (dashed black) is the particular solution $C=1$ corresponding to the initial condition $y(0)=1$. Specifying an initial condition singles out exactly one curve from the family."

t = np.linspace(-1, 1.2, 300)
fig, ax = plt.subplots()
colors = plt.cm.coolwarm(np.linspace(0, 1, 9))
C_vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
for C, color in zip(C_vals, colors):
    label = f"$C = {C}$" if C in [-2, -1, 0, 1, 2] else None
    ax.plot(t, C * np.exp(2 * t), color=color, lw=1.5, label=label)
# Highlight the particular solution for C=1, corresponding to the initial condition y(0)=1
ax.plot(t, 1 * np.exp(2 * t), color='black', lw=2.5, linestyle='--', label='$C = 1$ (particular solution, $y(0)=1$)')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.set_xlabel("$t$")
ax.set_ylabel("$y$")
ax.set_ylim(-6, 6)
ax.set_title(r"Solutions $y(t) = Ce^{2t}$ of $y' - 2y = 0$")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-ivp
#| fig-cap: "The IVP $y'=-3y$, $y(0)=5$ selects a unique solution curve (bold black) from the family $y(t)=Ce^{-3t}$ (gray). The initial condition is marked with a red dot."

t = np.linspace(-0.5, 2, 400)
fig, ax = plt.subplots()
for C in np.linspace(-8, 8, 17):
    ax.plot(t, C * np.exp(-3 * t), color='lightsteelblue', lw=0.9, alpha=0.7)
ax.plot(t, 5 * np.exp(-3 * t), color='black', lw=2.5, label=r'$y(t)=5e^{-3t}$')
ax.plot(0, 5, 'ro', markersize=8, zorder=5, label='Initial condition $(0, 5)$')
ax.set_ylim(-10, 10)
ax.set_xlabel('$t$')
ax.set_ylabel('$y$')
ax.set_title(r"IVP: $y' = -3y$, $y(0)=5$")
ax.legend()
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-sho
#| fig-cap: "Simple harmonic oscillator with $\\omega = 1$, $x(0)=1$, $v(0)=0$. Left: position $x(t)$ over time. Right: phase portrait — trajectories in the $(x,\\dot{x})$ plane. Each closed ellipse is a solution for a different initial condition; the highlighted trajectory corresponds to the given IVP."

omega = 1.0
t_span = (0, 4 * np.pi)
t_eval = np.linspace(*t_span, 500)

def sho(t, y):
    x, v = y
    return [v, -omega**2 * x]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Time-domain solution
sol = solve_ivp(sho, t_span, [1.0, 0.0], t_eval=t_eval)
axes[0].plot(sol.t, sol.y[0], color='steelblue', lw=2)
axes[0].set_xlabel('$t$')
axes[0].set_ylabel('$x(t)$')
axes[0].set_title('SHO: Position vs. Time')
axes[0].axhline(0, color='k', lw=0.5)

# Phase portrait
x_vals = np.linspace(-2.5, 2.5, 400)
v_vals = np.linspace(-2.5, 2.5, 400)
X, V = np.meshgrid(x_vals, v_vals)
DX = V
DV = -omega**2 * X
speed = np.sqrt(DX**2 + DV**2)
speed[speed == 0] = 1
axes[1].streamplot(x_vals, v_vals, DX / speed, DV / speed,
                   color=np.log1p(speed), cmap='Blues', linewidth=0.7, density=1.3)
# Highlight specific trajectory
for ic in [(1.0, 0.0), (1.8, 0.0), (0.6, 0.0)]:
    s = solve_ivp(sho, (0, 2*np.pi), list(ic), t_eval=np.linspace(0, 2*np.pi, 300))
    lw = 2.5 if ic == (1.0, 0.0) else 1.2
    color = 'crimson' if ic == (1.0, 0.0) else 'steelblue'
    axes[1].plot(s.y[0], s.y[1], color=color, lw=lw)
axes[1].set_xlabel('$x$')
axes[1].set_ylabel(r'$\dot{x}$')
axes[1].set_title('Phase Portrait')
axes[1].set_aspect('equal')

plt.suptitle(r'Simple Harmonic Oscillator: $\ddot{x}+\omega^2 x=0$, $\omega=1$', fontsize=12)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-rc
#| fig-cap: "Discharge of an RC circuit: $Q(t)=Q_0 e^{-t/\\tau}$ where $\\tau=RC$ is the time constant. The charge decays to $1/e \\approx 36.8\\%$ of its initial value after one time constant."

tau_vals = [0.5, 1.0, 2.0]
t = np.linspace(0, 5, 300)
fig, ax = plt.subplots()
for tau in tau_vals:
    ax.plot(t, np.exp(-t / tau), lw=2, label=fr'$\tau = RC = {tau}$')
ax.axhline(1/np.e, color='gray', linestyle=':', lw=1.5, label=r'$Q/Q_0 = 1/e$')
ax.set_xlabel('$t$')
ax.set_ylabel('$Q(t)/Q_0$')
ax.set_title('RC Circuit Discharge: $Q(t) = Q_0 e^{-t/RC}$')
ax.legend()
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-logistic
#| fig-cap: "Logistic growth ($r=1$, $K=100$) from several initial conditions. All trajectories converge to the carrying capacity $K=100$ (dashed line), illustrating the stability of the equilibrium."

r, K = 1.0, 100.0
t = np.linspace(0, 8, 400)
fig, ax = plt.subplots()
for P0 in [5, 20, 50, 80, 120, 150]:
    P = K / (1 + (K / P0 - 1) * np.exp(-r * t))
    ax.plot(t, P, lw=2, label=f'$P_0 = {P0}$')
ax.axhline(K, color='black', linestyle='--', lw=1.5, label=f'Carrying capacity $K={K}$')
ax.set_xlabel('$t$')
ax.set_ylabel('$P(t)$')
ax.set_title('Logistic Growth')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys # sys for system-specific parameters and functions
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
