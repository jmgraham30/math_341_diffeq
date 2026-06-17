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
#| label: fig-nonlinear-fail
#| fig-cap: "Superposition fails for the nonlinear ODE $x'=x^2$. The solution $x_1(t)=1/(1-t)$ (blue) blows up at $t=1$; $2x_1$ (orange dashed) is not a solution — it does not satisfy the ODE, as confirmed by the large residual shown in the right panel."

t_plot = np.linspace(-2, 0.85, 400)
x1 = 1.0/(1 - t_plot)
x2 = 2.0/(1 - t_plot)   # proposed "2*x1"

# Residual of 2x1 in x' = x^2
dx2_dt = np.gradient(x2, t_plot)
residual = dx2_dt - x2**2   # should be 0 if x2 were a solution

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(t_plot, x1, color='steelblue', lw=2, label=r'$x_1 = 1/(1-t)$ (solution)')
axes[0].plot(t_plot, x2, color='darkorange', lw=2, ls='--', label=r'$2x_1$ (not a solution)')
axes[0].set_ylim(-3, 4)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x$')
axes[0].set_title(r"$x' = x^2$: solutions")
axes[0].legend(fontsize=9)
axes[0].axhline(0, color='k', lw=0.5)

axes[1].plot(t_plot, residual, color='crimson', lw=2)
axes[1].axhline(0, color='k', lw=1, ls='--')
axes[1].set_xlabel('$t$'); axes[1].set_ylabel(r"$\frac{d}{dt}(2x_1) - (2x_1)^2$")
axes[1].set_title('Residual of $2x_1$ in $x\'=x^2$ (nonzero $\Rightarrow$ not a solution)')
axes[1].set_ylim(-5, 1)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-sho-super
#| fig-cap: "Simple harmonic oscillator $x'' + 4x = 0$ ($\\omega = 2$). Left: the two basis solutions $x_1 = \\cos(2t)$ (blue) and $x_2 = \\sin(2t)$ (orange). Right: several linear combinations $C_1\\cos(2t)+C_2\\sin(2t)$ — all are solutions by superposition, and each corresponds to a different initial condition."

omega = 2.0
t_plot = np.linspace(0, 2*np.pi, 400)

x1 = np.cos(omega * t_plot)
x2 = np.sin(omega * t_plot)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(t_plot, x1, color='steelblue', lw=2.5, label=r'$x_1 = \cos(2t)$')
axes[0].plot(t_plot, x2, color='darkorange', lw=2.5, label=r'$x_2 = \sin(2t)$')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x$')
axes[0].set_title('Basis solutions')
axes[0].legend(fontsize=10)

# Various linear combinations
pairs = [(3, -1), (1, 2), (-2, 1), (0, 2), (2, 0)]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(pairs)))
for (C1, C2), color in zip(pairs, colors):
    x_combo = C1*x1 + C2*x2
    axes[1].plot(t_plot, x_combo, color=color, lw=2,
                 label=f'$C_1={C1},\\;C_2={C2}$')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x$')
axes[1].set_title(r'Superposition: $C_1\cos(2t)+C_2\sin(2t)$')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-sho-ivp
#| fig-cap: "IVP for $x''+4x=0$ with $x(0)=3$, $x'(0)=-2$. The superposition formula $x=3\\cos(2t)-\\sin(2t)$ (solid blue) is confirmed by a numerical solve (red dots)."

t_plot = np.linspace(0, 2*np.pi, 400)
x_exact = 3*np.cos(2*t_plot) - np.sin(2*t_plot)

def sho(t, y): return [y[1], -4*y[0]]
sol = solve_ivp(sho, (0, 2*np.pi), [3.0, -2.0], dense_output=True, max_step=0.01)
t_dots = np.linspace(0, 2*np.pi, 20)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_plot, x_exact, color='steelblue', lw=2.5, label=r'$3\cos(2t)-\sin(2t)$')
ax.plot(t_dots, sol.sol(t_dots)[0], 'ro', markersize=6, label='Numerical solve')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"IVP: $x''+4x=0$, $x(0)=3$, $x'(0)=-2$")
ax.legend()
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-driven-super
#| fig-cap: "Driven oscillator $x''+4x=\\cos(t)+3\\sin(5t)$, zero initial conditions. The particular solution (green dashed) is built by superposing the responses to each forcing term separately. The full solution (blue) adds the transient homogeneous part."

t_plot = np.linspace(0, 4*np.pi, 600)

# Particular solution via superposition
xp1 = np.cos(t_plot) / 3
xp2 = -np.sin(5*t_plot) / 7
xp  = xp1 + xp2

# Full solution with x(0)=0, x'(0)=0
# Need C1, C2: x(0)=C1 + 1/3 = 0 => C1=-1/3
# x'(0) = 2*C2 - 5/7 = 0 => C2 = 5/14
C1, C2 = -1/3, 5/14
x_full = C1*np.cos(2*t_plot) + C2*np.sin(2*t_plot) + xp

# Numerical verification
def driven(t, y): return [y[1], -4*y[0] + np.cos(t) + 3*np.sin(5*t)]
sol_d = solve_ivp(driven, (0, 4*np.pi), [0.0, 0.0], dense_output=True, max_step=0.005)

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

axes[0].plot(t_plot, xp1, color='steelblue', lw=1.8, ls='--', label=r'$x_p^{(1)}=\cos(t)/3$ (response to $\cos t$)')
axes[0].plot(t_plot, xp2, color='darkorange', lw=1.8, ls='--', label=r'$x_p^{(2)}=-\sin(5t)/7$ (response to $3\sin 5t$)')
axes[0].plot(t_plot, xp,  color='seagreen',  lw=2.5, ls='--', label=r'$x_p = x_p^{(1)}+x_p^{(2)}$ (superposition)')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_ylabel('$x$')
axes[0].set_title(r"Superposition of particular solutions: $x''+4x=\cos(t)+3\sin(5t)$")
axes[0].legend(fontsize=8)

t_dots = np.linspace(0, 4*np.pi, 30)
axes[1].plot(t_plot, x_full, color='steelblue', lw=2, label='Full solution (analytical)')
axes[1].plot(t_dots, sol_d.sol(t_dots)[0], 'ro', markersize=5, label='Numerical solve')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x$')
axes[1].set_title('Full solution with zero initial conditions')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

t_s = sym.Symbol('t')
x_s = sym.Function('x')

# Define operator L[x] = x'' + 4x
def L(func):
    return sym.diff(func, t_s, 2) + 4*func

x1_s = sym.cos(2*t_s)
x2_s = sym.sin(2*t_s)
C1_s, C2_s = sym.symbols('C1 C2')

print("Verifying homogeneous superposition:")
combo = C1_s*x1_s + C2_s*x2_s
result = sym.simplify(L(combo))
print(f"  L[C1*cos(2t) + C2*sin(2t)] = {result}")

print("\nVerifying particular solution superposition:")
xp1_s = sym.cos(t_s)/3
xp2_s = -sym.sin(5*t_s)/7
print(f"  L[cos(t)/3]       = {sym.simplify(L(xp1_s))}")
print(f"  L[-sin(5t)/7]     = {sym.simplify(L(xp2_s))}")
print(f"  L[xp1 + xp2]      = {sym.simplify(L(xp1_s + xp2_s))}")
print(f"  Forcing cos(t)+3sin(5t) = cos(t) + 3sin(5t)")


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-circuit-super
#| fig-cap: "Superposition theorem for an RC circuit ($R=1$, $C=1$) driven by two sources: $E_1(t) = \\sin(t)$ and $E_2(t) = 0.5\\cos(3t)$. The total response $Q$ (blue) equals the sum of the individual responses $Q_1$ (orange) and $Q_2$ (green) — numerically verified (red dots)."

R_val, C_val = 1.0, 1.0   # tau = RC = 1

# Q' + Q/RC = E(t)/R => Q' + Q = E(t) for R=C=1
def rc_response(E_func, t_span, Q0=0.0, steps=2000):
    f = lambda t, Q: [-Q[0] + E_func(t)]
    sol = solve_ivp(f, t_span, [Q0], dense_output=True,
                    max_steps=steps, max_step=0.01)
    return sol

t_span = (0, 4*np.pi)
t_plot = np.linspace(0, 4*np.pi, 600)

E1 = lambda t: np.sin(t)
E2 = lambda t: 0.5*np.cos(3*t)
E_total = lambda t: E1(t) + E2(t)

sol1 = rc_response(E1, t_span)
sol2 = rc_response(E2, t_span)
sol_total = rc_response(E_total, t_span)

Q1 = sol1.sol(t_plot)[0]
Q2 = sol2.sol(t_plot)[0]
Q_sum = Q1 + Q2
Q_total = sol_total.sol(t_plot)[0]

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_plot, Q1,    color='darkorange', lw=1.8, ls='--', label=r'$Q_1$: response to $E_1=\sin t$')
ax.plot(t_plot, Q2,    color='seagreen',   lw=1.8, ls='--', label=r'$Q_2$: response to $E_2=0.5\cos 3t$')
ax.plot(t_plot, Q_sum, color='steelblue',  lw=2.5,          label=r'$Q_1+Q_2$ (superposition)')
t_dots = np.linspace(0, 4*np.pi, 25)
ax.plot(t_dots, sol_total.sol(t_dots)[0], 'ro', markersize=5, label='Direct solve of combined circuit')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$Q(t)$')
ax.set_title('RC Circuit: Superposition Theorem ($R=C=1$)')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-beam
#| fig-cap: "Deflection of a simply supported beam of length $L=1$ under three loading cases, computed by superposition. Each load produces a deflection (dashed lines); the total deflection (solid blue) is their sum. The beam equation is $EI\\,y''''=w(x)$ with $EI=1$."

from numpy.polynomial import polynomial as P

L_beam = 1.0   # beam length
EI = 1.0       # bending stiffness (normalized)
x_b = np.linspace(0, L_beam, 400)

# Closed-form deflections for a simply supported beam (SS: y=0, y''=0 at x=0,L)
# Under uniform load w0: y = w0/(24*EI) * x*(L^3 - 2L*x^2 + x^3)
# Under central point load P at x=L/2:
#   y = P*x*(3L^2-4x^2)/(48*EI) for x <= L/2 (symmetric)
# Under sinusoidal load w(x)=w1*sin(pi*x/L):
#   y = w1*L^4/(pi^4*EI) * sin(pi*x/L)  (exact solution)

w0   = 5.0   # uniform load intensity
P_c  = 3.0   # central point load
w1   = 4.0   # sinusoidal load amplitude

# Load 1: uniform
y1 = w0 / (24*EI) * x_b * (L_beam**3 - 2*L_beam*x_b**2 + x_b**3)

# Load 2: central point load (symmetric)
y2 = np.where(x_b <= L_beam/2,
              P_c * x_b * (3*L_beam**2 - 4*x_b**2) / (48*EI),
              P_c * (L_beam - x_b) * (3*L_beam**2 - 4*(L_beam-x_b)**2) / (48*EI))

# Load 3: sinusoidal
y3 = w1 * L_beam**4 / (np.pi**4 * EI) * np.sin(np.pi * x_b / L_beam)

y_total = y1 + y2 + y3

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(x_b, -y1,     color='darkorange', lw=1.8, ls='--', label=f'Uniform load $w_0={w0}$')
ax.plot(x_b, -y2,     color='seagreen',   lw=1.8, ls='--', label=f'Central point load $P={P_c}$')
ax.plot(x_b, -y3,     color='mediumpurple', lw=1.8, ls='--', label=r'Sinusoidal load $w_1\sin(\pi x/L)$')
ax.plot(x_b, -y_total, color='steelblue', lw=2.8, label='Total deflection (superposition)')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('Position $x$'); ax.set_ylabel('Deflection $y$ (downward positive)')
ax.set_title(r"Simply Supported Beam: $EI\,y^{(4)}=w(x)$, deflection by superposition")
ax.legend(fontsize=8); ax.invert_yaxis()
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-fourier-heat
#| fig-cap: "Fourier series solution to the heat equation $u_t = 0.1\\,u_{xx}$ on $[0,1]$ with $u(x,0)=x(1-x)$ and zero boundary conditions. Left: Fourier sine series approximation to the initial condition using $N=1,3,5,15$ terms — superposition of simple sine modes converges to the parabola. Right: evolution of the solution at several times; each mode decays exponentially at its own rate."

from scipy.integrate import quad

k_heat = 0.1
L_heat = 1.0
x_arr = np.linspace(0, L_heat, 300)

# Fourier coefficients for f(x) = x(1-x): b_n = 8/(n^3*pi^3) for odd n
def b_n(n):
    val, _ = quad(lambda x: x*(1-x)*np.sin(n*np.pi*x/L_heat), 0, L_heat)
    return 2*val

def fourier_approx(x, N, t=0.0):
    u = np.zeros_like(x)
    for n in range(1, N+1):
        bn = b_n(n)
        decay = np.exp(-n**2 * np.pi**2 * k_heat * t / L_heat**2)
        u += bn * np.sin(n*np.pi*x/L_heat) * decay
    return u

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: convergence of initial condition
f_exact = x_arr*(1 - x_arr)
axes[0].plot(x_arr, f_exact, 'k--', lw=2, label='$f(x)=x(1-x)$')
colors_l = plt.cm.plasma(np.linspace(0.1, 0.85, 4))
for N, color in zip([1, 3, 5, 15], colors_l):
    u_approx = fourier_approx(x_arr, N, t=0.0)
    axes[0].plot(x_arr, u_approx, color=color, lw=1.8,
                 label=f'$N={N}$ terms')
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$u(x,0)$')
axes[0].set_title('Fourier series: superposition of sine modes')
axes[0].legend(fontsize=8)

# Right: time evolution
times = [0, 0.05, 0.2, 0.5, 1.5]
colors_r = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(times)))
for t_val, color in zip(times, colors_r):
    u_t = fourier_approx(x_arr, 30, t=t_val)
    axes[1].plot(x_arr, u_t, color=color, lw=2,
                 label=f'$t={t_val}$')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$u(x,t)$')
axes[1].set_title('Heat equation: solution at various times')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
