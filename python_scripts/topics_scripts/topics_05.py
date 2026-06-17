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

g = 9.81  # m/s^2
for m_kg, DL_m in [(0.3, 0.05), (0.5, 0.08), (1.0, 0.12)]:
    k = m_kg * g / DL_m
    omega0 = np.sqrt(k / m_kg)
    period  = 2 * np.pi / omega0
    print(f"m={m_kg} kg, ΔL={DL_m} m: k={k:.2f} N/m, ω₀={omega0:.3f} rad/s, T={period:.3f} s")


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-sho
#| fig-cap: "Simple harmonic oscillator $x''+4x=0$ ($\\omega_0=2$). Left: solution curves for three different initial conditions, all with period $T=\\pi$. Right: phase portrait — each closed ellipse is a trajectory; the amplitude $A=\\sqrt{x_0^2+v_0^2/\\omega_0^2}$ is conserved. The energy level $E=\\frac{1}{2}kA^2=2A^2$ labels each ellipse."

omega0 = 2.0
k_v, m_v = omega0**2, 1.0
t_plot = np.linspace(0, 2*np.pi, 500)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Time-domain solutions
ICs = [(1.5, 0.0, 'steelblue', '$x_0=1.5,\\ v_0=0$'),
       (0.5, 2.0, 'darkorange', '$x_0=0.5,\\ v_0=2$'),
       (0.0, 3.0, 'crimson',  '$x_0=0,\\ v_0=3$')]

def sho_ode(t, y): return [y[1], -omega0**2*y[0]]

for x0, v0, color, lbl in ICs:
    sol = solve_ivp(sho_ode, (0, 2*np.pi), [x0, v0], dense_output=True, max_step=0.01)
    axes[0].plot(t_plot, sol.sol(t_plot)[0], color=color, lw=2, label=lbl)

axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x(t)$')
axes[0].set_title(r'SHO: $x\'\'+4x=0$, three initial conditions')
axes[0].legend(fontsize=8.5)
axes[0].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
axes[0].set_xticklabels(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

# Phase portrait
x_grid = np.linspace(-2.2, 2.2, 400)
v_grid = np.linspace(-4.5, 4.5, 400)
X, V = np.meshgrid(x_grid, v_grid)
# Energy levels E = 0.5*m*V^2 + 0.5*k*X^2
E_grid = 0.5*m_v*V**2 + 0.5*k_v*X**2
E_levels = [0.5*k_v*A**2 for A in [0.5, 1.0, 1.5, np.sqrt(0.5**2 + (2/omega0)**2), 3/omega0]]
E_levels = sorted(set([round(e, 3) for e in E_levels]))

for x0, v0, color, lbl in ICs:
    sol2 = solve_ivp(sho_ode, (0, 2*np.pi), [x0, v0], dense_output=True, max_step=0.005)
    t_ph = np.linspace(0, 2*np.pi, 600)
    xp, vp = sol2.sol(t_ph)
    axes[1].plot(xp, vp, color=color, lw=2, label=lbl)

axes[1].plot(0, 0, 'ko', markersize=8, zorder=5, label='Equilibrium')
axes[1].set_xlabel('$x$'); axes[1].set_ylabel("$x'$")
axes[1].set_title('Phase portrait')
axes[1].legend(fontsize=8.5)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-damped-regimes
#| fig-cap: "Damped oscillator $x''+2\\gamma_m x'+4x=0$ ($\\omega_0=2$, $x(0)=1$, $x'(0)=0$). Left: position vs. time for all three regimes. The underdamped case shows decaying oscillations; the dotted envelope $\\pm e^{-\\gamma_m t}$ bounds the oscillation amplitude. Right: phase portrait. The underdamped spiral converges to the origin; the overdamped and critically damped cases arrive along the positive-$x$ axis without orbiting."

omega0 = 2.0
t_plot = np.linspace(0, 8, 600)
x0, v0 = 1.0, 0.0

regimes = [
    (0.3,  'steelblue',  'Underdamped $\\gamma_m=0.3$'),
    (2.0,  'darkorange', 'Critically damped $\\gamma_m=2.0$'),
    (3.5,  'crimson',    'Overdamped $\\gamma_m=3.5$'),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for gamma_m, color, lbl in regimes:
    def dho(t, y, g=gamma_m):
        return [y[1], -2*g*y[1] - omega0**2*y[0]]
    sol = solve_ivp(dho, (0, 8), [x0, v0], t_eval=t_plot, max_step=0.02)
    axes[0].plot(sol.t, sol.y[0], color=color, lw=2, label=lbl)
    axes[1].plot(sol.y[0], sol.y[1], color=color, lw=2, label=lbl)

# Underdamped envelope
gamma_m_ud = 0.3
axes[0].plot(t_plot,  np.exp(-gamma_m_ud*t_plot), color='steelblue', lw=1, ls=':',
             label='Envelope $\\pm e^{-\\gamma_m t}$')
axes[0].plot(t_plot, -np.exp(-gamma_m_ud*t_plot), color='steelblue', lw=1, ls=':')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x(t)$')
axes[0].set_title(r'Damped oscillator: three regimes ($\omega_0=2$)')
axes[0].legend(fontsize=8)

axes[1].plot(x0, v0, 'ko', markersize=7, label='IC $(1,0)$', zorder=5)
axes[1].plot(0,  0,  'k*', markersize=10, label='Equilibrium', zorder=5)
axes[1].set_xlabel('$x$'); axes[1].set_ylabel("$x'$")
axes[1].set_title('Phase portrait')
axes[1].legend(fontsize=8); axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-energy
#| fig-cap: "Energy in the spring–mass system ($m=1$, $k=4$, $x(0)=1.5$, $x'(0)=0$). Left: undamped ($\\gamma=0$) — total energy $E$ (black dashed) is constant; kinetic $T$ and potential $V$ oscillate out of phase, trading energy back and forth. Right: damped ($\\gamma=1$) — total energy decays exponentially; the shaded area represents energy dissipated."

m_v, k_v = 1.0, 4.0
omega0 = np.sqrt(k_v/m_v)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Undamped
def sho(t, y): return [y[1], -(k_v/m_v)*y[0]]
t_plot = np.linspace(0, 2*np.pi, 400)
sol_u = solve_ivp(sho, (0, 2*np.pi), [1.5, 0.0], dense_output=True, max_step=0.01)
x_u, v_u = sol_u.sol(t_plot)
T_u = 0.5*m_v*v_u**2
V_u = 0.5*k_v*x_u**2
E_u = T_u + V_u

axes[0].plot(t_plot, T_u, color='steelblue',  lw=2, label='$T = \\frac{1}{2}m(x\')^2$')
axes[0].plot(t_plot, V_u, color='crimson',    lw=2, label='$V = \\frac{1}{2}kx^2$')
axes[0].plot(t_plot, E_u, color='black',      lw=2, ls='--', label='$E = T+V$ (conserved)')
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('Energy')
axes[0].set_title('Undamped: energy conservation')
axes[0].legend(fontsize=8.5)
axes[0].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
axes[0].set_xticklabels(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

# Damped
gamma_v = 1.0
def dho_en(t, y): return [y[1], -(gamma_v/m_v)*y[1] - (k_v/m_v)*y[0]]
t_plot2 = np.linspace(0, 8, 600)
sol_d = solve_ivp(dho_en, (0, 8), [1.5, 0.0], dense_output=True, max_step=0.01)
x_d, v_d = sol_d.sol(t_plot2)
T_d = 0.5*m_v*v_d**2
V_d = 0.5*k_v*x_d**2
E_d = T_d + V_d
E0  = E_d[0]

axes[1].plot(t_plot2, T_d, color='steelblue',  lw=2, label='$T$')
axes[1].plot(t_plot2, V_d, color='crimson',    lw=2, label='$V$')
axes[1].plot(t_plot2, E_d, color='black',      lw=2.5, ls='--', label='$E = T+V$')
axes[1].fill_between(t_plot2, E_d, E0, alpha=0.15, color='seagreen',
                     label='Energy dissipated')
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('Energy')
axes[1].set_title(f'Damped ($\\gamma={gamma_v}$): energy dissipation')
axes[1].legend(fontsize=8.5)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-analogy
#| fig-cap: "The mechanical–electrical analogy. Spring–mass (left) and RLC circuit (right) obey identical ODEs. The simulation shows $x(t)$ for the mechanical system (blue) and $Q(t)$ for the circuit (orange dashed) with matching dimensionless parameters $\\omega_0=2$, $\\gamma_m=0.4$, both starting from the same initial conditions. The curves are identical — up to the choice of units."

omega0 = 2.0
gamma_m = 0.4
t_plot = np.linspace(0, 6, 400)

def sys(t, y): return [y[1], -2*gamma_m*y[1] - omega0**2*y[0]]
sol = solve_ivp(sys, (0, 6), [1.0, 0.0], dense_output=True, max_step=0.01)
x_t = sol.sol(t_plot)[0]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Mechanical
axes[0].plot(t_plot, x_t, color='steelblue', lw=2.5, label='Displacement $x(t)$')
axes[0].axhline(0, color='k', lw=0.5)
axes[0].set_xlabel('$t$'); axes[0].set_ylabel('$x(t)$')
axes[0].set_title(r'Spring–mass: $mx\'\'+\gamma x\'+kx=0$')
axes[0].set_ylim(-1.2, 1.2); axes[0].legend(fontsize=9)

# Electrical (same ODE, different variable name)
axes[1].plot(t_plot, x_t, color='darkorange', lw=2.5, ls='--', label='Charge $Q(t)$')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$Q(t)$')
axes[1].set_title(r'RLC circuit: $LQ\'\'+RQ\'+Q/C=0$')
axes[1].set_ylim(-1.2, 1.2); axes[1].legend(fontsize=9)

plt.suptitle(f'Mechanical–Electrical Analogy ($\\omega_0={omega0}$, '
             f'$\\gamma_m={gamma_m}$, identical ODEs)', fontsize=11)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-frequency-response
#| fig-cap: "Frequency response of the forced damped oscillator ($\\omega_0=2$, $F_0/m=1$). Left: amplitude $G(\\Omega)$ vs. driving frequency for several damping ratios. The peak shifts below $\\omega_0=2$ (vertical dashed line) and broadens as damping increases. Right: time-domain steady-state solutions for three driving frequencies with $\\gamma_m=0.3$, illustrating below-resonance, at-resonance, and above-resonance behavior."

omega0 = 2.0; gamma_m_vals = [0.1, 0.3, 0.5, 1.0, 1.5]
Omega_arr = np.linspace(0.01, 5, 600)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

colors_amp = plt.cm.plasma(np.linspace(0.1, 0.85, len(gamma_m_vals)))
for gm, color in zip(gamma_m_vals, colors_amp):
    G = 1.0 / np.sqrt((omega0**2 - Omega_arr**2)**2 + (2*gm*Omega_arr)**2)
    axes[0].plot(Omega_arr, G, color=color, lw=2, label=f'$\\gamma_m={gm}$')
    if gm < omega0/np.sqrt(2):
        Omega_res = np.sqrt(omega0**2 - 2*gm**2)
        axes[0].plot(Omega_res, 1/np.sqrt((omega0**2-Omega_res**2)**2+(2*gm*Omega_res)**2),
                     'o', color=color, markersize=6)

axes[0].axvline(omega0, color='k', ls='--', lw=1.2, label=f'$\\omega_0={omega0}$')
axes[0].set_xlabel('Driving frequency $\\Omega$')
axes[0].set_ylabel('Amplitude $G(\\Omega)$')
axes[0].set_title('Amplitude response function')
axes[0].legend(fontsize=8, ncol=2); axes[0].set_ylim(0, 6)

# Time-domain for three Omega values
gm_fixed = 0.3
t_plot = np.linspace(0, 20, 800)
for Omega_val, color, lbl in [(1.0, 'steelblue', '$\\Omega=1$ (below)'),
                               (2.0, 'crimson',   '$\\Omega=2=\\omega_0$ (resonance)'),
                               (3.5, 'seagreen',  '$\\Omega=3.5$ (above)')]:
    def forced(t, y, Ov=Omega_val):
        return [y[1], -2*gm_fixed*y[1] - omega0**2*y[0] + np.cos(Ov*t)]
    sol_f = solve_ivp(forced, (0, 20), [0.0, 0.0], t_eval=t_plot, max_step=0.02)
    axes[1].plot(sol_f.t, sol_f.y[0], color=color, lw=1.8, label=lbl)

axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x(t)$')
axes[1].set_title(f'Forced oscillations ($\\gamma_m={gm_fixed}$, zero ICs)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"
#| label: fig-resonance-buildup
#| fig-cap: "Resonance amplitude buildup ($\\omega_0=2$, $\\gamma=0$, zero ICs). Without damping, the resonant particular solution $x_p = (t/4)\\sin(2t)$ grows linearly without bound (dashed line shows the envelope $\\pm t/4$). In practice, material nonlinearities or structural failure occur before infinite amplitude is reached."

omega0 = 2.0
t_plot = np.linspace(0, 20, 800)

def sho_forced_res(t, y): return [y[1], -omega0**2*y[0] + np.cos(omega0*t)]
sol_res = solve_ivp(sho_forced_res, (0, 20), [0.0, 0.0], t_eval=t_plot, max_step=0.01)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(sol_res.t, sol_res.y[0], color='crimson', lw=2, label='Resonant solution $x(t)$')
ax.plot(t_plot,  t_plot/4, color='gray', lw=1.5, ls='--', label='Envelope $\\pm t/4$')
ax.plot(t_plot, -t_plot/4, color='gray', lw=1.5, ls='--')
ax.axhline(0, color='k', lw=0.5)
ax.set_xlabel('$t$'); ax.set_ylabel('$x(t)$')
ax.set_title(r"Resonance: $x''+4x=\cos(2t)$, $x(0)=x'(0)=0$ (undamped)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


#| code-fold: true
#| code-summary: "Show the code"

import sys
print("Python version:", sys.version)
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
