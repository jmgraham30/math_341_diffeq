import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# Three different (r, K, P0) sets that share the same u0 = P0/K = 0.1
configs = [
    dict(r=1.0,  K=100, P0=10,  label=r'$r=1,\;K=100,\;P_0=10$',   color='steelblue'),
    dict(r=0.5,  K=200, P0=20,  label=r'$r=0.5,\;K=200,\;P_0=20$', color='crimson'),
    dict(r=2.0,  K=50,  P0=5,   label=r'$r=2,\;K=50,\;P_0=5$',     color='seagreen'),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for cfg in configs:
    r, K, P0 = cfg['r'], cfg['K'], cfg['P0']
    u0 = P0 / K

    # Dimensional time axis: show ~5 intrinsic time scales
    t_dim  = np.linspace(0, 5 / r, 400)
    P_sol  = K * u0 / (u0 + (1 - u0) * np.exp(-r * t_dim))
    axes[0].plot(t_dim, P_sol, lw=2, color=cfg['color'], label=cfg['label'])

    # Dimensionless axis tau = r*t, same u0
    tau = np.linspace(0, 10, 400)
    u_sol = u0 / (u0 + (1 - u0) * np.exp(-tau))
    axes[1].plot(tau, u_sol, lw=2, color=cfg['color'], label=cfg['label'])

axes[0].set_xlabel('$t$', fontsize=12)
axes[0].set_ylabel('$P(t)$', fontsize=12)
axes[0].set_title('Dimensional form', fontsize=12)
axes[0].legend(fontsize=8)

axes[1].set_xlabel(r'$\tau = rt$', fontsize=12)
axes[1].set_ylabel(r'$u(\tau) = P/K$', fontsize=12)
axes[1].set_title('Dimensionless form', fontsize=12)
axes[1].axhline(1, color='black', linestyle='--', lw=1, label='$u=1$ (carrying capacity)')
axes[1].legend(fontsize=8)

plt.suptitle('Logistic Growth: Dimensional vs. Dimensionless', fontsize=13)
plt.tight_layout()
plt.show()

nu_vals = np.linspace(0, 2.5, 800)
zeta_list = [0.05, 0.1, 0.2, 0.5, 1.0]
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(zeta_list)))

fig, ax = plt.subplots(figsize=(8, 5))
for zeta, color in zip(zeta_list, colors):
    R = 1.0 / np.sqrt((1 - nu_vals**2)**2 + 4 * zeta**2 * nu_vals**2)
    ax.plot(nu_vals, R, lw=2, color=color, label=fr'$\zeta = {zeta}$')

ax.axvline(1, color='black', linestyle=':', lw=1.2, label=r'$\nu = 1$ (resonance)')
ax.set_xlabel(r'Frequency ratio $\nu = \Omega/\omega_0$', fontsize=12)
ax.set_ylabel(r'Amplitude $R(\nu,\zeta)$', fontsize=12)
ax.set_title('Amplitude Response of the Dimensionless Forced Oscillator', fontsize=12)
ax.set_ylim(0, 7)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

def forced_osc(tau, y, zeta, nu):
    u, v = y
    return [v, np.cos(nu * tau) - 2 * zeta * v - u]

tau_span = (0, 60)
tau_eval = np.linspace(*tau_span, 3000)
zeta_list2 = [0.05, 0.1, 0.2, 0.5]
colors2 = plt.cm.viridis(np.linspace(0.1, 0.85, len(zeta_list2)))

fig, ax = plt.subplots(figsize=(10, 4))
for zeta, color in zip(zeta_list2, colors2):
    sol = solve_ivp(forced_osc, tau_span, [0, 0],
                    args=(zeta, 1.0), t_eval=tau_eval, max_step=0.05)
    ax.plot(sol.t, sol.y[0], lw=1.5, color=color, label=fr'$\zeta={zeta}$')

ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel(r'$\tau = \omega_0 t$', fontsize=12)
ax.set_ylabel(r'$u(\tau) = x / \mathcal{X}$', fontsize=12)
ax.set_title(r'Dimensionless Forced Oscillator at Resonance ($\nu=1$)', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()