import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

a, b, c, d = 0.6, 0.5, 0.3, 0.4
xe, ye = c/d, a/b   # non-trivial equilibrium

def f_lv(x, y): return a*x - b*x*y
def g_lv(x, y): return -c*y + d*x*y

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Phase portrait
lim_x, lim_y = 3.0, 3.0
x_g, y_g = np.meshgrid(np.linspace(0.02, lim_x, 22),
                        np.linspace(0.02, lim_y, 22))
dx = f_lv(x_g, y_g); dy = g_lv(x_g, y_g)
nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
axes[0].quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.25, color='gray', scale=28)

# Nullclines
x_nc = np.linspace(0.01, lim_x, 300)
axes[0].axvline(xe, color='crimson', ls='--', lw=1.8,
                label=rf"$y$-nullcline: $x={xe:.2f}$")
axes[0].axhline(ye, color='steelblue', ls='--', lw=1.8,
                label=rf"$x$-nullcline: $y={ye:.2f}$")
axes[0].axhline(0,  color='gray', ls='--', lw=0.8)
axes[0].axvline(0,  color='gray', ls='--', lw=0.8)

# Closed orbits
colors_orb = plt.cm.viridis(np.linspace(0.15, 0.85, 5))
for r_frac, color in zip([0.15, 0.35, 0.55, 0.75, 0.95], colors_orb):
    x0 = xe + r_frac * xe
    sol = solve_ivp(lambda t, z: [f_lv(z[0], z[1]), g_lv(z[0], z[1])],
                    (0, 60), [x0, ye], dense_output=True,
                    max_step=0.05, rtol=1e-8, atol=1e-10)
    xy = sol.y
    mask = (xy[0] > 0) & (xy[1] > 0) & (xy[0] < lim_x) & (xy[1] < lim_y)
    axes[0].plot(xy[0, mask], xy[1, mask], color=color, lw=1.5)

axes[0].plot(xe, ye, 'o', color='steelblue', markersize=9, zorder=6,
             label=f'Equil. $({xe:.2f},{ye:.2f})$ — center')
axes[0].plot(0, 0, 's', color='crimson', markersize=8, zorder=6,
             label='Origin — saddle')
axes[0].set_xlim(0, lim_x); axes[0].set_ylim(0, lim_y)
axes[0].set_xlabel('$x$ (prey)'); axes[0].set_ylabel('$y$ (predator)')
axes[0].set_title('Phase portrait')
axes[0].legend(fontsize=8)

# Time series
x0_ts, y0_ts = xe + 0.5*xe, ye
sol_ts = solve_ivp(lambda t, z: [f_lv(z[0], z[1]), g_lv(z[0], z[1])],
                   (0, 40), [x0_ts, y0_ts], dense_output=True,
                   max_step=0.05, rtol=1e-8, atol=1e-10)
t_plot = np.linspace(0, 40, 800)
axes[1].plot(t_plot, sol_ts.sol(t_plot)[0], color='steelblue', lw=2.5,
             label='$x(t)$ prey')
axes[1].plot(t_plot, sol_ts.sol(t_plot)[1], color='crimson', lw=2.5,
             label='$y(t)$ predator')
axes[1].axhline(xe, color='steelblue', ls=':', lw=1, alpha=0.7)
axes[1].axhline(ye, color='crimson',   ls=':', lw=1, alpha=0.7)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('Population')
axes[1].set_title('Time series')
axes[1].legend(fontsize=9)

plt.suptitle(f'Lotka–Volterra: $a={a}$, $b={b}$, $c={c}$, $d={d}$', fontsize=12)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

def competing(a1, b1, c1, a2, b2, c2):
    def f_(x, y): return x*(a1 - b1*x - c1*y)
    def g_(x, y): return y*(a2 - b2*y - c2*x)
    return f_, g_

cases = [
    dict(a1=1, b1=1, c1=0.3, a2=1, b2=1, c2=0.5,
         title="Weak competition — Coexistence\n($c_1=0.3$, $c_2=0.5$)"),
    dict(a1=1, b1=1, c1=1.4, a2=1, b2=1, c2=1.2,
         title="Strong competition — Exclusion\n($c_1=1.4$, $c_2=1.2$)"),
]

for ax, case in zip(axes, cases):
    a1,b1,c1,a2,b2,c2 = (case[k] for k in ('a1','b1','c1','a2','b2','c2'))
    f_, g_ = competing(a1,b1,c1,a2,b2,c2)

    lim = 1.6
    x_g, y_g = np.meshgrid(np.linspace(0, lim, 22), np.linspace(0, lim, 22))
    dx = f_(x_g, y_g); dy = g_(x_g, y_g)
    nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
    ax.quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.25, color='gray', scale=28)

    # Nullclines
    x_nc = np.linspace(0, lim, 300)
    ax.plot(x_nc, (a1 - b1*x_nc)/c1, 'steelblue', ls='--', lw=1.8,
            label=r"$x'=0$")
    ax.plot(x_nc, (a2 - c2*x_nc)/b2, 'crimson', ls='--', lw=1.8,
            label=r"$y'=0$")

    # Orbits
    np.random.seed(42)
    for _ in range(14):
        x0 = np.random.uniform(0.05, 1.4)
        y0 = np.random.uniform(0.05, 1.4)
        sol = solve_ivp(lambda t, z: [f_(z[0],z[1]), g_(z[0],z[1])],
                        (0, 20), [x0, y0], dense_output=True, max_step=0.05)
        xy = sol.y
        mask = (xy[0]>=0) & (xy[1]>=0) & (xy[0]<=lim) & (xy[1]<=lim)
        ax.plot(xy[0,mask], xy[1,mask], 'k-', lw=0.9, alpha=0.5)

    ax.plot(0, 0, 'ks', markersize=7, zorder=5)
    ax.plot(a1/b1, 0, 'o', color='steelblue', markersize=8, zorder=5)
    ax.plot(0, a2/b2, 'o', color='crimson', markersize=8, zorder=5)

    # Interior equilibrium
    denom = b1*b2 - c1*c2
    if abs(denom) > 1e-6:
        xs = (a1*b2 - c1*a2)/denom
        ys = (a2*b1 - c2*a1)/denom
        if xs > 0 and ys > 0:
            color_int = 'seagreen' if c1*c2 < b1*b2 else 'darkorange'
            ax.plot(xs, ys, '*', color=color_int, markersize=12, zorder=6,
                    label=f'Interior $({xs:.2f},{ys:.2f})$')

    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
    ax.set_title(case['title'], fontsize=10)
    ax.legend(fontsize=8)

plt.suptitle("Competing Species Model", fontsize=12)
plt.tight_layout()
plt.show()

beta, gamma_sir = 0.3, 0.1
N = 1.0
R0 = beta * N / gamma_sir
print(f"R_0 = beta*N/gamma = {R0:.2f}")

def f_sir(S, I): return -beta * S * I
def g_sir(S, I): return beta * S * I - gamma_sir * I

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Phase portrait in (S, I) plane
lim_S, lim_I = 1.05, 0.35
S_g, I_g = np.meshgrid(np.linspace(0, lim_S, 22), np.linspace(0, lim_I, 22))
dS = f_sir(S_g, I_g); dI = g_sir(S_g, I_g)
nrm = np.sqrt(dS**2 + dI**2 + 1e-10)
axes[0].quiver(S_g, I_g, dS/nrm, dI/nrm, alpha=0.25, color='gray', scale=28)

S_thresh = gamma_sir / beta
axes[0].axvline(S_thresh, color='crimson', ls='--', lw=2.0,
                label=rf"$S^*=\gamma/\beta={S_thresh:.2f}$")

colors_ic = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
for I0, color in zip([0.01, 0.03, 0.06, 0.10, 0.15, 0.20], colors_ic):
    S0 = N - I0
    sol = solve_ivp(lambda t, z: [f_sir(z[0],z[1]), g_sir(z[0],z[1])],
                    (0, 200), [S0, I0], dense_output=True,
                    max_step=0.1, rtol=1e-8, atol=1e-10)
    Si = sol.y
    mask = (Si[0]>=0) & (Si[1]>=0) & (Si[0]<=lim_S) & (Si[1]<=lim_I)
    axes[0].plot(Si[0,mask], Si[1,mask], color=color, lw=1.8, alpha=0.85)

axes[0].set_xlim(0, lim_S); axes[0].set_ylim(0, lim_I)
axes[0].set_xlabel('$S$ (susceptible)'); axes[0].set_ylabel('$I$ (infected)')
axes[0].set_title(f'Phase portrait ($R_0={R0:.1f}$)')
axes[0].legend(fontsize=9)

# Time series
I0_ts = 0.01; S0_ts = N - I0_ts; R0_ts = 0.0
sol_ts = solve_ivp(
    lambda t, z: [f_sir(z[0],z[1]), g_sir(z[0],z[1]), gamma_sir*z[1]],
    (0, 120), [S0_ts, I0_ts, R0_ts], dense_output=True, max_step=0.1)
t_plot = np.linspace(0, 120, 600)
vals = sol_ts.sol(t_plot)
axes[1].plot(t_plot, vals[0], color='steelblue', lw=2.5, label='$S(t)$')
axes[1].plot(t_plot, vals[1], color='crimson',   lw=2.5, label='$I(t)$')
axes[1].plot(t_plot, vals[2], color='seagreen',  lw=2.5, label='$R(t)$')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$ (days)'); axes[1].set_ylabel('Fraction of population')
axes[1].set_title(f'Time series ($I_0={I0_ts}$, $R_0={R0:.1f}$)')
axes[1].legend(fontsize=9)

plt.suptitle(rf'SIR Epidemic Model: $\beta={beta}$, $\gamma={gamma_sir}$, $R_0={R0:.1f}$',
             fontsize=12)
plt.tight_layout()
plt.show()

k1, k2 = 0.5, 0.2

def f_ck(x, y): return -k1 * x
def g_ck(x, y): return k1*x - k2*y

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

lim = 1.1
x_g, y_g = np.meshgrid(np.linspace(0, lim, 22), np.linspace(0, lim, 22))
dx = f_ck(x_g, y_g); dy = g_ck(x_g, y_g)
nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
axes[0].quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.25, color='gray', scale=28)

colors_ck = plt.cm.plasma(np.linspace(0.1, 0.85, 6))
for x0, y0, color in zip([1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                          [0.0, 0.2, 0.2, 0.3, 0.4, 0.8], colors_ck):
    sol = solve_ivp(lambda t, z: [f_ck(z[0],z[1]), g_ck(z[0],z[1])],
                    (0, 20), [x0, y0], dense_output=True, max_step=0.1)
    xy = sol.y
    mask = (xy[0]>=0) & (xy[1]>=0) & (xy[0]<=lim) & (xy[1]<=lim)
    axes[0].plot(xy[0,mask], xy[1,mask], color=color, lw=1.8)

axes[0].plot(0, 0, 'o', color='steelblue', markersize=9, zorder=5,
             label='Stable node $(0,0)$')
axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
axes[0].set_xlabel('$x=[A]$'); axes[0].set_ylabel('$y=[B]$')
axes[0].set_title(r'Phase portrait ($A\to B\to C$)')
axes[0].legend(fontsize=9)

# Time series
t_plot = np.linspace(0, 20, 400)
x_t = np.exp(-k1 * t_plot)
y_t = (k1/(k2-k1)) * (np.exp(-k1*t_plot) - np.exp(-k2*t_plot))
z_t = 1 - x_t - y_t
axes[1].plot(t_plot, x_t, color='steelblue', lw=2.5, label='$[A](t)$')
axes[1].plot(t_plot, y_t, color='crimson',   lw=2.5, label='$[B](t)$')
axes[1].plot(t_plot, z_t, color='seagreen',  lw=2.5, ls='--', label='$[C](t)=1-x-y$')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('Concentration')
axes[1].set_title(f'Time series ($k_1={k1}$, $k_2={k2}$)')
axes[1].legend(fontsize=9)

plt.suptitle('Chemical Kinetics: $A \\to B \\to C$', fontsize=12)
plt.tight_layout()
plt.show()

omega0 = 1.0

def f_pend(x, y): return y
def g_pend(x, y): return -omega0**2 * np.sin(x)
def H_pend(x, y): return 0.5*y**2 - omega0**2 * np.cos(x)

fig, ax = plt.subplots(figsize=(10, 6))

x_g = np.linspace(-3.5*np.pi, 3.5*np.pi, 300)
y_g = np.linspace(-3.0, 3.0, 300)
X, Y = np.meshgrid(x_g, y_g)
H_vals = H_pend(X, Y)

# Filled contours for energy level
contour_f = ax.contourf(X, Y, H_vals, levels=40, cmap='RdYlBu_r', alpha=0.35)

# Level curves
levels_osc = np.linspace(-omega0**2 + 0.02, omega0**2 - 0.05, 6)
levels_rot = [omega0**2 + 0.4, omega0**2 + 1.2, omega0**2 + 2.5]
ax.contour(X, Y, H_vals, levels=levels_osc,
           colors='steelblue', linewidths=1.5, alpha=0.85)
ax.contour(X, Y, H_vals, levels=levels_rot,
           colors='seagreen', linewidths=1.5, alpha=0.85)

# Separatrix
ax.contour(X, Y, H_vals, levels=[omega0**2],
           colors='black', linewidths=2.5)

# Critical points
for n in range(-3, 4):
    xe_p = 2*n*np.pi
    ax.plot(xe_p, 0, 'o', color='steelblue', markersize=9, zorder=6)
for n in range(-2, 3):
    xe_p = (2*n+1)*np.pi
    ax.plot(xe_p, 0, 's', color='crimson', markersize=9, zorder=6)

# Legend proxies
ax.plot([], [], 'o', color='steelblue', markersize=9,
        label='Center (pendulum down — oscillation)')
ax.plot([], [], 's', color='crimson', markersize=9,
        label='Saddle (pendulum up — unstable)')
ax.plot([], [], 'k-', lw=2.5, label='Separatrix ($H=\\omega_0^2$)')
ax.plot([], [], color='steelblue', lw=1.5,
        label='Oscillating orbits ($H<\\omega_0^2$)')
ax.plot([], [], color='seagreen', lw=1.5,
        label='Rotating orbits ($H>\\omega_0^2$)')

ax.set_xlim(-3.5*np.pi, 3.5*np.pi)
ax.set_ylim(-3.0, 3.0)
ax.set_xlabel(r'$x = \theta$ (angle, radians)')
ax.set_ylabel(r"$y = \theta'$ (angular velocity)")
ax.set_title(r"Nonlinear Pendulum: $x'=y$, $y'=-\omega_0^2\sin x$ ($\omega_0=1$)", fontsize=11)
ax.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
ax.set_xticklabels([r'$-2\pi$', r'$-\pi$', '$0$', r'$\pi$', r'$2\pi$'])
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.show()

eps = 1.0

def f_vdp(x, y): return y
def g_vdp(x, y): return eps*(1 - x**2)*y - x

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

lim = 3.5
x_g, y_g = np.meshgrid(np.linspace(-lim, lim, 24), np.linspace(-lim, lim, 24))
dx = f_vdp(x_g, y_g); dy = g_vdp(x_g, y_g)
nrm = np.sqrt(dx**2 + dy**2 + 1e-10)
axes[0].quiver(x_g, y_g, dx/nrm, dy/nrm, alpha=0.2, color='gray', scale=30)

# Orbits spiraling out from near origin
for r in [0.2, 0.5, 0.9]:
    sol = solve_ivp(lambda t, z: [f_vdp(z[0],z[1]), g_vdp(z[0],z[1])],
                    (0, 30), [r, 0.0], dense_output=True, max_step=0.05)
    xy = sol.y; mask = np.all(np.abs(xy) < lim+0.5, axis=0)
    axes[0].plot(xy[0,mask], xy[1,mask], color='crimson', lw=1.3, alpha=0.7)

# Orbits spiraling in from outside
for r in [3.0, 2.5]:
    sol = solve_ivp(lambda t, z: [f_vdp(z[0],z[1]), g_vdp(z[0],z[1])],
                    (0, 20), [r, 0.0], dense_output=True, max_step=0.05)
    xy = sol.y; mask = np.all(np.abs(xy) < lim+0.5, axis=0)
    axes[0].plot(xy[0,mask], xy[1,mask], color='steelblue', lw=1.3, alpha=0.7)

# Limit cycle (long-time behavior)
sol_lc = solve_ivp(lambda t, z: [f_vdp(z[0],z[1]), g_vdp(z[0],z[1])],
                   (0, 100), [0.5, 0.0], dense_output=True, max_step=0.02)
xy_lc = sol_lc.y
# Take last two periods approximately
mask_lc = sol_lc.t > 80
axes[0].plot(xy_lc[0, mask_lc], xy_lc[1, mask_lc], 'k-', lw=2.8,
             label='Limit cycle', zorder=5)

axes[0].plot(0, 0, 's', color='darkorange', markersize=9, zorder=6,
             label='Unstable origin ($\\varepsilon>0$)')
axes[0].plot([], [], color='crimson', lw=1.5, label='Outward spiral (from inside)')
axes[0].plot([], [], color='steelblue', lw=1.5, label='Inward spiral (from outside)')
axes[0].set_xlim(-lim, lim); axes[0].set_ylim(-lim, lim)
axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
axes[0].set_title(f'Phase portrait ($\\varepsilon={eps}$)')
axes[0].legend(fontsize=8)
axes[0].axhline(0, color='k', lw=0.4); axes[0].axvline(0, color='k', lw=0.4)

# Time series
t_ts = np.linspace(0, 30, 600)
sol_ts = solve_ivp(lambda t, z: [f_vdp(z[0],z[1]), g_vdp(z[0],z[1])],
                   (0, 30), [0.1, 0.0], dense_output=True, max_step=0.05)
axes[1].plot(t_ts, sol_ts.sol(t_ts)[0], color='steelblue', lw=2.5, label='$x(t)$')
axes[1].plot(t_ts, sol_ts.sol(t_ts)[1], color='crimson', lw=2.0, ls='--',
             label="$y(t) = x'(t)$", alpha=0.8)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('$t$'); axes[1].set_ylabel('$x,\\ y$')
axes[1].set_title(f'Time series ($\\varepsilon={eps}$, $x(0)=0.1$)')
axes[1].legend(fontsize=9)

plt.suptitle(f'Van der Pol Oscillator ($\\varepsilon={eps}$)', fontsize=12)
plt.tight_layout()
plt.show()