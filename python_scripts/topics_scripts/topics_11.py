import numpy as np
import sympy as sym
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from IPython.display import Math, display
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

examples = [
    ("Steel bearing, still air",      0.50, 200.0, 25.0, 15.0, 0.05, 500.0),
    ("Coffee cup, still air",         0.30,  90.0, 22.0,  8.0, 0.02, 4200.0),
    ("CPU heat sink, forced air",     0.05,  85.0, 25.0, 40.0, 0.01, 900.0),
]
# columns: label, m (kg), T0 (C), Tenv (C), h (W/m2K), A (m^2), c (J/kgK)

t_plot = np.linspace(0, 1800, 500)
colors = ['steelblue', 'darkorange', 'crimson']

fig, ax = plt.subplots(figsize=(8, 4.5))
for (lbl, m, T0, Tenv, h, A, c), color in zip(examples, colors):
    k = h*A/(m*c)
    tau = 1/k
    T = Tenv + (T0 - Tenv)*np.exp(-k*t_plot)
    ax.plot(t_plot, T, color=color, lw=2.2, label=f'{lbl} ($\\tau={tau:.0f}$ s)')
    ax.axhline(Tenv, color='gray', lw=0.5, ls=':')

ax.set_xlabel('$t$ (s)'); ax.set_ylabel('Temperature (°C)')
ax.set_title("Newton's law of cooling: $T(t) = T_{env} + (T_0-T_{env})e^{-t/\\tau}$")
ax.legend(fontsize=8.5)
plt.tight_layout()
plt.show()

tau = 1.0
t_plot = np.linspace(0, 6, 400)
resp = 1 - np.exp(-t_plot/tau)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(t_plot, resp, color='steelblue', lw=2.5, label='$(T-T_0)/(T_{env}-T_0)$')
axes[0].axhline(1.0, color='k', ls='--', lw=1)
axes[0].axvline(1.0, color='crimson', ls=':', lw=1.3, label=r'$t=\tau$ (63.2\%)')
axes[0].set_xlabel('$t/\\tau$'); axes[0].set_ylabel('Normalized temperature')
axes[0].set_title(r'Thermal step response: $mc\,T\prime + hA\,T = hA\,T_{env}$')
axes[0].legend(fontsize=9); axes[0].set_ylim(0, 1.15)

axes[1].plot(t_plot, resp, color='darkorange', lw=2.5, ls='--', label='$V_C/E_0$')
axes[1].axhline(1.0, color='k', ls='--', lw=1)
axes[1].axvline(1.0, color='crimson', ls=':', lw=1.3, label=r'$t=\tau$ (63.2\%)')
axes[1].set_xlabel('$t/\\tau$'); axes[1].set_ylabel('Normalized voltage')
axes[1].set_title(r'Electrical step response: $RC\,V_C\prime + V_C = E_0$')
axes[1].legend(fontsize=9); axes[1].set_ylim(0, 1.15)

plt.suptitle('Identical mathematics: thermal vs. electrical first-order response', fontsize=11)
plt.tight_layout()
plt.show()

k1env, k2env, k12 = 0.02, 0.08, 0.15   # 1/s
Tenv = 25.0

A_mat = np.array([[-(k1env+k12),  k12],
                   [k12,          -(k2env+k12)]])
b_vec = np.array([k1env*Tenv, k2env*Tenv])

eigvals, eigvecs = np.linalg.eig(A_mat)
print("Eigenvalues (1/s):", eigvals)
print("Time constants (s):", -1/eigvals)

def rhs(t, T):
    return A_mat @ T + b_vec

T0 = np.array([90.0, 25.0])
t_plot = np.linspace(0, 120, 500)
sol = solve_ivp(rhs, (0, 120), T0, t_eval=t_plot, max_step=0.1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(sol.t, sol.y[0], color='steelblue', lw=2.2, label='$T_1$ (chip)')
axes[0].plot(sol.t, sol.y[1], color='darkorange', lw=2.2, label='$T_2$ (heat sink)')
axes[0].axhline(Tenv, color='gray', ls=':', lw=1, label='$T_{env}=25°C$')
axes[0].set_xlabel('$t$ (s)'); axes[0].set_ylabel('Temperature (°C)')
axes[0].set_title('Two-node system: temperature histories')
axes[0].legend(fontsize=8.5)

axes[1].plot(sol.y[0], sol.y[1], color='seagreen', lw=2.2)
axes[1].plot(T0[0], T0[1], 'ko', markersize=7, label='Start', zorder=5)
axes[1].plot(Tenv, Tenv, 'g*', markersize=12, label='Equilibrium', zorder=5)
lims = [20, 95]
axes[1].plot(lims, lims, color='gray', ls='--', lw=1, label='$T_1=T_2$')
axes[1].set_xlabel('$T_1$ (°C)'); axes[1].set_ylabel('$T_2$ (°C)')
axes[1].set_title('Phase portrait: fast mode then slow mode')
axes[1].legend(fontsize=8.5); axes[1].set_xlim(lims); axes[1].set_ylim(lims)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()

k = 2*np.pi/48      # 1/hr; thermal time constant tau = 1/k ~ 7.64 hr
T0mean = 10.0        # deg C, mean ambient
A0 = 8.0             # deg C, ambient swing amplitude
omega = 2*np.pi/24   # 1/hr, 24-hour period

def rhs(t, T):
    Tenv = T0mean + A0*np.sin(omega*t)
    return -k*(T - Tenv)

t_plot = np.linspace(0, 96, 700)
sol = solve_ivp(rhs, (0, 96), [T0mean], t_eval=t_plot, max_step=0.05)

G = 1/np.sqrt(1 + (omega/k)**2)
phi = np.arctan(omega/k)
print(f"Amplitude ratio G = {G:.3f},  phase lag = {phi:.3f} rad = {phi/omega:.2f} hr")

Tenv_plot = T0mean + A0*np.sin(omega*t_plot)
T_ss = T0mean + A0*G*np.sin(omega*t_plot - phi)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_plot, Tenv_plot, color='darkorange', lw=2, label='Ambient (outdoor) temperature')
ax.plot(sol.t, sol.y[0], color='steelblue', lw=2.2, label='Interior temperature (numerical)')
ax.plot(t_plot, T_ss, color='crimson', lw=1.5, ls='--',
        label=f'Steady-state prediction ($G={G:.2f}$, lag$={phi/omega:.1f}$ hr)')
ax.set_xlabel('$t$ (hours)'); ax.set_ylabel('Temperature (°C)')
ax.set_title('Periodic ambient forcing: amplitude attenuation and phase lag')
ax.legend(fontsize=8.5)
ax.set_xticks([0, 24, 48, 72, 96])
plt.tight_layout()
plt.show()

sigma_SB = 5.670374419e-8
eps = 0.9
A_area = 0.02   # m^2
m_mass = 0.5    # kg
c_spec = 500.0  # J/(kg K)
Tenv_K = 300.0  # K
T0_K   = 1000.0 # K

a_rad = eps*sigma_SB*A_area/(m_mass*c_spec)

def rad_rhs(t, T):
    return -a_rad*(T**4 - Tenv_K**4)

t_plot = np.linspace(0, 20000, 2000)
sol = solve_ivp(rad_rhs, (0, 20000), [T0_K], t_eval=t_plot, max_step=5.0, rtol=1e-10, atol=1e-8)
T_num = sol.y[0]

b_K = Tenv_K
def F_implicit(T):
    return (1/(4*b_K**3))*np.log(np.abs((T-b_K)/(T+b_K))) - (1/(2*b_K**3))*np.arctan(T/b_K)

C0 = F_implicit(T0_K)
lhs_check = F_implicit(T_num) - C0

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(t_plot, T_num, color='crimson', lw=2.2)
axes[0].axhline(Tenv_K, color='gray', ls=':', lw=1, label='$T_{env}=300$ K')
axes[0].set_xlabel('$t$ (s)'); axes[0].set_ylabel('$T(t)$ (K)')
axes[0].set_title('Radiative cooling (nonlinear ODE)')
axes[0].legend(fontsize=8.5)

axes[1].plot(t_plot, lhs_check, color='steelblue', lw=2.2, label='$F(T(t))-F(T_0)$')
axes[1].plot(t_plot, -a_rad*t_plot, color='k', ls='--', lw=1.3, label='$-at$ (predicted)')
axes[1].set_xlabel('$t$ (s)'); axes[1].set_ylabel('Implicit relation LHS')
axes[1].set_title('Verifying the implicit closed-form solution')
axes[1].legend(fontsize=8.5)

plt.tight_layout()
plt.show()

sigma_SB = 5.670374419e-8
eps_c   = 0.85
h_c     = 12.0     # W/m2K
A_c     = 0.03     # m^2
m_c     = 0.4      # kg
c_c     = 460.0    # J/kgK
Tenv_c  = 293.0    # K
T0_c    = 900.0    # K

def f_comb(t, T):
    return -(h_c*A_c/(m_c*c_c))*(T-Tenv_c) - (eps_c*sigma_SB*A_c/(m_c*c_c))*(T**4 - Tenv_c**4)

def euler_step(f, T0, t0, t_end, dt):
    n = int(round((t_end-t0)/dt))
    t, T = t0, T0
    for _ in range(n):
        T = T + dt*f(t, T); t += dt
    return T

def heun_step(f, T0, t0, t_end, dt):
    n = int(round((t_end-t0)/dt))
    t, T = t0, T0
    for _ in range(n):
        k1 = f(t, T)
        k2 = f(t+dt, T+dt*k1)
        T = T + dt/2*(k1+k2); t += dt
    return T

def rk4_step(f, T0, t0, t_end, dt):
    n = int(round((t_end-t0)/dt))
    t, T = t0, T0
    for _ in range(n):
        k1 = f(t, T)
        k2 = f(t+dt/2, T+dt/2*k1)
        k3 = f(t+dt/2, T+dt/2*k2)
        k4 = f(t+dt, T+dt*k3)
        T = T + dt/6*(k1+2*k2+2*k3+k4); t += dt
    return T

# Reference solution (very fine tolerance)
t_check = 200.0
ref = solve_ivp(f_comb, (0, t_check), [T0_c], max_step=0.01, rtol=1e-13, atol=1e-12, dense_output=True)
T_ref = ref.sol(t_check)[0]

t_plot = np.linspace(0, 3000, 600)
ref_full = solve_ivp(f_comb, (0, 3000), [T0_c], t_eval=t_plot, max_step=0.1)

dt_values = np.array([20, 10, 5, 2.5, 1.25])
errors = {'Euler': [], 'Heun': [], 'RK4': []}
for dt in dt_values:
    errors['Euler'].append(abs(euler_step(f_comb, T0_c, 0, t_check, dt) - T_ref))
    errors['Heun'].append(abs(heun_step(f_comb, T0_c, 0, t_check, dt) - T_ref))
    errors['RK4'].append(abs(rk4_step(f_comb, T0_c, 0, t_check, dt) - T_ref))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(ref_full.t, ref_full.y[0], color='crimson', lw=2.2)
axes[0].axhline(Tenv_c, color='gray', ls=':', lw=1, label='$T_{env}=293$ K')
axes[0].axvline(t_check, color='k', ls='--', lw=1, label='$t=200$ s (comparison point)')
axes[0].set_xlabel('$t$ (s)'); axes[0].set_ylabel('$T(t)$ (K)')
axes[0].set_title('Combined convection + radiation cooling')
axes[0].legend(fontsize=8.5)

colors_m = {'Euler':'steelblue', 'Heun':'darkorange', 'RK4':'seagreen'}
for name in errors:
    axes[1].loglog(dt_values, errors[name], 'o-', color=colors_m[name], lw=2, label=name)

# Reference slope lines
dt_ref = dt_values
for order, style, lbl in [(1, ':', '$O(\\Delta t)$'), (2, '--', '$O(\\Delta t^2)$'), (4, '-.', '$O(\\Delta t^4)$')]:
    scale = errors['Euler'][-1] / dt_ref[-1]**1 if order == 1 else \
            errors['Heun'][-1] / dt_ref[-1]**2 if order == 2 else \
            errors['RK4'][-1] / dt_ref[-1]**4
    axes[1].loglog(dt_ref, scale*dt_ref**order, color='gray', ls=style, lw=1, label=lbl)

axes[1].set_xlabel('Step size $\\Delta t$ (s)'); axes[1].set_ylabel('Error in $T$ at $t=200$ s (K)')
axes[1].set_title('Convergence: Euler $O(\\Delta t)$, Heun $O(\\Delta t^2)$, RK4 $O(\\Delta t^4)$')
axes[1].legend(fontsize=7.5)

plt.tight_layout()
plt.show()

R_gas, cv_gas = 287.0, 718.0     # air, J/(kg K)
m_gas = 0.002                    # kg
A_wall = 0.01                    # m^2
Tenv_gas = 300.0                 # K
V0_gas, v_pist = 0.0005, 2.0e-5  # m^3, m^3/s
T0_gas = 500.0                   # K
t_end_gas = 20.0

def V_of_t(t): return V0_gas + v_pist*t

def make_rhs(h_val):
    def rhs(t, T):
        Vt = V_of_t(t)
        P = m_gas*R_gas*T/Vt
        dU_from_Q = -h_val*A_wall*(T-Tenv_gas)
        dW = P*v_pist
        return (dU_from_Q - dW)/(m_gas*cv_gas)
    return rhs

gamma_gas = (cv_gas+R_gas)/cv_gas
t_plot = np.linspace(0, t_end_gas, 400)
V_plot = V_of_t(t_plot)
T_adiabatic = T0_gas*(V0_gas/V_plot)**(gamma_gas-1)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(V_plot*1e6, T_adiabatic, color='k', ls='--', lw=1.8, label=r'Adiabatic reversible ($h=0$ analytic): $TV^{\gamma-1}=$const')

for h_val, color in zip([0.0, 2.0, 8.0, 20.0], plt.cm.plasma(np.linspace(0.15, 0.85, 4))):
    sol = solve_ivp(make_rhs(h_val), (0, t_end_gas), [T0_gas], t_eval=t_plot, max_step=0.01)
    ax.plot(V_plot*1e6, sol.y[0], color=color, lw=2, label=f'$h={h_val:.0f}$ W/m$^2$K')

ax.axhline(Tenv_gas, color='gray', ls=':', lw=1, label='$T_{env}=300$ K (isothermal limit)')
ax.set_xlabel('Volume $V$ (cm$^3$)'); ax.set_ylabel('Temperature $T$ (K)')
ax.set_title('Piston–cylinder expansion: adiabatic to isothermal, via Newton cooling')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()